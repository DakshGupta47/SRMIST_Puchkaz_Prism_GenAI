"""
validate_results.py

Validates every line in results.jsonl against:
  1. Valid JSON format and top-level response envelope.
  2. src/schema.py Pydantic schema (ContextDeeplinkResponse, Goal, Action, StepGroup, Deeplink).
  3. src/scrubber.py rules:
     - Zero-URL-leak: No web URLs (http://, https://, etc.), www., bare domains,
       or email-like strings in any user-visible text field.
     - Deeplink schemes: Deeplink URIs permitted ONLY in deeplink fields;
       must not leak into user-visible text fields.
     - Web URLs rejected in deeplink fields (deeplink must be valid non-web scheme).
     - Field validation:
         * goal matches GOAL_PATTERN r"^Follow these steps to perform this .+ (Troubleshooting|Configuration)$"
         * title is 2-3 words
         * description starts with "It will" and has 5-7 words
         * auto actions possess a valid non-empty deeplink
         * empty contexts carries fallback ("no_match" or "no_siis_context")

Usage:
  python scripts/validate_results.py [--input results.jsonl]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pydantic import ValidationError  # noqa: E402

try:
    from src.schema import ContextDeeplinkResponse, actionCategory
    from src.scrubber import (
        GOAL_PATTERN,
        _DOMAIN_RE,
        _EMAIL_RE,
        _SCHEME_RE,
        _WWW_RE,
        _is_valid_description,
        _is_valid_goal,
        _is_valid_title,
        _scrub_deeplink,
        _scrub_text,
        _word_count,
    )
except ImportError:
    from schema import ContextDeeplinkResponse, actionCategory  # type: ignore[no-redef]  # noqa: E402
    from scrubber import (  # type: ignore[no-redef]  # noqa: E402
        GOAL_PATTERN,
        _DOMAIN_RE,
        _EMAIL_RE,
        _SCHEME_RE,
        _WWW_RE,
        _is_valid_description,
        _is_valid_goal,
        _is_valid_title,
        _scrub_deeplink,
        _scrub_text,
        _word_count,
    )


logger = logging.getLogger("validate_results")


def validate_record(record: dict[str, Any], line_num: int = 1) -> list[str]:
    """Validate a single parsed results.jsonl record dictionary.

    Returns a list of error strings (empty if valid).
    """
    errors: list[str] = []

    if not isinstance(record, dict):
        return [f"Line {line_num}: Record is not a JSON object"]

    if "query" not in record or not isinstance(record["query"], str):
        errors.append(f"Line {line_num}: Missing or invalid 'query' field")

    if "response" not in record or not isinstance(record["response"], dict):
        errors.append(f"Line {line_num}: Missing or invalid 'response' object")
        return errors

    resp = record["response"]

    # 1. Validate against Pydantic schema in src/schema.py
    try:
        model_obj = ContextDeeplinkResponse.model_validate(resp)
    except ValidationError as val_err:
        errors.append(f"Line {line_num}: Schema validation error: {val_err}")
        return errors

    contexts = model_obj.contexts

    # 2. Check fallback rules
    if not contexts:
        fallback = resp.get("fallback")
        if fallback not in ("no_match", "no_siis_context"):
            errors.append(
                f"Line {line_num}: Empty contexts must have fallback 'no_match' or 'no_siis_context', got: {fallback!r}"
            )
    else:
        if "fallback" in resp and resp["fallback"] is not None:
            errors.append(
                f"Line {line_num}: Non-empty contexts must not have fallback field, got: {resp['fallback']!r}"
            )

    # 3. Check scrubber and schema rules on each Goal in contexts
    for g_idx, goal in enumerate(contexts):
        # Goal string
        if not _is_valid_goal(goal.goal):
            errors.append(f"Line {line_num} Goal {g_idx+1}: Invalid goal format: {goal.goal!r}")
        cleaned_goal, mod_goal = _scrub_text(goal.goal)
        if mod_goal:
            errors.append(f"Line {line_num} Goal {g_idx+1}: Goal contains URL, email, or domain leak: {goal.goal!r}")

        # Title string
        if not _is_valid_title(goal.title):
            errors.append(
                f"Line {line_num} Goal {g_idx+1}: Title must be 2-3 words, got '{goal.title}' ({_word_count(goal.title)} words)"
            )
        cleaned_title, mod_title = _scrub_text(goal.title)
        if mod_title:
            errors.append(f"Line {line_num} Goal {g_idx+1}: Title contains URL, email, or domain leak: {goal.title!r}")

        # Actions
        for a_idx, action in enumerate(goal.actions):
            # actionName
            if not action.actionName or not action.actionName.strip():
                errors.append(f"Line {line_num} Goal {g_idx+1} Action {a_idx+1}: Empty actionName")
            _, mod_act_name = _scrub_text(action.actionName)
            if mod_act_name:
                errors.append(
                    f"Line {line_num} Goal {g_idx+1} Action {a_idx+1}: actionName contains URL leak: {action.actionName!r}"
                )

            # description
            if not _is_valid_description(action.description):
                errors.append(
                    f"Line {line_num} Goal {g_idx+1} Action {a_idx+1}: Description must start with 'It will' and be 5-7 words, got: {action.description!r}"
                )
            _, mod_desc = _scrub_text(action.description)
            if mod_desc:
                errors.append(
                    f"Line {line_num} Goal {g_idx+1} Action {a_idx+1}: Description contains URL leak: {action.description!r}"
                )

            # Category auto downgrade check
            if action.category == actionCategory.auto:
                has_act_deeplink = any(
                    sg.actionableDeeplink and sg.actionableDeeplink.deeplink and sg.actionableDeeplink.deeplink.strip()
                    for sg in action.stepGroups
                )
                if not has_act_deeplink:
                    errors.append(
                        f"Line {line_num} Goal {g_idx+1} Action {a_idx+1}: Action category is 'auto' but no valid actionableDeeplink exists"
                    )

            # StepGroups
            for sg_idx, sg in enumerate(action.stepGroups):
                for s_idx, step in enumerate(sg.steps):
                    _, mod_step = _scrub_text(step)
                    if mod_step:
                        errors.append(
                            f"Line {line_num} Goal {g_idx+1} Action {a_idx+1} StepGroup {sg_idx+1} Step {s_idx+1}: Step contains URL leak: {step!r}"
                        )

                # actionableDeeplink
                if sg.actionableDeeplink:
                    dl = sg.actionableDeeplink.deeplink
                    dl_clean, dl_cleared = _scrub_deeplink(dl)
                    if dl_cleared or not dl_clean:
                        errors.append(
                            f"Line {line_num} Goal {g_idx+1} Action {a_idx+1} StepGroup {sg_idx+1}: Invalid or disallowed actionableDeeplink: {dl!r}"
                        )
                    if sg.actionableDeeplink.description:
                        _, mod_dl_desc = _scrub_text(sg.actionableDeeplink.description)
                        if mod_dl_desc:
                            errors.append(
                                f"Line {line_num} Goal {g_idx+1} Action {a_idx+1} StepGroup {sg_idx+1}: actionableDeeplink description contains URL leak: {sg.actionableDeeplink.description!r}"
                            )
                    if sg.actionableDeeplink.message:
                        _, mod_dl_msg = _scrub_text(sg.actionableDeeplink.message)
                        if mod_dl_msg:
                            errors.append(
                                f"Line {line_num} Goal {g_idx+1} Action {a_idx+1} StepGroup {sg_idx+1}: actionableDeeplink message contains URL leak: {sg.actionableDeeplink.message!r}"
                            )

                # validationDeeplink
                if sg.validationDeeplink:
                    vdl = sg.validationDeeplink.deeplink
                    vdl_clean, vdl_cleared = _scrub_deeplink(vdl)
                    if vdl_cleared or not vdl_clean:
                        errors.append(
                            f"Line {line_num} Goal {g_idx+1} Action {a_idx+1} StepGroup {sg_idx+1}: Invalid or disallowed validationDeeplink: {vdl!r}"
                        )
                    if sg.validationDeeplink.key:
                        _, mod_k = _scrub_text(sg.validationDeeplink.key)
                        if mod_k:
                            errors.append(
                                f"Line {line_num} Goal {g_idx+1} Action {a_idx+1} StepGroup {sg_idx+1}: validationDeeplink key contains URL leak: {sg.validationDeeplink.key!r}"
                            )

    return errors


def validate_file(path: Path) -> tuple[int, int, list[str]]:
    """Validate all lines of a results.jsonl file.

    Returns (total_lines, passed_count, all_errors).
    """
    if not path.is_file():
        return 0, 0, [f"File not found: {path}"]

    lines = path.read_text(encoding="utf-8").splitlines()
    total_lines = len(lines)
    all_errors: list[str] = []
    passed_count = 0

    if total_lines == 0:
        return 0, 0, [f"File is empty: {path}"]

    for line_num, line_str in enumerate(lines, start=1):
        line_str = line_str.strip()
        if not line_str:
            all_errors.append(f"Line {line_num}: Empty line encountered")
            continue

        try:
            record = json.loads(line_str)
        except Exception as exc:
            all_errors.append(f"Line {line_num}: JSON decode error: {exc}")
            continue

        row_errors = validate_record(record, line_num=line_num)
        if row_errors:
            all_errors.extend(row_errors)
        else:
            passed_count += 1

    return total_lines, passed_count, all_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate results.jsonl against schema and scrubber rules")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT_DIR / "results.jsonl",
        help="Path to results.jsonl file to validate",
    )
    args = parser.parse_args()

    total, passed, errors = validate_file(args.input)

    print("\n" + "=" * 60)
    print("RESULTS VALIDATION SUMMARY")
    print("=" * 60)
    print(f"File tested          : {args.input.name}")
    print(f"Total lines tested   : {total}")
    print(f"Passed               : {passed}")
    print(f"Failed               : {total - passed}")
    print(f"Status               : {'PASSED (All valid)' if not errors else 'FAILED'}")
    print("=" * 60)

    if errors:
        print("\nErrors encountered:")
        for err in errors[:25]:
            print(f"  - {err}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more errors")
        sys.exit(1)
    else:
        print("\nAll lines strictly comply with src/schema.py and scrubber rules.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
