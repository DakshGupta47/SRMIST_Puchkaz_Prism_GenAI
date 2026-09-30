# AI Usage DISCLOSURE FORM

## 1. Team Details
**Team Name:** SRMIST_Puchkaz
**Project / Product Name:** Smart Guided Troubleshooting Engine (Theme 2)
**Organization / Institution (if any):** SRMIST KTR
**Submission Date:** 30/09/2026

## 2. AI Usage Declaration
**Did your team use any Artificial Intelligence (AI) in developing this project?**
Yes

## 3. Purpose of AI Usage
- **Idea generation / brainstorming:** No
- **Code generation or assistance:** Yes (Regex optimization, TF-IDF algorithm tuning, Dockerization)
- **UI / UX design:** No
- **Content creation:** No
- **Data analysis:** Yes (Parsing and restructuring the SIIS mock dataset)
- **Testing / debugging:** Yes (Generating Pytest edge-case suites and debugging concurrency locks)
- **Other:** N/A

## 4. Feature Origin Classification

**1. Feature Name:** Stage 0 Enrichment & Zero-Hallucination Gate
**2. Self-Generated/AI-Generated/Both:** Both
**3. Description:** **Tool Used:** Google Gemini. **Prompt used:** "Write Python regex to extract device models and symptoms from messy text, and generate edge-case Pytest functions." **Output Summary:** Generated strict Regular Expressions (_DEVICE_RE and _SYMPTOM_RE) and edge-case unit tests (e.g., handling "aaaaaaarrrgghh"). **Modification:** We manually integrated and modified the validation logic into our pipeline to trigger the unclassified_issue fallback.

**1. Feature Name:** Semantic Deeplink Mapping (TF-IDF Penalty Logic)
**2. Self-Generated/AI-Generated/Both:** Both
**3. Description:** **Tool Used:** Google Gemini. **Prompt used:** "How do I prevent TF-IDF from confusing 'Auto-Sync' and 'Backup' when they share generic keywords?" **Output Summary:** Suggested a dual-layer multiplier (0.5x penalty / 1.5x boost) based on strict action verb matching. **Modification:** We manually adapted and wrote the final implementation to fit our specific device catalog schema and integration pipeline.

**1. Feature Name:** Concurrency Locks & Deployment Setup
**2. Self-Generated/AI-Generated/Both:** AI-Generated
**3. Description:** **Tool Used:** Google Gemini. **Prompt used:** "Create a Dockerfile and docker-compose.yml for this FastAPI app, and make my semantic cache thread-safe." **Output Summary:** Generated the Docker deployment files and Python 	hreading.Lock mechanisms. **Modification:** Minor path adjustments for deployment; used as generated to ensure the Semantic Cache remained memory-safe during high-concurrency testing.

## 5. Ethical & Compliance Confirmation
**AI usage complies with guidelines and policies:** Yes
**No proprietary or copyrighted data misused:** I Agree

## 6. Declaration & Sign-Off
**Name of Team Representative:** PRISHA AGARWAL
**Role:** Team Leader / Developer
**Signature:** *Prisha Agarwal*
**Date:** 30/09/2026
