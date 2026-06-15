# AI-recommendation-system

# Intelligent Candidate Discovery & Ranking
### Redrob × Hack2Skill — India.Runs Hackathon

> **Rank 100,000 candidates for a Senior AI Engineer role the way a great recruiter would — not by matching keywords, but by understanding who genuinely fits.**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Problem Statement](#problem-statement)
3. [Solution Overview](#solution-overview)
4. [Scoring Architecture](#scoring-architecture)
5. [Component Details](#component-details)
6. [Anti-Trap Layer](#anti-trap-layer)
7. [Behavioral Multiplier](#behavioral-multiplier)
8. [Technologies Used](#technologies-used)
9. [Results](#results)
10. [File Structure](#file-structure)
11. [How to Reproduce](#how-to-reproduce)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
cd redrob-ranker

# Place the candidate data file in the same directory
cp /path/to/candidates.jsonl .

# Run the ranker — no pip install needed
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate the output format
python3 validate_submission.py submission.csv
# Expected: "Submission is valid."
```

**Runtime:** ~55 seconds · CPU only · No GPU · No network · No pip install · ~700 MB RAM peak

---

## Problem Statement

Recruiters reviewing hundreds of profiles still miss the right person — not because the talent is absent, but because keyword filters cannot see what actually matters. A Marketing Manager who lists "Pinecone" and "RAG" as skills ranks above a genuine NLP Engineer who built a retrieval system at Flipkart but never used those exact words. A skill claimed for one month counts equally to one used for five years.

The challenge: build a system that ranks 100,000 candidates for a **Senior AI Engineer** role the way an experienced recruiter would — by understanding career trajectory, verifying skill evidence, and checking real availability.

---

## Solution Overview

A **multi-component weighted scoring engine** that processes all 100,000 candidates in a single pass and produces a ranked shortlist of 100 candidates with human-readable reasoning for each.

The core formula:

```
Final Score = (Career×0.35 + Skills×0.30 + Experience×0.10 + Location×0.05 + Education×0.05)
              × Required Skill Gate [0.30 – 1.04]
              × Behavioral Availability Multiplier [0.35 – 1.20]
```

Three principles drive the design:

**1. Demonstrated evidence over claimed skills.**
Career titles and role descriptions are harder to fake than a skills list. The system weights career history at 35% — the single largest component — because a person who has spent four years building ML systems at a product company is more credible than one who listed twelve AI tools on their profile.

**2. Trust-weighted skills over raw keyword matching.**
Each skill is scored as `importance × proficiency × trust_factor`. The trust factor is derived from `duration_months` and `endorsements`. A skill with zero months and zero endorsements receives `trust = 0.08` — near zero — making keyword stuffing effectively useless without any blacklist.

**3. Availability multiplier, not additive.**
An inactive candidate with a perfect resume is a failed hire. A multiplicative behavioral signal ensures that even top-scoring profiles are down-weighted if the person is not actually reachable or available.

---

## Scoring Architecture

### Component Weights

| Component | Weight | What it captures |
|---|---|---|
| Career history | 35% | ML/AI titles, product vs. consulting firms, production evidence in descriptions, GitHub proxy, company quality |
| Skills match | 30% | 80+ JD-derived tokens weighted by proficiency and trust factor |
| Experience years | 10% | 5–9 year optimal band with graceful falloff |
| Location | 5% | India target cities preferred; international penalised |
| Education | 5% | Institution tier × field of study alignment |

### Multipliers Applied After Weighted Sum

| Multiplier | Range | What it does |
|---|---|---|
| Required Skill Gate | 0.30 – 1.04 | Checks coverage of 4 critical categories |
| Behavioral Multiplier | 0.35 – 1.20 | Adjusts for real-world availability |

---

## Component Details

### 1. Career History (35%)

The most predictive signal. Scored across four sub-components:

**Title match.**
Each job title in the candidate's career history is matched against a scored taxonomy:

| Title pattern | Score |
|---|---|
| Senior ML / AI / NLP Engineer | 10 |
| Applied Scientist, Applied ML/AI | 9 |
| Search / Relevance / Ranking Engineer | 9–10 |
| Recommendation Systems Engineer | 9 |
| Senior Data Scientist | 8 |
| Data Scientist | 7 |
| Research Engineer | 7 |
| Software / Backend Engineer | 3–4 |
| Marketing / HR / Finance | 0 |

**Company quality multiplier.**
Applied to the raw career score before weighted combination:

| Company tier | Examples | Multiplier |
|---|---|---|
| Tier 1 — FAANG + frontier AI | Google, Meta, Apple, Netflix, Microsoft, OpenAI | ×1.14 |
| Tier 2 — Top Indian product companies | Zomato, Swiggy, Flipkart, CRED, Razorpay, Paytm | ×1.07 |
| Tier 3 — Mid-tier AI/product companies | PolicyBazaar, MakeMyTrip, Freshworks, Zoho | ×1.04 |
| Default | All others | ×1.00 |

**Production evidence in career descriptions.**
Fourteen regex phrase patterns scan role descriptions for evidence of real production ML work:

```
"built [ranking/search/recommendation] system"   → 3.0 pts
"deployed [model/embedding/pipeline] to production" → 2.5 pts
"improved [NDCG/recall/precision/MRR] by [X]%"     → 2.5 pts
"[N] million/billion users/requests"               → 2.0 pts
"hybrid search / dense+sparse retrieval"           → 2.0 pts
"two-stage / multi-stage retrieval / re-ranking"   → 2.0 pts
"A/B test / online experiment"                     → 1.5 pts
"offline/online evaluation / evaluation framework" → 1.0 pts
```

Candidates whose descriptions mention specific systems built and metrics improved score significantly higher than those with generic responsibility lists.

**GitHub engineering proxy.**
GitHub activity score is folded into the career score because this is a hands-on engineering role:

| GitHub score | Career bonus |
|---|---|
| ≥ 80 | +12 pts |
| ≥ 60 | +6 pts |
| ≥ 40 | +2 pts |
| −1 (not linked) | −3 pts |

**Consulting career penalty.**
If more than 85% of a candidate's career was spent at firms like TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, or HCL, the career score is multiplied by 0.20. This reflects the JD's explicit note that services backgrounds rarely produce production-grade ML engineers.

**Title-chasing penalty.**
Three or more roles with tenure under 18 months triggers a cumulative deduction of 4 points per short-tenure role.

---

### 2. Skills Match (30%)

#### Three-tier skill taxonomy

**Tier A — Critical (directly required by JD)**

Embeddings and retrieval: `sentence-transformers`, `bge`, `e5`, `text embeddings`, `vector embeddings`, `embeddings`

Vector databases: `pinecone`, `weaviate`, `qdrant`, `milvus`, `faiss`, `pgvector`, `opensearch`, `elasticsearch`, `vector database`, `vector search`, `hybrid search`, `ann`

Retrieval and ranking: `bm25`, `semantic search`, `dense retrieval`, `information retrieval`, `colbert`, `learning to rank`, `reranking`, `cross-encoder`, `bi-encoder`

Evaluation: `ndcg`, `mrr`, `a/b testing`, `offline evaluation`, `evaluation framework`

Other critical: `rag`, `retrieval augmented generation`, `python`

**Tier B — Strong (highly relevant)**

`nlp`, `transformers`, `huggingface`, `bert`, `llm`, `fine-tuning`, `lora`, `qlora`, `peft`, `pytorch`, `tensorflow`, `scikit-learn`, `recommendation systems`, `mlops`, `model serving`, `xgboost`

**Tier C — Nice to have**

`docker`, `kubernetes`, `fastapi`, `aws`, `gcp`, `azure`, `ray`, `mlflow`

Tier B skills receive a 0.80× discount relative to Tier A. Tier C skills receive a 0.55× discount.

#### Trust factor per skill

```
if duration_months > 0 and endorsements > 0:
    trust = 0.48 + 0.30 × min(duration_months/24, 1.0) + min(endorsements/80, 0.25)
    # trust range: 0.48 to 1.00

elif duration_months == 0 and endorsements > 0:
    trust = 0.28 + min(endorsements/80, 0.25)
    # trust range: 0.28 to 0.53

elif duration_months == 0 and endorsements == 0:
    trust = 0.08   # keyword stuffer weight
```

If the candidate's career descriptions contain ML vocabulary, all skill trust scores receive an additional ×1.15 boost.

#### Keyword stuffer penalty

More than 3 zero-evidence skills (duration=0, endorsements=0) triggers a compounding penalty on the total skills score: `score × (1 − 0.08 × (zero_count − 3))`, capped at 0.60× maximum reduction.

#### CV/Speech exclusion

Skills indicating computer vision, speech recognition, or robotics focus (`image classification`, `asr`, `tts`, `wav2vec`, `robotics`, `object detection`, etc.) are excluded from scoring entirely and logged as misalignment signals, reflecting the JD's explicit note that these backgrounds are not a fit.

---

### 3. Required Skill Gate (multiplier)

Four categories must be present across skills AND career descriptions combined:

| Category | Example tokens |
|---|---|
| Python | `python` |
| Embeddings | `embed`, `sentence-transform`, `bge`, `text embed`, `vector embed` |
| Vector search | `pinecone`, `weaviate`, `qdrant`, `milvus`, `faiss`, `vector search`, `ann` |
| Retrieval / ranking | `retrieval`, `ranking`, `bm25`, `hybrid search`, `learning to rank`, `ndcg`, `rerank` |

| Categories covered | Multiplier |
|---|---|
| 4 of 4 | ×1.04 |
| 3 of 4 | ×1.00 |
| 2 of 4 | ×0.80 |
| 1 of 4 | ×0.55 |
| 0 of 4 | ×0.30 |

---

### 4. Experience Years (10%)

| Years of experience | Score |
|---|---|
| 5.0 – 9.0 | 100 |
| 4.0 – 5.0 or 9.0 – 11.0 | 86 |
| 11.0 – 14.0 | 72 |
| 3.0 – 4.0 | 66 |
| > 14.0 | 62 |
| 2.0 – 3.0 | 45 |
| < 2.0 | 20 |

---

### 5. Location (5%)

| Location | Score |
|---|---|
| Pune, Noida, Delhi/NCR, Gurgaon, Hyderabad, Mumbai (JD target cities) | 100 |
| Other major India cities + willing to relocate | 83 |
| Other major India cities, not relocating | 68 |
| India, other locations + willing to relocate | 73 |
| India, other locations, not relocating | 56 |
| Outside India + willing to relocate | 42 |
| Outside India, not relocating | 18 |

---

### 6. Education (5%)

Institution tier × field alignment:

| Tier | Base score |
|---|---|
| Tier 1 | 100 |
| Tier 2 | 82 |
| Tier 3 | 62 |
| Tier 4 | 42 |
| Unknown | 52 |

If field of study is CS, ECE, Electronics, Software Engineering, Data Science, AI/ML, Mathematics, Statistics, or Physics, the score is kept at full value. All other fields receive a 0.80× field-alignment discount.

---

## Anti-Trap Layer

Four independent checks run before any score is finalised.

### Honeypot Detection → score forced to 0.1

A candidate is flagged as a honeypot and assigned score 0.1 if any of the following are true:

- **Mass expert stuffing:** 5 or more skills marked as `expert` proficiency with both `duration_months = 0` AND `endorsements = 0`. No genuine expert has zero evidence across multiple claimed top-tier skills.
- **Impossible tenure:** A single job's `duration_months` exceeds `(years_of_experience × 12) + 36`. You cannot have worked somewhere longer than you have been working.
- **Timeline contradiction:** Total months across all career history exceeds `(years_of_experience × 12) + 60` and the total is more than 72 months. Contradicts the stated profile.
- **All-zero skill evidence:** All skills in a profile of 10 or more skills have both zero duration and zero endorsements simultaneously.

### Keyword Stuffer Penalty

- Each skill with `duration_months = 0` AND `endorsements = 0` receives `trust = 0.08`.
- If the candidate's career descriptions contain no ML vocabulary, all skill trust scores are additionally halved.
- More than 3 zero-evidence skills triggers a compounding penalty on the total skills score.

### Cross-Signal Integrity Check → 0.20× or 0.50× penalty

If the candidate's current job title is clearly non-technical (Marketing, HR, Operations Manager, Content Writer, Civil Engineer, etc.) AND their career descriptions contain no ML vocabulary, a 0.20× penalty is applied to the entire base score. If some career ML evidence exists but there are 15+ skills claimed, a 0.50× penalty applies.

### Pure Consulting Disqualifier → career score ×0.20

If more than 85% of total career months were spent at consulting or services firms, the career component score is multiplied by 0.20 before entering the weighted sum.

---

## Behavioral Multiplier

Nine platform signals from `redrob_signals` combine into a multiplier clamped to [0.35, 1.20]:

| Signal | Contribution |
|---|---|
| Last active ≤ 14 days | +0.18 |
| Last active ≤ 30 days | +0.12 |
| Last active ≤ 90 days | +0.05 |
| Last active > 180 days | −0.12 |
| Open to work flag | +0.10 |
| Recruiter response rate | rate × 0.15 |
| Avg response time ≤ 6 hours | +0.06 |
| Avg response time ≤ 24 hours | +0.03 |
| Notice period ≤ 15 days | +0.11 |
| Notice period ≤ 30 days | +0.08 |
| Notice period ≤ 60 days | +0.04 |
| Notice period 61–90 days | −0.03 |
| Notice period > 90 days | −0.13 |
| Verified email + phone | +0.05 |
| Profile completeness | score/100 × 0.07 |
| Saved by recruiters ≥ 10 (30d) | +0.07 |
| Saved by recruiters ≥ 5 | +0.04 |
| Search appearances ≥ 200 (30d) | +0.04 |
| GitHub activity ≥ 75 | +0.07 |
| GitHub activity ≥ 50 | +0.04 |
| Interview completion rate | rate × 0.05 |
| Offer acceptance rate (if known) | rate × 0.04 |

**Baseline:** 0.55 before any signal adjustments.

---

## Technologies Used

| Technology | Why |
|---|---|
| **Python 3.11** | Zero external dependencies. Runs on any machine with Python 3.8+. Meets the no-network, CPU-only constraint without compromise. |
| **json module** | Line-by-line JSONL streaming. Never loads the full 100K dataset into memory at once. |
| **csv module** | Direct, spec-compliant CSV output. Handles quoting and encoding correctly without a library. |
| **re module** | 14 phrase-level regex patterns on career descriptions. Deterministic, microsecond-per-match, fully interpretable. |
| **gzip module** | Native .jsonl.gz support. No separate decompression step, no extra tooling. |
| **datetime module** | Calculates days since last_active_date and signup_date relative to the competition reference date. |
| **Rule-based scoring** | Explicit weights derived from JD reasoning. No training data, no model drift, no black box. Every decision is defensible. |

---

## Results

| Metric | Value |
|---|---|
| Candidates scored | 100,000 |
| Runtime | ~55 seconds (CPU only) |
| Peak RAM | ~700 MB |
| Honeypots in top-100 | 0 |
| India-based in top-100 | 93% |
| Score at Rank 1 | 0.9955 |
| Score at Rank 100 | 0.6786 |

### Top 10 Candidates

| Rank | Score | Title | Company | YoE | GitHub | Notice | Key Skills |
|---|---|---|---|---|---|---|---|
| 1 | 0.9955 | Sr ML Engineer | Zomato | 7.2 | 95 | 15d | Weaviate, Pinecone, Info Retrieval, Milvus |
| 2 | 0.9811 | Sr AI Engineer | Apple | 5.9 | 97 | 30d | FAISS, OpenSearch, Sentence-Transformers |
| 3 | 0.9743 | Sr Applied Scientist | Meta | 16 | 78 | 30d | Qdrant, BM25, OpenSearch, Weaviate |
| 4 | 0.9734 | Staff ML Engineer | Paytm | 7.0 | 68 | 60d | Semantic Search, pgvector, Pinecone |
| 5 | 0.9657 | Sr AI Engineer | Netflix | 7.8 | 83 | 45d | Learning to Rank, BM25, Pinecone |
| 6 | 0.9520 | Lead AI Engineer | Razorpay | 6.7 | 34 | 30d | Info Retrieval, pgvector, Elasticsearch |
| 7 | 0.9476 | AI Engineer | Microsoft | 6.9 | 64 | 30d | Python, Sentence-Transformers, RAG |
| 8 | 0.9376 | Rec. Systems Engineer | CRED | 8.0 | 71 | 60d | FAISS, Milvus, PEFT, QLoRA |
| 9 | 0.9306 | Search Engineer | Sarvam AI | 7.6 | 61 | 45d | Milvus, Semantic Search, Weaviate |
| 10 | 0.9170 | Applied ML Engineer | Freshworks | 6.0 | 86 | 90d | Qdrant, FAISS, OpenSearch, QLoRA |

---

## File Structure

```
redrob-ranker/
├── rank.py                    # Main ranker — single file, zero dependencies
├── submission.csv             # Final output: 100 ranked candidates
├── requirements.txt           # stdlib only for ranker; pandas/jupyter for notebook
├── submission_metadata.yaml   # Hackathon submission metadata
├── analysis.ipynb             # EDA notebook: data exploration and insights
├── .gitignore                 # Excludes candidates.jsonl (500MB) from repo
└── README.md                  # This file
```

> **Note:** `candidates.jsonl` and `candidates.jsonl.gz` are excluded from the repository via `.gitignore` due to file size. Download them from the hackathon portal and place them in the root directory before running.

---

## How to Reproduce

### Prerequisites

- Python 3.8 or later
- No pip installs required for the core ranker
- `candidates.jsonl` or `candidates.jsonl.gz` from the hackathon portal

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
cd redrob-ranker
```

### Step 2 — Place candidate data

```bash
# JSONL format
cp /path/to/candidates.jsonl .

# Or gzipped format — both are supported
cp /path/to/candidates.jsonl.gz .
```

### Step 3 — Run the ranker

```bash
# Standard JSONL input
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Gzipped input
python3 rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

Progress is printed to stderr every 25,000 candidates. Total runtime is approximately 55 seconds.

### Step 4 — Validate the output

```bash
python3 validate_submission.py submission.csv
# Expected output: Submission is valid.
```

### Step 5 — (Optional) Run the EDA notebook

```bash
pip install jupyter pandas matplotlib seaborn
jupyter notebook analysis.ipynb
```

---

## Design Decisions

**Why 35% on career history?**
A Marketing Manager with 15 AI skills on their profile should not rank above an NLP Engineer who has spent four years shipping ML systems. Skills are self-reported and unverifiable at scale. Career history — titles held, companies worked at, descriptions of actual work done — is harder to fabricate and more predictive of real capability.

**Why a trust factor on skills instead of just proficiency?**
Proficiency is self-declared. Duration and endorsements are harder to fake in bulk. A skill marked `expert` with zero months of usage and zero endorsements from colleagues is statistically indistinguishable from a keyword stuffed by someone who read a blog post. The 0.08× trust weight makes stuffed skills contribute almost nothing without requiring any manual curation.

**Why a multiplicative behavioral signal instead of additive?**
If behavioral signals were additive, a highly available but unqualified candidate could climb the rankings just by being responsive. Multiplicative structure ensures that behavioral signals can only modulate an already-computed quality score — they cannot substitute for genuine fit.

**Why no LLMs or ML models?**
Three reasons: the compute constraint (CPU only, no network, 5 minutes), interpretability (every score must be explainable), and robustness (no hallucination risk in reasoning strings). A rule-based scorer with 80+ semantic tokens derived from careful JD analysis achieves recruiter-grade ranking quality in 55 seconds and produces reasoning that is guaranteed to reference only data present in the profile.

**Why phrase-level regex over simple keyword matching?**
Simple keyword matching counts the word `ranking` the same whether it appears in "I am interested in ranking algorithms" or "built a ranking system serving 50 million users that improved NDCG by 14%." Phrase-level patterns target the second kind of sentence — the kind that proves the candidate actually did the work.
