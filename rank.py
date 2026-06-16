#!/usr/bin/env python3
"""
Redrob Hackathon — Senior AI Engineer Candidate Ranker  v3
===========================================================
v3 improvements over v2:
  1. Career recency decay — recent roles weighted more than old ones
  2. Upward trajectory bonus — reward clear career progression
  3. Skill co-occurrence cluster bonus — covering all three pillars
  4. Salary range soft filter — flag extreme salary mismatch
  5. Notice period buyout window — JD can buy out 30 days notice

Runtime: ~60s for 100K candidates · CPU-only · No network · ~700MB RAM
"""

import argparse
import csv
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────

REFERENCE_DATE = date(2026, 6, 9)

W_CAREER = 0.35
W_SKILLS = 0.30
W_EXP    = 0.10
W_LOC    = 0.05
W_EDU    = 0.05

# ── Improvement 4: Salary soft filter ──────────────────────────
# JD budget estimate for Senior AI Engineer in India: 25–60 LPA
SALARY_BUDGET_MAX_LPA = 65.0   # above this → soft down-weight

# ─────────────────────────────────────────────────────────────────
# 1. COMPANY QUALITY TAXONOMY
# ─────────────────────────────────────────────────────────────────

PREMIUM_T1 = {
    "google", "deepmind", "waymo",
    "meta", "facebook",
    "amazon", "aws",
    "apple",
    "netflix",
    "microsoft", "openai",
    "anthropic", "cohere", "mistral", "stability ai",
    "hugging face", "huggingface",
}

PREMIUM_T2 = {
    "zomato", "swiggy", "blinkit", "dunzo",
    "flipkart", "meesho", "myntra", "nykaa", "lenskart",
    "razorpay", "cred", "phonepe", "paytm", "groww",
    "ola", "rapido",
    "byju", "unacademy", "upgrad",
    "sarvam", "krutrim", "simplismart",
    "yellow.ai", "yellowai", "haptik",
    "freshworks", "zoho", "chargebee", "postman",
    "sharechat", "moj", "inmobi", "vdo.ai",
    "games24x7", "dream11",
    "practo", "mfine", "niramai", "healthkart",
    "slice", "jupiter", "fi money",
}

PREMIUM_T3 = {
    "aganitha", "saarthi", "wysa", "rephrase",
    "skit", "avaamo", "speechmatics",
    "sigmoid", "fractal", "latentview", "mu sigma",
    "leadsquared", "darwinbox", "greythr",
    "policybazaar", "paisabazaar",
    "mmt", "makemytrip", "goibibo",
    "yulu", "ather", "ola electric",
}

COMPANY_MULT = {1: 1.14, 2: 1.07, 3: 1.04}


def company_tier(company_name: str) -> int:
    cn = company_name.lower()
    if any(t in cn for t in PREMIUM_T1): return 1
    if any(t in cn for t in PREMIUM_T2): return 2
    if any(t in cn for t in PREMIUM_T3): return 3
    return 0


# ─────────────────────────────────────────────────────────────────
# 2. SKILL TAXONOMY
# ─────────────────────────────────────────────────────────────────

SKILL_TIER_A: Dict[str, float] = {
    "sentence-transformers": 10, "sentence_transformers": 10,
    "sentence transformer": 10, "bge": 9, "e5 embedding": 9,
    "text embeddings": 9, "vector embeddings": 9, "embeddings": 8, "embedding": 7,

    "pinecone": 10, "weaviate": 10, "qdrant": 10, "milvus": 10,
    "faiss": 10, "pgvector": 8, "nmslib": 7, "annoy": 7,
    "opensearch": 9, "elasticsearch": 9,
    "vector database": 10, "vector db": 10, "vector search": 10,
    "hybrid search": 10, "ann": 7, "approximate nearest neighbor": 9,

    "bm25": 9, "semantic search": 9, "dense retrieval": 9,
    "sparse retrieval": 8, "information retrieval": 9,
    "colbert": 9, "splade": 9,

    "learning to rank": 10, "ltr": 9,
    "reranking": 9, "re-ranking": 9, "cross-encoder": 9, "bi-encoder": 8,
    "ranking": 7, "relevance": 7,

    "ndcg": 8, "mrr": 8, "map@": 7, "precision@": 7,
    "a/b testing": 8, "ab testing": 8,
    "offline evaluation": 7, "online evaluation": 7,
    "evaluation framework": 7,

    "rag": 9, "retrieval augmented generation": 10,
    "python": 10,
}

SKILL_TIER_B: Dict[str, float] = {
    "nlp": 7, "natural language processing": 7,
    "transformers": 8, "huggingface": 7, "hugging face": 7,
    "bert": 7, "roberta": 6, "t5": 6,
    "llm": 7, "large language model": 7,
    "fine-tuning": 7, "fine tuning": 7, "fine-tuning llms": 8,
    "lora": 7, "qlora": 7, "peft": 7,
    "pytorch": 7, "tensorflow": 6, "jax": 6,
    "scikit-learn": 6, "sklearn": 6,
    "machine learning": 6, "deep learning": 6,
    "recommendation systems": 7, "recommender systems": 7,
    "recommendation": 5, "mlops": 6, "model serving": 6,
    "xgboost": 6, "lightgbm": 6,
    "vllm": 6, "triton": 5, "onnx": 5,
}

SKILL_TIER_C: Dict[str, float] = {
    "distributed systems": 5, "inference optimization": 5,
    "fastapi": 4, "flask": 3, "docker": 3, "kubernetes": 3,
    "aws": 3, "gcp": 3, "azure": 3, "ray": 4,
    "spark": 3, "kafka": 3, "airflow": 3,
    "search": 4, "mlflow": 4, "dvc": 4,
}

ALL_SKILLS: Dict[str, float] = {**SKILL_TIER_A, **SKILL_TIER_B, **SKILL_TIER_C}

REQUIRED_CATEGORIES: Dict[str, List[str]] = {
    "python":         ["python"],
    "embeddings":     ["embed", "sentence-transform", "bge", "e5 ", "text embed",
                       "vector embed", "openai embed"],
    "vector_search":  ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "pgvector",
                       "opensearch", "elasticsearch", "vector search", "vector db",
                       "vector database", "ann"],
    "retrieval_rank": ["retrieval", " ranking", "bm25", "hybrid search",
                       "learning to rank", "ltr", "ndcg", "mrr", "rerank",
                       "cross-encoder", "bi-encoder", "information retrieval",
                       "semantic search", "dense retrieval"],
}

REQUIRED_GATE: Dict[int, float] = {0: 0.30, 1: 0.55, 2: 0.80, 3: 1.00, 4: 1.04}

CV_SPEECH_TOKENS = {
    "computer vision", "image classification", "object detection", "image segmentation",
    "ocr", "image recognition", "image processing", "scene understanding",
    "speech recognition", "asr", "tts", "text-to-speech", "text to speech",
    "speech synthesis", "speaker identification", "wav2vec", "whisper",
    "robotics", "ros", "slam", "lidar", "pose estimation",
    "3d reconstruction", "stereo vision",
}

# ── Improvement 3: Skill cluster definitions ────────────────────
SKILL_CLUSTER_VECTOR_DB = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "pgvector",
    "opensearch", "elasticsearch", "vector search", "vector database", "ann"
}
SKILL_CLUSTER_EMBEDDINGS = {
    "sentence-transformers", "sentence transformer", "bge", "embeddings",
    "embedding", "text embeddings", "vector embeddings", "e5 embedding"
}
SKILL_CLUSTER_EVAL_RANK = {
    "ndcg", "mrr", "learning to rank", "ltr", "a/b testing", "ab testing",
    "reranking", "re-ranking", "cross-encoder", "evaluation framework",
    "bm25", "hybrid search", "semantic search", "information retrieval",
    "dense retrieval", "colbert"
}

# ─────────────────────────────────────────────────────────────────
# 3. CAREER CONSTANTS
# ─────────────────────────────────────────────────────────────────

ML_TITLE_SCORES: List[Tuple[str, float]] = [
    ("senior ai engineer", 10), ("staff ai engineer", 10), ("principal ai engineer", 10),
    ("senior machine learning engineer", 10), ("staff machine learning engineer", 10),
    ("principal ml engineer", 10), ("distinguished engineer", 9),
    ("machine learning engineer", 10), ("ml engineer", 10),
    ("ai engineer", 10),
    ("nlp engineer", 10), ("nlu engineer", 9), ("nlp scientist", 9),
    ("applied scientist", 9), ("applied ml", 9), ("applied ai", 9),
    ("search engineer", 9), ("relevance engineer", 10),
    ("ranking engineer", 10), ("recommendation systems engineer", 9),
    ("recommendation engineer", 9),
    ("information retrieval", 9), ("ir engineer", 9),
    ("senior data scientist", 8), ("lead data scientist", 8),
    ("data scientist", 7), ("senior nlp engineer", 9), ("lead nlp", 9),
    ("research engineer", 7),
    ("ml platform", 8), ("ml infrastructure", 8), ("mlops engineer", 6),
    ("ai researcher", 5), ("research scientist", 5),
    ("software engineer", 4), ("senior software engineer", 4),
    ("data engineer", 3), ("backend engineer", 3),
    ("full stack", 2), ("frontend", 1),
    ("product manager", 1), ("marketing", 0), ("sales", 0),
    ("hr ", 0), ("recruiter", 0), ("accountant", 0),
    ("content writer", 0), ("designer", 0), ("civil engineer", 0),
    ("mechanical engineer", 0), ("operations manager", 0),
]

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl technologies",
    "hcltech", "tech mahindra", "mphasis", "hexaware",
    "niit technologies", "l&t infotech", "ltimindtree", "mindtree",
    "birlasoft", "coforge", "zensar",
}

PRODUCTION_PHRASES: List[Tuple[str, float]] = [
    (r"built.{0,40}(ranking|search|recommendation|retrieval|embedding)\s*(system|pipeline|model|engine|service)", 3.0),
    (r"(designed|architect|led).{0,40}(ranking|retrieval|search|recommendation)", 2.5),
    (r"deployed.{0,30}(model|embedding|index|pipeline|service)\s*(to|in|for)?\s*production", 2.5),
    (r"\d+[.,]?\d*\s*(million|billion|m\+?|b\+?)\s*(users|requests|queries|documents|items)", 2.0),
    (r"(improved|increased|boosted).{0,30}(ndcg|recall|precision|mrr|map|ctr|click).{0,20}\d+", 2.5),
    (r"(a/b\s*test|online\s*experiment|statistical\s*significance)", 1.5),
    (r"(offline|online)\s*eval|evaluation\s*framework|benchmark", 1.0),
    (r"(hybrid\s*search|dense.{0,10}sparse|bm25.{0,20}embed)", 2.0),
    (r"(two.stage|multi.stage|candidate\s*generation|re.?rank)", 2.0),
    (r"(semantic\s*search|dense\s*retrieval|vector\s*search).{0,30}(production|serving|scale|deploy)", 2.0),
    (r"(latency|p99|p95|throughput).{0,20}(ms|milliseconds?|qps|rps)", 1.5),
    (r"(feature\s*store|model\s*registry|mlflow|kubeflow|vertex)", 1.0),
    (r"(founded|co.?founded|started).{0,30}(company|startup|product)", 1.5),
    (r"(patent|paper|publication).{0,30}(accepted|published|filed)", 1.0),
]

INDUSTRY_BONUS: Dict[str, float] = {
    "hr tech": 1.15, "hrtech": 1.15, "talent": 1.15, "recruiting": 1.15,
    "staffing": 1.10, "hiring": 1.10, "workforce": 1.08,
    "e-commerce": 1.10, "ecommerce": 1.10, "marketplace": 1.10,
    "retail": 1.06, "quick commerce": 1.10,
    "fintech": 1.06, "financial tech": 1.06, "payments": 1.05,
    "saas": 1.04, "enterprise software": 1.03,
    "artificial intelligence": 1.05, "machine learning": 1.05,
}

# ── Improvement 2: Title tier for trajectory ────────────────────
def title_tier(title: str) -> int:
    t = title.lower()
    if any(k in t for k in ["staff", "principal", "head of", "vp ", "director",
                             "distinguished", "fellow"]): return 6
    if any(k in t for k in ["lead ", "senior ", "sr "]): return 5
    if any(k in t for k in ["ml engineer", "ai engineer", "nlp engineer",
                             "applied scientist", "search engineer", "ranking engineer",
                             "recommendation systems", "machine learning engineer"]): return 4
    if any(k in t for k in ["data scientist", "engineer", "scientist",
                             "analyst", "developer"]): return 3
    if any(k in t for k in ["intern", "trainee", "fresher", "junior"]): return 1
    return 2


# ─────────────────────────────────────────────────────────────────
# 4. LOCATION + EDUCATION
# ─────────────────────────────────────────────────────────────────

TOP_INDIA_CITIES = {
    "pune", "pimpri", "noida", "delhi", "new delhi",
    "gurgaon", "gurugram", "faridabad", "hyderabad",
    "mumbai", "thane", "navi mumbai",
}
OK_INDIA_CITIES = {
    "bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad",
    "jaipur", "kochi", "trivandrum", "thiruvananthapuram",
    "chandigarh", "bhopal", "indore", "nagpur", "surat", "vadodara",
    "vizag", "visakhapatnam", "bhubaneswar",
    "mysore", "mangalore", "coimbatore",
}

TIER_SCORE = {"tier_1": 100, "tier_2": 82, "tier_3": 62, "tier_4": 42, "unknown": 52}
GOOD_FIELDS = {
    "computer science", "cs", "information technology", "it",
    "electronics", "ece", "electrical", "software engineering",
    "data science", "artificial intelligence", "machine learning",
    "statistics", "mathematics", "math", "physics", "information systems",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.35, "intermediate": 0.60,
    "advanced": 0.85, "expert": 1.00,
}

# ─────────────────────────────────────────────────────────────────
# 5. UTILITIES
# ─────────────────────────────────────────────────────────────────

def days_since(ds: Optional[str]) -> int:
    if not ds: return 999
    try:
        return (REFERENCE_DATE - datetime.strptime(ds, "%Y-%m-%d").date()).days
    except ValueError:
        return 999


def months_ago(date_str: Optional[str]) -> float:
    """Months between date_str and REFERENCE_DATE."""
    if not date_str: return 999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days / 30.44
    except ValueError:
        return 999


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def tokenize(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1}


# ─────────────────────────────────────────────────────────────────
# 6. HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_honeypot(candidate: Dict) -> Tuple[bool, str]:
    skills  = candidate.get("skills", [])
    yoe     = candidate.get("profile", {}).get("years_of_experience", 0)
    history = candidate.get("career_history", [])

    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and (s.get("duration_months") or 0) == 0
        and (s.get("endorsements") or 0) == 0
    )
    if expert_zero >= 5:
        return True, f"{expert_zero} expert skills with zero evidence"

    for job in history:
        dur = int(job.get("duration_months") or 0)
        if dur > (yoe * 12 + 36):
            return True, f"impossible tenure {dur}m vs {yoe}y total"

    total_career = sum(int(j.get("duration_months") or 0) for j in history)
    if total_career > (yoe * 12 + 60) and total_career > 72:
        return True, f"career total {total_career//12}y >> stated {yoe}y"

    if len(skills) >= 10:
        if all(
            (s.get("duration_months") or 0) == 0 and (s.get("endorsements") or 0) == 0
            for s in skills
        ):
            return True, "all skills zero duration and zero endorsements"

    return False, ""


# ─────────────────────────────────────────────────────────────────
# 7. REQUIRED SKILL CATEGORY GATE
# ─────────────────────────────────────────────────────────────────

def count_required_categories(candidate: Dict) -> Tuple[int, List[str]]:
    skill_names = " ".join(
        (s.get("name") or "").lower() for s in candidate.get("skills", [])
    )
    career_text = " ".join(
        (j.get("title", "") + " " + j.get("description", "")).lower()
        for j in candidate.get("career_history", [])
    )
    combined = skill_names + " " + career_text

    covered = []
    for cat, tokens in REQUIRED_CATEGORIES.items():
        if any(tok in combined for tok in tokens):
            covered.append(cat)
    return len(covered), covered


# ─────────────────────────────────────────────────────────────────
# 8. IMPROVEMENT 3: SKILL CLUSTER BONUS
# ─────────────────────────────────────────────────────────────────

def skill_cluster_bonus(candidate: Dict) -> float:
    """
    Bonus for covering all three skill pillars:
      Cluster A — Vector databases / ANN
      Cluster B — Embedding models
      Cluster C — Evaluation / ranking frameworks

    Covering all 3 with evidenced skills → +8 pts on skills score
    Covering 2 of 3 → +4 pts
    Covering 1 of 3 → +0 pts (no bonus)
    """
    skills = candidate.get("skills", [])
    evidenced = set()
    for sk in skills:
        name = (sk.get("name") or "").lower()
        dur  = int(sk.get("duration_months") or 0)
        end  = int(sk.get("endorsements") or 0)
        # Only count evidenced skills (not zero-zero)
        if dur > 0 or end > 0:
            evidenced.add(name)

    # Also check career description text for cluster signals
    career_text = " ".join(
        (j.get("description", "") + " " + j.get("title", "")).lower()
        for j in candidate.get("career_history", [])
    )

    def covers(cluster: set) -> bool:
        for token in cluster:
            if token in evidenced or token in career_text:
                return True
        return False

    hits = sum([
        covers(SKILL_CLUSTER_VECTOR_DB),
        covers(SKILL_CLUSTER_EMBEDDINGS),
        covers(SKILL_CLUSTER_EVAL_RANK),
    ])

    if hits == 3: return 8.0
    if hits == 2: return 4.0
    return 0.0


# ─────────────────────────────────────────────────────────────────
# 9. CAREER SCORING (v3: recency decay + trajectory)
# ─────────────────────────────────────────────────────────────────

def score_career(candidate: Dict) -> Tuple[float, List[str], bool]:
    history  = candidate.get("career_history", [])
    redrob   = candidate.get("redrob_signals", {})
    if not history:
        return 0.0, [], False

    # Sort jobs chronologically (oldest first)
    sorted_jobs = sorted(
        history,
        key=lambda j: j.get("start_date") or "1990-01-01"
    )

    total_months = ml_months = product_months = consulting_months = 0
    prod_evidence_score = 0.0
    current_is_ml       = False
    short_tenures       = 0
    career_desc_has_ml  = False
    best_company_tier   = 0
    industry_bonus      = 1.0
    signals: List[str]  = []

    # ── Improvement 1: Recency decay weights ──────────────────
    # Jobs are sorted oldest→newest; assign higher weight to recent jobs
    n_jobs = len(sorted_jobs)
    recency_weights = []
    for idx in range(n_jobs):
        # linear ramp: oldest job → 0.45×, most recent → 1.0×
        if n_jobs == 1:
            recency_weights.append(1.0)
        else:
            w = 0.45 + (0.55 * idx / (n_jobs - 1))
            recency_weights.append(w)

    # Additionally: date-based decay for jobs that started long ago
    def date_decay(start_date_str: Optional[str]) -> float:
        age_months = months_ago(start_date_str)
        if age_months <= 24:   return 1.00
        if age_months <= 48:   return 0.85
        if age_months <= 72:   return 0.70
        if age_months <= 96:   return 0.58
        return 0.45

    # ── Improvement 2: Trajectory detection ───────────────────
    title_tiers = [title_tier(j.get("title", "")) for j in sorted_jobs]
    trajectory_bonus = 0.0
    trajectory_penalty = 0.0

    if len(title_tiers) >= 2:
        # Clear upward trajectory: last role strictly higher than first
        if title_tiers[-1] > title_tiers[0]:
            delta = title_tiers[-1] - title_tiers[0]
            trajectory_bonus = min(10.0, delta * 3.5)
            signals.append(f"upward career trajectory (+{trajectory_bonus:.0f}pts)")

        # Significant downward trajectory: PM → Engineer, Senior → Junior
        elif title_tiers[-1] < title_tiers[0] - 1:
            trajectory_penalty = min(12.0, (title_tiers[0] - title_tiers[-1]) * 4.0)
            signals.append("declining career trajectory (penalty applied)")

    for idx, job in enumerate(sorted_jobs):
        title    = (job.get("title") or "").lower()
        company  = (job.get("company") or "").lower()
        dur      = int(job.get("duration_months") or 0)
        is_curr  = bool(job.get("is_current"))
        desc     = (job.get("description") or "").lower()
        industry = (job.get("industry") or "").lower()
        start_dt = job.get("start_date")

        total_months += dur

        # Recency weight for this specific job
        r_weight = recency_weights[idx] * date_decay(start_dt)

        # Title score
        ts = 2.0
        for pattern, score in ML_TITLE_SCORES:
            if pattern in title:
                ts = score
                break
        ml_role = ts >= 7

        # Consulting check
        is_consulting = any(f in company for f in CONSULTING_FIRMS)
        if is_consulting:
            consulting_months += dur
        else:
            product_months += dur

        # Company quality tier
        tier = company_tier(company)
        if tier > best_company_tier:
            best_company_tier = tier

        # Industry alignment
        for ind_tok, mult in INDUSTRY_BONUS.items():
            if ind_tok in industry or ind_tok in company:
                industry_bonus = max(industry_bonus, mult)

        # Production evidence — phrase-level scoring with recency weight
        phrase_score = 0.0
        for pattern, weight in PRODUCTION_PHRASES:
            if re.search(pattern, desc):
                phrase_score += weight
        prod_evidence_score += phrase_score * r_weight   # v3: recency-weighted evidence

        # Career ML vocabulary check
        desc_words = tokenize(desc)
        ml_tokens  = {"machine", "learning", "nlp", "retrieval", "ranking",
                      "embedding", "recommendation", "search", "deep", "model"}
        if ml_tokens & desc_words:
            career_desc_has_ml = True

        if ml_role:
            ml_months += dur * r_weight   # v3: recency-weighted ML months
            if is_curr:
                current_is_ml = True

        if 0 < dur < 18:
            short_tenures += 1

    # --- Base career score ---
    score = 0.0
    if total_months > 0:
        score += (ml_months / total_months) * 30        # recency-weighted ML frac
        score += (product_months / total_months) * 18
    score += min(18.0, prod_evidence_score * 2.0)       # recency-weighted evidence
    if current_is_ml:
        score += 10.0

    # v3: Apply trajectory adjustments
    score += trajectory_bonus
    score  = max(0.0, score - trajectory_penalty)

    # GitHub engineering bonus
    gh = float(redrob.get("github_activity_score") or -1)
    if gh >= 80:
        score += 12.0; signals.append(f"GitHub {gh:.0f}/100")
    elif gh >= 60:
        score += 6.0
    elif gh >= 40:
        score += 2.0
    elif gh == -1:
        score -= 3.0

    # Company quality multiplier
    if best_company_tier > 0:
        mult = COMPANY_MULT[best_company_tier]
        score = min(100.0, score * mult)
        if best_company_tier == 1:
            signals.append("FAANG / top AI lab experience")
        elif best_company_tier == 2:
            signals.append("top Indian product company background")

    # Industry alignment (only if no company tier bonus already applied)
    if best_company_tier == 0:
        score = min(100.0, score * industry_bonus)

    # Penalties
    if total_months > 0 and consulting_months / total_months > 0.85:
        score *= 0.20
        signals.append("entire career at consulting/services firms (disqualifier)")

    if short_tenures >= 3:
        score = max(0.0, score - min(18, short_tenures * 4))
        signals.append(f"title-chasing pattern ({short_tenures} roles < 18 months)")

    score = clamp(score)

    # Signals for reasoning
    if ml_months >= 36:
        signals.append(f"{int(ml_months / 12)}+ yrs ML/AI (recency-weighted)")
    if product_months > 24 and consulting_months / max(total_months, 1) < 0.5:
        signals.append("product-company background")
    if prod_evidence_score >= 5:
        signals.append("strong production ML evidence in descriptions")
    elif prod_evidence_score >= 2.5:
        signals.append("production ML evidence in descriptions")
    if current_is_ml:
        signals.append("currently in ML/AI role")

    return score, signals, career_desc_has_ml


# ─────────────────────────────────────────────────────────────────
# 10. SKILLS SCORING (v3: cluster bonus integrated)
# ─────────────────────────────────────────────────────────────────

def score_skills(candidate: Dict, career_desc_has_ml: bool) -> Tuple[float, List[str]]:
    skills   = candidate.get("skills", [])
    if not skills:
        return 0.0, []

    raw_score        = 0.0
    strong_skills    : List[str] = []
    cv_speech_count  = 0
    zero_evidence    = 0
    total_skills     = len(skills)
    assessment_scores = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    )

    for sk in skills:
        name  = (sk.get("name") or "").lower().strip()
        prof  = sk.get("proficiency", "beginner")
        endors = int(sk.get("endorsements") or 0)
        dur    = int(sk.get("duration_months") or 0)

        if any(cv in name for cv in CV_SPEECH_TOKENS):
            cv_speech_count += 1
            continue

        # Trust factor
        if dur == 0 and endors == 0:
            zero_evidence += 1
            trust = 0.08
        elif dur == 0:
            trust = 0.28 + min(0.25, endors / 80)
        else:
            dur_f = min(1.0, dur / 24)
            end_f = min(0.25, endors / 80)
            trust = 0.48 + 0.30 * dur_f + end_f

        if career_desc_has_ml:
            trust = min(1.0, trust * 1.15)

        # Tier A
        best_val = 0.0
        for sk_key, sk_val in SKILL_TIER_A.items():
            if sk_key in name or name in sk_key:
                best_val = max(best_val, sk_val)

        # Tier B (0.80× discount)
        if best_val == 0:
            for sk_key, sk_val in SKILL_TIER_B.items():
                if sk_key in name or name in sk_key:
                    best_val = max(best_val, sk_val * 0.80)

        # Tier C (0.55× discount)
        if best_val == 0:
            for sk_key, sk_val in SKILL_TIER_C.items():
                if sk_key in name or name in sk_key:
                    best_val = max(best_val, sk_val * 0.55)

        if best_val > 0:
            prof_w  = PROFICIENCY_WEIGHT.get(prof, 0.5)
            contrib = best_val * prof_w * trust

            # Platform assessment bonus
            for asmt_name, asmt_score in assessment_scores.items():
                if asmt_name.lower() in name or name in asmt_name.lower():
                    contrib += (asmt_score / 100) * best_val * 0.30
                    break

            raw_score += contrib
            if best_val >= 7 and prof_w >= 0.6 and trust >= 0.45:
                strong_skills.append(sk.get("name", name))

    # Keyword-stuffer penalty
    if zero_evidence > 3:
        raw_score *= (1.0 - min(0.60, (zero_evidence - 3) * 0.08))

    # CV/speech penalty
    raw_score *= (1.0 - min(0.35, cv_speech_count * 0.09))

    # Misaligned profile penalty
    if not career_desc_has_ml and zero_evidence > total_skills * 0.4:
        raw_score *= 0.35

    base_score = clamp(raw_score * 1.25)

    # v3: Add skill cluster bonus
    cluster_bonus = skill_cluster_bonus(candidate)
    score = clamp(base_score + cluster_bonus)

    signals: List[str] = []
    if strong_skills:
        signals.append(f"key skills: {', '.join(strong_skills[:4])}")
    if cluster_bonus >= 8.0:
        signals.append("covers all 3 skill pillars (vector DB + embeddings + eval/ranking)")
    elif cluster_bonus >= 4.0:
        signals.append("covers 2 of 3 skill pillars")
    if zero_evidence > 3:
        signals.append(f"keyword-stuffing flag ({zero_evidence} zero-evidence skills)")
    if cv_speech_count >= 2:
        signals.append("CV/speech-heavy (JD mismatch)")

    return score, signals


# ─────────────────────────────────────────────────────────────────
# 11. EXPERIENCE, LOCATION, EDUCATION
# ─────────────────────────────────────────────────────────────────

def score_experience(candidate: Dict) -> Tuple[float, str]:
    yoe = float(candidate.get("profile", {}).get("years_of_experience") or 0)
    if   5.0 <= yoe <=  9.0:  s = 100.0
    elif 4.0 <= yoe <  5.0:   s =  86.0
    elif 9.0 < yoe  <= 11.0:  s =  86.0
    elif 3.0 <= yoe <  4.0:   s =  66.0
    elif 11.0 < yoe <= 14.0:  s =  72.0
    elif 2.0 <= yoe <  3.0:   s =  45.0
    elif yoe > 14.0:           s =  62.0
    else:                      s =  20.0
    return s, f"{yoe:.1f} yrs exp"


def score_location(candidate: Dict) -> Tuple[float, str]:
    profile  = candidate.get("profile", {})
    location = (profile.get("location") or "").lower()
    country  = (profile.get("country") or "").lower()
    relocate = bool(candidate.get("redrob_signals", {}).get("willing_to_relocate"))
    is_india = "india" in country or country == "in"

    if any(c in location for c in TOP_INDIA_CITIES):
        return 100.0, f"based in target city ({location.split(',')[0].strip().title()})"
    if any(c in location for c in OK_INDIA_CITIES):
        return (83.0 if relocate else 68.0), f"India ({location.split(',')[0].strip().title()})"
    if is_india:
        return (73.0 if relocate else 56.0), "India-based"
    return (42.0 if relocate else 18.0), f"outside India ({country})"


def score_education(candidate: Dict) -> Tuple[float, str]:
    education = candidate.get("education", [])
    if not education:
        return 40.0, "no education listed"
    best_tier = 0
    good_field = False
    for edu in education:
        best_tier = max(best_tier, TIER_SCORE.get(edu.get("tier", "unknown"), 52))
        if any(f in (edu.get("field_of_study") or "").lower() for f in GOOD_FIELDS):
            good_field = True
    return best_tier * (1.0 if good_field else 0.80), (
        "CS/ML-aligned degree" if good_field else "degree field not CS/ML"
    )


# ─────────────────────────────────────────────────────────────────
# 12. BEHAVIORAL MULTIPLIER (v3: improved notice buyout window)
# ─────────────────────────────────────────────────────────────────

def behavioral_multiplier(candidate: Dict) -> Tuple[float, List[str]]:
    s       = candidate.get("redrob_signals", {})
    score   = 0.55
    signals : List[str] = []

    # Recency
    inactive = days_since(s.get("last_active_date"))
    if   inactive <= 14:  score += 0.18
    elif inactive <= 30:  score += 0.12; signals.append("active last 30d")
    elif inactive <= 90:  score += 0.05
    elif inactive <= 180: score -= 0.02
    else:
        score -= 0.12; signals.append(f"inactive {inactive//30}m")

    # Open to work
    if s.get("open_to_work_flag"):
        score += 0.10; signals.append("open to work")

    # Response rate
    rr = float(s.get("recruiter_response_rate") or 0)
    score += rr * 0.15

    # Response speed
    avg_rt = float(s.get("avg_response_time_hours") or 999)
    if   avg_rt <= 6:   score += 0.06
    elif avg_rt <= 24:  score += 0.03
    elif avg_rt > 72:   score -= 0.02

    # ── Improvement 5: Notice period with buyout window ────────
    # JD says: preferred sub-30d, can buy out 30 days notice
    # Effective notice = max(0, notice - 30) after buyout
    notice = int(s.get("notice_period_days") or 90)
    effective_notice = max(0, notice - 30)   # 30-day buyout applied

    if   effective_notice == 0  and notice <= 15:  score += 0.11; signals.append(f"notice: {notice}d (immediate)")
    elif effective_notice == 0  and notice <= 30:  score += 0.09; signals.append(f"notice: {notice}d")
    elif effective_notice <= 15:                   score += 0.07; signals.append(f"notice: {notice}d (≤15d effective after buyout)")
    elif effective_notice <= 30:                   score += 0.04
    elif effective_notice <= 60:                   score -= 0.02
    else:
        score -= 0.12; signals.append(f"notice: {notice}d (long)")

    # Verification
    if s.get("verified_email") and s.get("verified_phone"):
        score += 0.05
    elif not s.get("verified_email"):
        score -= 0.03

    # Profile completeness
    score += float(s.get("profile_completeness_score") or 50) / 100 * 0.07

    # Saved by recruiters
    saved = int(s.get("saved_by_recruiters_30d") or 0)
    if saved >= 10: score += 0.07
    elif saved >= 5: score += 0.04
    elif saved >= 2: score += 0.02

    # Search appearances
    appearances = int(s.get("search_appearance_30d") or 0)
    if appearances >= 200: score += 0.04
    elif appearances >= 100: score += 0.02

    # GitHub activity
    gh2 = float(s.get("github_activity_score") or -1)
    if   gh2 >= 75:   score += 0.07
    elif gh2 >= 50:   score += 0.04
    elif gh2 >= 30:   score += 0.01
    elif gh2 == -1:   score -= 0.02

    # Interview completion rate
    icr = float(s.get("interview_completion_rate") or 0)
    score += icr * 0.05

    # Offer acceptance rate
    oar = float(s.get("offer_acceptance_rate") or -1)
    if oar >= 0:
        score += oar * 0.04

    mult = clamp(score, 0.35, 1.20)
    return mult, signals


# ─────────────────────────────────────────────────────────────────
# 13. CROSS-SIGNAL INTEGRITY
# ─────────────────────────────────────────────────────────────────

def cross_signal_penalty(candidate: Dict, career_desc_has_ml: bool) -> float:
    curr_title = (candidate.get("profile", {}).get("current_title") or "").lower()
    disq_titles = {
        "marketing", "sales", "hr ", "accountant", "content writer",
        "graphic designer", "civil engineer", "mechanical engineer",
        "operations manager", "customer support",
    }
    title_disq = any(dt in curr_title for dt in disq_titles)
    skill_count = len(candidate.get("skills", []))
    if title_disq and not career_desc_has_ml:
        return 0.20
    if title_disq and career_desc_has_ml and skill_count > 15:
        return 0.50
    return 1.0


# ─────────────────────────────────────────────────────────────────
# 14. IMPROVEMENT 4: SALARY SOFT FILTER
# ─────────────────────────────────────────────────────────────────

def salary_multiplier(candidate: Dict) -> float:
    """
    Soft down-weight candidates whose minimum salary expectation
    far exceeds the estimated JD budget.
    This improves recruiter usability — a great candidate who won't
    accept the offer is not actually placeable.
    """
    s = candidate.get("redrob_signals", {})
    sal = s.get("expected_salary_range_inr_lpa") or {}
    min_sal = float(sal.get("min") or 0)

    if min_sal <= 0:
        return 1.0   # no salary data — neutral

    if min_sal > SALARY_BUDGET_MAX_LPA * 1.5:
        return 0.82  # minimum expectation 50%+ above budget — significant mismatch
    if min_sal > SALARY_BUDGET_MAX_LPA * 1.2:
        return 0.92  # minimum expectation 20%+ above budget — mild mismatch
    return 1.0


# ─────────────────────────────────────────────────────────────────
# 15. REASONING BUILDER
# ─────────────────────────────────────────────────────────────────

def build_reasoning(
    candidate: Dict,
    career_signals: List[str], skill_signals: List[str],
    beh_signals: List[str],
    career_score: float, skills_score: float, final_score: float,
) -> str:
    p      = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})
    title  = p.get("current_title", "N/A")
    yoe    = float(p.get("years_of_experience") or 0)
    company= p.get("current_company", "N/A")
    loc    = (p.get("location") or "").split(",")[0].strip().title()
    gh     = float(redrob.get("github_activity_score") or -1)
    notice = int(redrob.get("notice_period_days") or 90)

    parts: List[str] = []
    parts.append(f"{title} ({yoe:.0f} yrs, {company}, {loc})")

    positives = []
    if gh >= 75:
        positives.append(f"GitHub {gh:.0f}/100")

    tier = max(
        (company_tier(j.get("company", "")) for j in candidate.get("career_history", [])),
        default=0,
    )
    if tier == 1:
        positives.append("FAANG / top AI lab background")

    traj = next((s for s in career_signals if "trajectory" in s and "+" in s), None)
    if traj:
        positives.append("rising career trajectory")

    if "strong production ML evidence" in " ".join(career_signals):
        positives.append("production ML system evidence")

    ml_yrs = next((s for s in career_signals if "yrs ML" in s), None)
    if ml_yrs:
        positives.append(ml_yrs)

    sk_sig = next((s for s in skill_signals if "key skills" in s), None)
    if sk_sig:
        positives.append(sk_sig.replace("key skills: ", ""))

    cluster_sig = next((s for s in skill_signals if "pillars" in s), None)
    if cluster_sig:
        positives.append("all 3 skill pillars covered")

    effective_notice = max(0, notice - 30)
    if effective_notice == 0 and notice <= 30:
        positives.append(f"{notice}d notice")
    elif effective_notice <= 15:
        positives.append(f"{notice}d notice (≤15d effective)")

    if redrob.get("open_to_work_flag"):
        positives.append("open to work")

    negatives = [
        s for s in career_signals + skill_signals
        if any(k in s.lower() for k in [
            "disqualifier", "stuffing", "mismatch", "long",
            "inactive", "consulting only", "title-chasing", "declining"
        ])
    ]

    pos_str = "; ".join(positives[:4]) if positives else ""
    neg_str = ("Concerns: " + "; ".join(negatives[:2])) if negatives else ""

    sentence2_parts = [p for p in [pos_str, neg_str] if p]
    if sentence2_parts:
        parts.append(". ".join(sentence2_parts))

    reasoning = ". ".join(parts)
    if len(reasoning) > 290:
        reasoning = reasoning[:287] + "..."
    return reasoning


# ─────────────────────────────────────────────────────────────────
# 16. MAIN SCORER
# ─────────────────────────────────────────────────────────────────

def score_candidate(candidate: Dict) -> Tuple[float, str]:
    # Honeypot gate
    is_hp, hp_reason = detect_honeypot(candidate)
    if is_hp:
        return 0.1, f"HONEYPOT: {hp_reason}"

    # Components
    c_score, c_sigs, has_ml_career = score_career(candidate)
    sk_score, sk_sigs              = score_skills(candidate, has_ml_career)
    ex_score, ex_sig               = score_experience(candidate)
    lo_score, lo_sig               = score_location(candidate)
    ed_score, ed_sig               = score_education(candidate)

    # Weighted base
    base = (
        c_score  * W_CAREER +
        sk_score * W_SKILLS +
        ex_score * W_EXP    +
        lo_score * W_LOC    +
        ed_score * W_EDU
    )

    # Cross-signal integrity
    base *= cross_signal_penalty(candidate, has_ml_career)

    # Required skill gate
    n_req, req_cats = count_required_categories(candidate)
    base *= REQUIRED_GATE.get(n_req, 0.30)

    # v3: Salary soft filter
    base *= salary_multiplier(candidate)

    # Behavioral multiplier
    beh_mult, beh_sigs = behavioral_multiplier(candidate)
    final = clamp(base * beh_mult)

    reasoning = build_reasoning(
        candidate, c_sigs, sk_sigs, beh_sigs, c_score, sk_score, final
    )
    return final, reasoning


# ─────────────────────────────────────────────────────────────────
# 17. ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def rank_candidates(candidates_path: str, output_path: str) -> None:
    import time
    t0 = time.time()

    print(f"[rank.py v3] Loading {candidates_path}...", file=sys.stderr)
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    n = len(candidates)
    print(f"[rank.py v3] Loaded {n:,}. Scoring...", file=sys.stderr)

    scored: List[Tuple[float, str, str]] = []
    for i, cand in enumerate(candidates):
        score, reasoning = score_candidate(cand)
        scored.append((score, cand["candidate_id"], reasoning))
        if (i + 1) % 25_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"[rank.py v3]   {i+1:,}/{n:,}  "
                f"({elapsed:.0f}s elapsed, ~{(n-i-1)/rate:.0f}s remaining)",
                file=sys.stderr,
            )

    scored.sort(key=lambda x: (-x[0], x[1]))
    top100 = scored[:100]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, reasoning) in enumerate(top100, start=1):
            writer.writerow([cid, rank, round(score / 100, 6), reasoning])

    elapsed = time.time() - t0
    print(
        f"[rank.py v3] Done in {elapsed:.1f}s\n"
        f"  Rank 1   : {top100[0][1]}  (score {top100[0][0]:.2f})\n"
        f"  Rank 10  : {top100[9][1]}  (score {top100[9][0]:.2f})\n"
        f"  Rank 100 : {top100[99][1]}  (score {top100[99][0]:.2f})",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out",        default="./submission.csv")
    args = parser.parse_args()

    if args.candidates.endswith(".gz"):
        import gzip, tempfile, os
        print("[rank.py v3] Decompressing...", file=sys.stderr)
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="wb")
        with gzip.open(args.candidates, "rb") as gz:
            tmp.write(gz.read())
        tmp.close()
        rank_candidates(tmp.name, args.out)
        os.unlink(tmp.name)
    else:
        rank_candidates(args.candidates, args.out)


if __name__ == "__main__":
    main()
