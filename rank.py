#!/usr/bin/env python3
"""
Redrob Hackathon — Senior AI Engineer Candidate Ranker
=======================================================
Architecture: Multi-component weighted scoring + behavioral signal multiplier

Scoring components (base weights):
  Career history   35%  — titles, company type, production evidence, trajectory
  Skills match     30%  — weighted by proficiency + trust (duration + endorsements)
  Experience       10%  — years in ideal 5-9 band
  Location          5%  — India target cities preferred
  Education         5%  — tier + field alignment
  Behavioral mult  ×    — availability/engagement multiplier (0.35–1.20)

Runtime: ~60-90s for 100K candidates, CPU-only, no network, ~500MB RAM peak
"""

import argparse
import csv
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────

REFERENCE_DATE = date(2026, 6, 9)   # Hackathon date

# ── Component weights ─────────────────────────────────────────────
W_CAREER   = 0.35
W_SKILLS   = 0.30
W_EXP      = 0.10
W_LOC      = 0.05
W_EDU      = 0.05
# Remaining 0.15 budget is "absorbed" by behavioral multiplier headroom

# ── JD-derived skill taxonomy (token → importance 0-10) ───────────
# Tokens are checked as substrings against lower-cased skill names.
# Multiple tokens can match the same skill; we take the max.
CORE_SKILL_MAP: Dict[str, float] = {
    # === REQUIRED: Embeddings & dense retrieval ===
    "sentence-transformers": 10, "sentence_transformers": 10,
    "sentence transformer": 10,
    "bge": 9, "e5 embedding": 9, "openai embedding": 8,
    "text embeddings": 9, "vector embeddings": 9,
    "embeddings": 8, "embedding": 7,

    # === REQUIRED: Vector databases / ANN ===
    "pinecone": 9, "weaviate": 9, "qdrant": 9, "milvus": 9,
    "faiss": 9, "nmslib": 7, "annoy": 7, "pgvector": 7,
    "opensearch": 8, "elasticsearch": 8,
    "vector database": 9, "vector db": 9, "vector search": 9,
    "hybrid search": 9, "ann": 7,
    "approximate nearest neighbor": 8,

    # === REQUIRED: Retrieval & ranking ===
    "bm25": 8, "semantic search": 8, "dense retrieval": 8,
    "sparse retrieval": 7, "retrieval": 6, "information retrieval": 8,
    "colbert": 8, "splade": 8,
    "reranking": 8, "re-ranking": 8, "cross-encoder": 8,
    "bi-encoder": 7, "learning to rank": 9, "ltr": 8,
    "ranking": 7, "listwise": 7, "pairwise ranking": 7,
    "relevance": 6,

    # === REQUIRED: Evaluation frameworks ===
    "ndcg": 7, "mrr": 7, "map@": 7, "precision@": 6,
    "a/b testing": 7, "ab testing": 7, "experimentation": 6,
    "offline evaluation": 6, "online evaluation": 6,
    "evaluation framework": 7,

    # === NLP / LLMs ===
    "nlp": 7, "natural language processing": 7,
    "transformers": 8, "huggingface": 7, "hugging face": 7,
    "bert": 7, "roberta": 6, "t5": 5,
    "llm": 7, "large language model": 7,
    "rag": 8, "retrieval augmented generation": 9,
    "fine-tuning": 7, "fine tuning": 7, "fine-tuning llms": 8,
    "lora": 7, "qlora": 7, "peft": 7,

    # === REQUIRED: Python ===
    "python": 9,

    # === ML frameworks ===
    "pytorch": 7, "tensorflow": 6, "jax": 6,
    "scikit-learn": 6, "sklearn": 6,
    "machine learning": 6, "deep learning": 6,

    # === Nice to have ===
    "xgboost": 6, "lightgbm": 6, "gradient boosting": 5,
    "recommendation systems": 7, "recommender systems": 7,
    "recommendation": 5, "mlops": 6, "model serving": 6,
    "distributed systems": 5, "inference optimization": 6,
    "mlflow": 5, "ray": 5, "triton": 5, "onnx": 5, "vllm": 6,

    # === Weak-positive infra ===
    "docker": 3, "kubernetes": 3, "fastapi": 4, "flask": 3,
    "aws": 3, "gcp": 3, "azure": 3,
}

# Skills indicating CV/Speech focus — explicitly not wanted per JD
CV_SPEECH_TOKENS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "ocr", "image recognition", "image processing",
    "speech recognition", "asr", "tts", "text-to-speech",
    "text to speech", "speech synthesis", "speaker identification",
    "wav2vec", "robotics", "ros", "slam", "lidar", "pose estimation",
}

# Consulting / body-shop firms (penalty if entire career is at these)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl technologies",
    "hcltech", "tech mahindra", "mphasis", "hexaware",
    "niit technologies", "l&t infotech", "ltimindtree", "mindtree",
    "birlasoft", "coforge", "zensar",
}

# ML/AI job title patterns → score (checked with `in` on lower-case title)
ML_TITLE_SCORES: List[Tuple[str, float]] = [
    # Ideal: senior AI/ML engineer at product company
    ("senior ai engineer", 10), ("staff ai engineer", 10),
    ("senior machine learning engineer", 10), ("principal ml", 10),
    ("machine learning engineer", 10), ("ml engineer", 10),
    ("ai engineer", 10),
    ("nlp engineer", 10), ("nlu engineer", 9),
    ("applied scientist", 9), ("applied ml", 9), ("applied ai", 9),
    ("search engineer", 9), ("relevance engineer", 10),
    ("ranking engineer", 10), ("recommendation systems engineer", 9),
    ("recommendation engineer", 9),
    ("information retrieval", 9),
    ("senior data scientist", 8), ("lead data scientist", 8),
    ("data scientist", 7), ("nlp scientist", 9),
    ("research engineer", 7),   # research tilt → slight discount
    ("ml platform", 8), ("ml infrastructure", 8),
    ("mlops engineer", 6),
    ("ai researcher", 5),
    # Neutral / below-ideal
    ("software engineer", 4), ("senior software engineer", 4),
    ("data engineer", 3), ("backend engineer", 3),
    ("full stack", 2), ("frontend", 1),
    # Disqualifying
    ("product manager", 1), ("marketing", 0), ("sales", 0),
    ("hr ", 0), ("recruiter", 0), ("accountant", 0),
    ("content writer", 0), ("designer", 0), ("civil engineer", 0),
    ("mechanical engineer", 0), ("operations manager", 0),
]

# Target and acceptable India locations (lower-cased tokens)
TOP_INDIA_CITIES = {
    "pune", "pimpri", "noida", "delhi", "new delhi",
    "gurgaon", "gurugram", "hyderabad", "mumbai",
    "thane", "navi mumbai",
}
OK_INDIA_CITIES = {
    "bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad",
    "jaipur", "kochi", "trivandrum", "chandigarh", "bhopal",
    "indore", "nagpur", "surat", "vadodara", "vizag", "bhubaneswar",
    "coimbatore", "mysore", "mangalore", "lucknow",
}

# Production evidence tokens in career descriptions
PROD_TOKENS = {
    "production", "deployed", "serving", "inference", "real-time",
    "latency", "throughput", "a/b", "ab test", "experiment",
    "embedding", "vector", "retrieval", "ranking", "semantic",
    "recall", "precision", "ndcg", "mrr", "offline", "online",
    "recommendation", "search", "relevance", "scale", "million",
    "billion", "requests", "users", "pipeline", "feature store",
    "model registry", "learning to rank", "reranking", "hybrid",
    "candidate generation", "indexing", "index", "similarity",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.35, "intermediate": 0.60,
    "advanced": 0.85, "expert": 1.00,
}


# ──────────────────────────────────────────────────────────────────
# 1. UTILITIES
# ──────────────────────────────────────────────────────────────────

def days_since(date_str: Optional[str]) -> int:
    """Days between date_str (YYYY-MM-DD) and REFERENCE_DATE."""
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except ValueError:
        return 999


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def tokenize(text: str) -> set:
    """Lower-case word tokens; skip single chars."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1}


# ──────────────────────────────────────────────────────────────────
# 2. HONEYPOT DETECTION
# ──────────────────────────────────────────────────────────────────

def detect_honeypot(candidate: Dict) -> Tuple[bool, str]:
    """
    Return (is_honeypot, reason). Checks for subtly impossible profiles.
    Honeypots are forced to relevance tier 0 in ground truth.
    """
    skills    = candidate.get("skills", [])
    yoe       = candidate.get("profile", {}).get("years_of_experience", 0)
    history   = candidate.get("career_history", [])

    # Check A: Many expert skills with BOTH zero duration AND zero endorsements
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and (s.get("duration_months") or 0) == 0
        and (s.get("endorsements") or 0) == 0
    )
    if expert_zero >= 5:
        return True, f"{expert_zero} expert skills with zero evidence"

    # Check B: Single job duration exceeds total YOE by large margin
    for job in history:
        dur_months = job.get("duration_months") or 0
        if dur_months > (yoe * 12 + 36):   # 3-year buffer
            return True, (
                f"impossible tenure: {dur_months}m at one company "
                f"vs {yoe}y total experience"
            )

    # Check C: Summed career months wildly exceeds profile YOE
    total_career_months = sum(j.get("duration_months") or 0 for j in history)
    if total_career_months > (yoe * 12 + 60) and total_career_months > 72:
        return True, (
            f"career total ({total_career_months//12}y) far exceeds "
            f"stated YOE ({yoe}y)"
        )

    # Check D: All skills have BOTH zero duration AND zero endorsements (mass stuffing)
    if len(skills) >= 10:
        all_zero = all(
            (s.get("duration_months") or 0) == 0
            and (s.get("endorsements") or 0) == 0
            for s in skills
        )
        if all_zero:
            return True, "all skills zero duration and zero endorsements"

    return False, ""


# ──────────────────────────────────────────────────────────────────
# 3. SKILL SCORING
# ──────────────────────────────────────────────────────────────────

def score_skills(candidate: Dict, career_desc_has_ml: bool) -> Tuple[float, List[str]]:
    """
    Score skills list. Returns (0-100, signal_strings).

    Trust factor per skill:
      - duration_months + endorsements → both contribute
      - If career shows real ML work, trust boost for all skills
    Keyword stuffers: high skill count but no career-description alignment
    → skills are severely down-weighted.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0, []

    raw_score = 0.0
    strong_skills: List[str] = []
    cv_speech_count = 0
    zero_evidence_count = 0
    total_skills = len(skills)

    assessment_scores = (
        candidate.get("redrob_signals", {})
        .get("skill_assessment_scores", {})
    )

    for sk in skills:
        name_raw = sk.get("name", "")
        name = name_raw.lower().strip()
        prof  = sk.get("proficiency", "beginner")
        endors = int(sk.get("endorsements") or 0)
        dur    = int(sk.get("duration_months") or 0)

        # CV/speech exclusion
        if any(cv in name for cv in CV_SPEECH_TOKENS):
            cv_speech_count += 1
            continue

        # Trust factor: how much we believe this skill is real
        if dur == 0 and endors == 0:
            zero_evidence_count += 1
            trust = 0.08  # essentially unbelievable
        elif dur == 0:
            # Only endorsements
            trust = 0.25 + min(0.25, endors / 80)
        else:
            dur_factor  = min(1.0, dur / 24)          # saturates at 2 years
            end_factor  = min(0.25, endors / 80)
            trust       = 0.45 + 0.30 * dur_factor + end_factor

        # Career alignment boosts trust slightly
        if career_desc_has_ml:
            trust = min(1.0, trust * 1.15)

        # Find best matching core skill
        best_val = 0.0
        for skill_key, skill_val in CORE_SKILL_MAP.items():
            if skill_key in name or name in skill_key:
                best_val = max(best_val, skill_val)

        if best_val > 0:
            prof_w  = PROFICIENCY_WEIGHT.get(prof, 0.5)
            contrib = best_val * prof_w * trust

            # Bonus from platform assessment if available
            for asmt_name, asmt_score in assessment_scores.items():
                if asmt_name.lower() in name or name in asmt_name.lower():
                    contrib += (asmt_score / 100) * best_val * 0.25
                    break

            raw_score += contrib
            if best_val >= 7 and prof_w >= 0.6 and trust >= 0.45:
                strong_skills.append(name_raw)

    # Keyword-stuffer penalty: many zero-evidence skills
    if zero_evidence_count > 3:
        kw_penalty = min(0.6, (zero_evidence_count - 3) * 0.08)
        raw_score *= (1.0 - kw_penalty)

    # CV/speech heavy penalty (not the focus of this JD)
    cv_penalty = min(0.35, cv_speech_count * 0.09)
    raw_score *= (1.0 - cv_penalty)

    # If career has no ML evidence but skills claim ML → stuffed
    if not career_desc_has_ml and zero_evidence_count > total_skills * 0.4:
        raw_score *= 0.35

    # Normalise to 0-100 (empirical: 80 raw = roughly great skills)
    score = clamp(raw_score * 1.3)

    signals: List[str] = []
    if strong_skills:
        display = strong_skills[:4]
        signals.append(f"core skills: {', '.join(display)}")
    if zero_evidence_count > 3:
        signals.append(f"keyword-stuffing flag ({zero_evidence_count} zero-evidence skills)")
    if cv_speech_count >= 2:
        signals.append("CV/speech-heavy skills (misaligned with JD)")

    return score, signals


# ──────────────────────────────────────────────────────────────────
# 4. CAREER HISTORY SCORING
# ──────────────────────────────────────────────────────────────────

def score_career(candidate: Dict) -> Tuple[float, List[str], bool]:
    """
    Returns (0-100, signal_list, career_desc_has_ml_flag).

    Penalises: pure consulting career, title-chasing, non-ML background.
    Rewards: ML/AI titles at product companies, production deployment evidence.
    """
    history  = candidate.get("career_history", [])
    if not history:
        return 0.0, [], False

    total_months        = 0
    ml_months           = 0
    product_months      = 0
    consulting_months   = 0
    prod_evidence_roles = 0
    current_is_ml       = False
    short_tenures       = 0
    career_desc_has_ml  = False
    signals: List[str]  = []

    PROD_TOKENS_LOCAL = PROD_TOKENS   # module-level set

    for job in history:
        title    = (job.get("title") or "").lower()
        company  = (job.get("company") or "").lower()
        dur      = int(job.get("duration_months") or 0)
        is_curr  = bool(job.get("is_current"))
        desc     = (job.get("description") or "").lower()
        industry = (job.get("industry") or "").lower()

        total_months += dur

        # ── Title score ──────────────────────────────────────────
        title_score = 2.0    # default (unmatched = generic engineer)
        for pattern, ts in ML_TITLE_SCORES:
            if pattern in title:
                title_score = ts
                break

        ml_role = title_score >= 7

        # ── Company type ─────────────────────────────────────────
        is_consulting = any(firm in company for firm in CONSULTING_FIRMS)
        if is_consulting:
            consulting_months += dur
        else:
            product_months += dur

        # ── Description: production evidence ─────────────────────
        desc_words = tokenize(desc)
        prod_count = len(PROD_TOKENS_LOCAL & desc_words)
        if prod_count >= 4:
            prod_evidence_roles += 1
        if prod_count >= 3:
            career_desc_has_ml = True

        # Check for ML industry keywords even without production count
        ml_industry_tokens = {
            "machine", "learning", "nlp", "retrieval", "ranking",
            "embedding", "recommendation", "search", "ai", "deep"
        }
        if ml_industry_tokens & desc_words:
            career_desc_has_ml = True

        if ml_role:
            ml_months += dur
            if is_curr:
                current_is_ml = True

        # Tenure check (title-chasing)
        if 0 < dur < 18:
            short_tenures += 1

    # ── Composite career score ────────────────────────────────────
    score = 0.0

    # Fraction of career in ML/AI roles (up to 40 pts)
    if total_months > 0:
        ml_frac   = ml_months / total_months
        score    += ml_frac * 40

    # Product-company experience (up to 25 pts)
    if total_months > 0:
        prod_frac = product_months / total_months
        score    += prod_frac * 25

    # Production deployment evidence (up to 20 pts)
    score += min(20.0, prod_evidence_roles * 8)

    # Currently in ML role (bonus 12 pts)
    if current_is_ml:
        score += 12

    # ── Penalties ────────────────────────────────────────────────
    # Pure consulting disqualifier
    if total_months > 0 and consulting_months / total_months > 0.85:
        score *= 0.20
        signals.append("entire career at consulting/services firms (disqualifier)")

    # Title-chasing penalty
    if short_tenures >= 3:
        penalty = min(18, short_tenures * 4)
        score   = max(0.0, score - penalty)
        signals.append(f"title-chasing pattern ({short_tenures} roles < 18 months)")

    score = clamp(score)

    # ── Signals for reasoning ─────────────────────────────────────
    if ml_months >= 36:
        signals.append(f"{ml_months // 12}+ yrs in ML/AI roles")
    if product_months > 24 and consulting_months / max(total_months, 1) < 0.5:
        signals.append("strong product-company background")
    if prod_evidence_roles >= 2:
        signals.append("evidence of production ML system deployment")
    if current_is_ml:
        signals.append("currently in ML/AI role")

    return score, signals, career_desc_has_ml


# ──────────────────────────────────────────────────────────────────
# 5. EXPERIENCE SCORING
# ──────────────────────────────────────────────────────────────────

def score_experience(candidate: Dict) -> Tuple[float, str]:
    yoe = float(candidate.get("profile", {}).get("years_of_experience") or 0)
    if   5.0 <= yoe <=  9.0:  s = 100.0
    elif 4.0 <= yoe <  5.0:   s =  85.0
    elif 9.0 < yoe  <= 11.0:  s =  85.0
    elif 3.0 <= yoe <  4.0:   s =  65.0
    elif 11.0 < yoe <= 14.0:  s =  70.0
    elif 2.0 <= yoe <  3.0:   s =  45.0
    elif yoe > 14.0:           s =  60.0
    else:                      s =  20.0
    return s, f"{yoe:.1f} yrs exp"


# ──────────────────────────────────────────────────────────────────
# 6. LOCATION SCORING
# ──────────────────────────────────────────────────────────────────

def score_location(candidate: Dict) -> Tuple[float, str]:
    profile   = candidate.get("profile", {})
    location  = (profile.get("location") or "").lower()
    country   = (profile.get("country") or "").lower()
    signals   = candidate.get("redrob_signals", {})
    relocate  = bool(signals.get("willing_to_relocate"))

    is_india  = "india" in country or country in ("in",)

    if any(city in location for city in TOP_INDIA_CITIES):
        return 100.0, f"in target city ({location.split(',')[0].strip()})"

    if any(city in location for city in OK_INDIA_CITIES):
        return (82.0 if relocate else 68.0), f"India ({location.split(',')[0].strip()})"

    if is_india:
        return (72.0 if relocate else 55.0), "India-based"

    return (40.0 if relocate else 18.0), f"outside India ({country})"


# ──────────────────────────────────────────────────────────────────
# 7. EDUCATION SCORING
# ──────────────────────────────────────────────────────────────────

TIER_SCORE = {"tier_1": 100, "tier_2": 80, "tier_3": 60, "tier_4": 40, "unknown": 50}
GOOD_FIELDS = {
    "computer science", "cs", "information technology", "it",
    "electronics", "ece", "electrical", "software engineering",
    "data science", "artificial intelligence", "machine learning",
    "statistics", "mathematics", "math", "physics", "information systems",
}

def score_education(candidate: Dict) -> Tuple[float, str]:
    education = candidate.get("education", [])
    if not education:
        return 40.0, "no education listed"

    best_tier  = 0
    good_field = False
    for edu in education:
        tier      = edu.get("tier", "unknown")
        field_raw = (edu.get("field_of_study") or "").lower()
        best_tier = max(best_tier, TIER_SCORE.get(tier, 50))
        if any(f in field_raw for f in GOOD_FIELDS):
            good_field = True

    final = best_tier * (1.0 if good_field else 0.80)
    label = "CS/ML-aligned degree" if good_field else "degree (field not CS/ML)"
    return final, label


# ──────────────────────────────────────────────────────────────────
# 8. BEHAVIORAL SIGNAL MULTIPLIER
# ──────────────────────────────────────────────────────────────────

def behavioral_multiplier(candidate: Dict) -> Tuple[float, List[str]]:
    """
    Returns a multiplier in [0.35, 1.20] that adjusts the base score
    for real-world availability and platform engagement.

    An inactive / unresponsive candidate is, for hiring purposes,
    not actually available — as the JD hackathon note explicitly states.
    """
    s       = candidate.get("redrob_signals", {})
    score   = 0.55   # neutral baseline
    signals: List[str] = []

    # Recency of platform activity
    inactive_days = days_since(s.get("last_active_date"))
    if   inactive_days <= 14:  score += 0.18
    elif inactive_days <= 30:  score += 0.12; signals.append("active last 30 days")
    elif inactive_days <= 90:  score += 0.05
    elif inactive_days <= 180: score -= 0.02
    else:
        score -= 0.12
        signals.append(f"inactive {inactive_days // 30} months")

    # Open to work flag
    if s.get("open_to_work_flag"):
        score += 0.10; signals.append("open to work")

    # Recruiter response rate (very predictive of actual availability)
    rr = float(s.get("recruiter_response_rate") or 0)
    score += rr * 0.15

    # Response speed
    avg_rt = float(s.get("avg_response_time_hours") or 999)
    if   avg_rt <= 6:   score += 0.06
    elif avg_rt <= 24:  score += 0.03
    elif avg_rt > 72:   score -= 0.02

    # Notice period (JD prefers ≤30 days, can buy out 30 days)
    notice = int(s.get("notice_period_days") or 90)
    if   notice <= 30:  score += 0.09; signals.append(f"notice: {notice}d")
    elif notice <= 60:  score += 0.04
    elif notice > 90:   score -= 0.05; signals.append(f"notice: {notice}d (long)")

    # Verification
    if s.get("verified_email") and s.get("verified_phone"):
        score += 0.05
    elif not s.get("verified_email"):
        score -= 0.03

    # Profile completeness
    completeness = float(s.get("profile_completeness_score") or 50) / 100
    score += completeness * 0.07

    # GitHub activity (directly relevant for senior AI eng role)
    gh = float(s.get("github_activity_score") or -1)
    if gh >= 60:
        score += 0.09; signals.append(f"GitHub active ({gh:.0f}/100)")
    elif gh >= 30:
        score += 0.04
    elif gh == -1:
        pass          # neutral — no GitHub linked

    # Interview completion rate
    icr = float(s.get("interview_completion_rate") or 0)
    score += icr * 0.05

    # Offer acceptance rate (positive signal)
    oar = float(s.get("offer_acceptance_rate") or -1)
    if oar >= 0:
        score += oar * 0.04

    mult = clamp(score, 0.35, 1.20)
    return mult, signals


# ──────────────────────────────────────────────────────────────────
# 9. CROSS-SIGNAL INTEGRITY CHECK
# ──────────────────────────────────────────────────────────────────

def cross_signal_penalty(candidate: Dict, title_score: float,
                          career_desc_has_ml: bool) -> float:
    """
    Penalise candidates whose title/career clearly contradicts their
    skills. A Marketing Manager claiming 15 AI skills is a trap.
    Returns a multiplier in (0, 1].
    """
    profile = candidate.get("profile", {})
    curr_title = (profile.get("current_title") or "").lower()

    # Clearly non-technical current title with lots of AI skills
    disqualifying_titles = {
        "marketing", "sales", "hr ", "accountant", "content writer",
        "graphic designer", "civil engineer", "mechanical engineer",
        "operations manager", "customer support", "business analyst",
    }

    title_disq = any(dt in curr_title for dt in disqualifying_titles)
    skill_count = len(candidate.get("skills", []))

    if title_disq and not career_desc_has_ml:
        # Clear mismatch: non-technical job + AI skills → keyword stuffer
        return 0.20

    if title_disq and career_desc_has_ml and skill_count > 15:
        # Partial mismatch
        return 0.50

    return 1.0


# ──────────────────────────────────────────────────────────────────
# 10. MAIN SCORER
# ──────────────────────────────────────────────────────────────────

def score_candidate(candidate: Dict) -> Tuple[float, str]:
    """
    Returns (composite_score 0-100, reasoning_string).
    """
    # ── Honeypot gate ────────────────────────────────────────────
    is_hp, hp_reason = detect_honeypot(candidate)
    if is_hp:
        return 0.1, f"HONEYPOT flagged: {hp_reason}"

    # ── Career (provides ML flag for downstream components) ──────
    c_score, c_sigs, has_ml_career = score_career(candidate)

    # ── Current title score (used in integrity check) ────────────
    curr_title = (
        candidate.get("profile", {}).get("current_title") or ""
    ).lower()
    title_score = 2.0
    for pattern, ts in ML_TITLE_SCORES:
        if pattern in curr_title:
            title_score = ts
            break

    # ── Skills ──────────────────────────────────────────────────
    sk_score, sk_sigs = score_skills(candidate, has_ml_career)

    # ── Experience ───────────────────────────────────────────────
    ex_score, ex_sig = score_experience(candidate)

    # ── Location ─────────────────────────────────────────────────
    lo_score, lo_sig = score_location(candidate)

    # ── Education ────────────────────────────────────────────────
    ed_score, ed_sig = score_education(candidate)

    # ── Weighted base ────────────────────────────────────────────
    base = (
        c_score  * W_CAREER  +
        sk_score * W_SKILLS  +
        ex_score * W_EXP     +
        lo_score * W_LOC     +
        ed_score * W_EDU
    )

    # ── Cross-signal integrity ───────────────────────────────────
    integrity = cross_signal_penalty(candidate, title_score, has_ml_career)
    base *= integrity

    # ── Behavioral multiplier ────────────────────────────────────
    beh_mult, beh_sigs = behavioral_multiplier(candidate)
    final = clamp(base * beh_mult)

    # ── Build reasoning ──────────────────────────────────────────
    profile  = candidate.get("profile", {})
    title    = profile.get("current_title", "N/A")
    yoe      = float(profile.get("years_of_experience") or 0)
    company  = profile.get("current_company", "N/A")
    location = profile.get("location", "N/A")

    all_pos  = c_sigs + beh_sigs
    all_neg  = [
        s for s in c_sigs + sk_sigs
        if any(kw in s.lower() for kw in [
            "flag", "disqualifier", "stuffing", "mismatch",
            "long", "inactive", "consulting", "pattern"
        ])
    ]
    all_pos  = [s for s in all_pos if s not in all_neg]

    # Pick top signals
    positive_str = "; ".join(all_pos[:3]) if all_pos else ""
    skill_str    = "; ".join(sk_sigs[:2]) if sk_sigs and "flag" not in (sk_sigs[0] if sk_sigs else "") else ""
    negative_str = "Concerns: " + "; ".join(all_neg[:2]) if all_neg else ""

    parts = [f"{title} ({yoe:.0f} yrs, {company}, {location.split(',')[0].strip()})"]
    if skill_str and "flag" not in skill_str:
        parts.append(skill_str)
    if positive_str:
        parts.append(positive_str)
    if negative_str:
        parts.append(negative_str)

    reasoning = ". ".join(p for p in parts if p)
    if len(reasoning) > 280:
        reasoning = reasoning[:277] + "..."

    return final, reasoning


# ──────────────────────────────────────────────────────────────────
# 11. ENTRY POINT
# ──────────────────────────────────────────────────────────────────

def rank_candidates(candidates_path: str, output_path: str) -> None:
    import time
    t0 = time.time()

    print(f"[rank.py] Loading {candidates_path}...", file=sys.stderr)
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    n = len(candidates)
    print(f"[rank.py] Loaded {n} candidates. Scoring...", file=sys.stderr)

    scored: List[Tuple[float, str, str]] = []   # (score, cid, reasoning)
    for i, cand in enumerate(candidates):
        score, reasoning = score_candidate(cand)
        scored.append((score, cand["candidate_id"], reasoning))
        if (i + 1) % 20_000 == 0:
            elapsed = time.time() - t0
            print(
                f"[rank.py]   {i+1}/{n} candidates scored "
                f"({elapsed:.1f}s elapsed, "
                f"~{(n - i - 1) / ((i+1) / elapsed):.0f}s remaining)",
                file=sys.stderr,
            )

    print("[rank.py] Sorting...", file=sys.stderr)
    # Sort: descending score; tie-break by candidate_id ascending
    scored.sort(key=lambda x: (-x[0], x[1]))
    top100 = scored[:100]

    print(f"[rank.py] Writing {output_path}...", file=sys.stderr)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, reasoning) in enumerate(top100, start=1):
            # Scores are already non-increasing after sort; normalise to [0,1]
            norm_score = round(score / 100, 6)
            writer.writerow([cid, rank, norm_score, reasoning])

    elapsed = time.time() - t0
    print(
        f"[rank.py] Done in {elapsed:.1f}s.\n"
        f"  Rank 1:   {top100[0][1]} (score {top100[0][0]:.2f})\n"
        f"  Rank 10:  {top100[9][1]} (score {top100[9][0]:.2f})\n"
        f"  Rank 100: {top100[99][1]} (score {top100[99][0]:.2f})",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Senior AI Engineer candidate ranker"
    )
    parser.add_argument(
        "--candidates", default="./candidates.jsonl",
        help="Path to candidates.jsonl (or .jsonl.gz)",
    )
    parser.add_argument(
        "--out", default="./submission.csv",
        help="Output path for the submission CSV",
    )
    args = parser.parse_args()

    # Handle gzipped input
    if args.candidates.endswith(".gz"):
        import gzip, tempfile, os
        print("[rank.py] Decompressing .gz file...", file=sys.stderr)
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
