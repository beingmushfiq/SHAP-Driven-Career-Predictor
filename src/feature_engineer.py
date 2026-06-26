"""
Feature Engineering Module for Career Predictor.

Adds analytical composites, communication composites, experience score,
domain-specific interactions, and career suitability features.
"""

import pandas as pd
import numpy as np
from src.config import Config


def create_skills_intelligence_score(df: pd.DataFrame) -> np.ndarray:
    """
    Create Skills Intelligence Score: weighted representation of skills.
    
    Formula:
        Skills Score = (Technical + Analytical + Problem-Solving) / 3
                     × (1 + domain_booster)
    
    Domain boosters:
        - CS field: +0.3 if coding_skills > 3
        - Engineering: +0.2 if problem_solving > 3
        - Finance: +0.3 if analytical_skills > 3
    
    Returns:
        Skills intelligence scores (0-5 normalized scale)
    """
    df = df.copy()
    
    # Base skills average
    coding = pd.to_numeric(df.get('coding_skills', 0), errors='coerce').fillna(0)
    analytical = pd.to_numeric(df.get('analytical_skills', 0), errors='coerce').fillna(0)
    problem_solving = pd.to_numeric(df.get('problem_solving_skills', 0), errors='coerce').fillna(0)
    
    base_score = (coding + analytical + problem_solving) / 3.0
    
    # Field booster
    field_str = df.get('field', '').astype(str).str.lower().str.strip()
    booster = np.ones(len(df))
    
    booster = np.where(
        (field_str == 'computer science') & (coding > 3),
        1.3, booster
    )
    booster = np.where(
        (field_str == 'engineering') & (problem_solving > 3),
        1.2, booster
    )
    booster = np.where(
        (field_str == 'finance') & (analytical > 3),
        1.3, booster
    )
    booster = np.where(
        (field_str == 'business') & (analytical > 3),
        1.15, booster
    )
    
    skills_score = base_score * booster
    # Normalize to 0-5 scale
    return np.clip(skills_score, 0, 5)


def create_education_alignment_score(df: pd.DataFrame) -> np.ndarray:
    """
    Create Education Alignment Score: field-career compatibility.
    
    Based on FIELD_CAREER_ALIGNMENT mapping in config.
    Scores how well the academic field aligns with typical career paths.
    
    This is a static score per field (actual alignment evaluated post-prediction).
    
    Returns:
        Alignment scores (0-100 scale)
    """
    field_str = df.get('field', '').astype(str).str.lower().str.strip()
    
    # Field importance score (higher for STEM, professional fields)
    field_importance = {
        'computer science': 0.95,
        'engineering': 0.9,
        'finance': 0.9,
        'physics': 0.85,
        'chemistry': 0.85,
        'business': 0.8,
        'psychology': 0.75,
        'architecture': 0.8,
        'education': 0.7,
        'marketing': 0.7,
        'art': 0.6,
        'music': 0.6,
    }
    
    alignment = np.array([
        field_importance.get(f, 0.5) * 100
        for f in field_str
    ])
    
    return np.clip(alignment, 0, 100)


def create_interest_compatibility_score(df: pd.DataFrame) -> np.ndarray:
    """
    Create Interest Compatibility Score: how aligned interests are with field.
    
    This uses field-based typical interests (from CAREER_INTERESTS in config).
    In production, this would compare actual user interests with career interests.
    
    Returns:
        Interest alignment scores (0-100 scale)
    """
    # For now, use field-based heuristic (interests not in current data)
    # In real scenario: would compare user_interests with career typical interests
    
    field_str = df.get('field', '').astype(str).str.lower().str.strip()
    
    # Field interest alignment (how specific/aligned the field is)
    field_interest_specificity = {
        'computer science': 0.9,
        'engineering': 0.85,
        'finance': 0.85,
        'psychology': 0.8,
        'art': 0.75,
        'music': 0.7,
        'business': 0.7,
        'education': 0.7,
        'physics': 0.9,
        'chemistry': 0.85,
        'architecture': 0.85,
        'marketing': 0.7,
    }
    
    interest_score = np.array([
        field_interest_specificity.get(f, 0.6) * 100
        for f in field_str
    ])
    
    return np.clip(interest_score, 0, 100)


def create_career_suitability_index(df: pd.DataFrame) -> np.ndarray:
    """
    Create Career Suitability Index: composite score combining multiple factors.
    
    Formula:
        Suitability = 0.40 × Skills + 0.30 × Interests + 0.20 × Education + 0.10 × Experience
    
    Returns:
        Suitability index (0-100 scale)
    """
    # Get component scores
    skills = create_skills_intelligence_score(df)
    interests = create_interest_compatibility_score(df)
    education = create_education_alignment_score(df)
    
    # Experience score (normalize to 0-100)
    experience = pd.to_numeric(df.get('experience_score', 0), errors='coerce').fillna(0)
    experience_norm = np.clip((experience / 10.0) * 100, 0, 100)
    
    # Weighted composite
    suitability = (
        0.40 * skills +
        0.30 * interests +
        0.20 * education +
        0.10 * experience_norm
    )
    
    return np.clip(suitability, 0, 100)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the dataset.
    Works for both training DataFrame and single-row inference DataFrame.
    """
    df = df.copy()
    
    # 1. Analytical Composite (analytical + problem solving + coding)
    df['analytical_composite'] = (
        pd.to_numeric(df.get('analytical_skills', 0), errors='coerce').fillna(0) + 
        pd.to_numeric(df.get('problem_solving_skills', 0), errors='coerce').fillna(0) + 
        pd.to_numeric(df.get('coding_skills', 0), errors='coerce').fillna(0)
    )
    
    # 2. Communication Composite (communication + presentation + networking)
    df['communication_composite'] = (
        pd.to_numeric(df.get('communication_skills', 0), errors='coerce').fillna(0) + 
        pd.to_numeric(df.get('presentation_skills', 0), errors='coerce').fillna(0) + 
        pd.to_numeric(df.get('networking_skills', 0), errors='coerce').fillna(0)
    )
    
    # 3. Experience Score (internships * 2 + projects + research * 2 + leadership)
    df['experience_score'] = (
        pd.to_numeric(df.get('internships', 0), errors='coerce').fillna(0) * 2 +
        pd.to_numeric(df.get('projects', 0), errors='coerce').fillna(0) +
        pd.to_numeric(df.get('research_experience', 0), errors='coerce').fillna(0) * 2 +
        pd.to_numeric(df.get('leadership_positions', 0), errors='coerce').fillna(0)
    )
    
    # 4. Domain-specific field×skill interaction features
    is_cs = df.get('field', '').astype(str).str.lower().str.strip() == 'computer science'
    is_finance = df.get('field', '').astype(str).str.lower().str.strip() == 'finance'
    
    df['cs_coding_interaction'] = np.where(is_cs, pd.to_numeric(df.get('coding_skills', 0), errors='coerce').fillna(0.0), 0.0)
    df['finance_analytical_interaction'] = np.where(is_finance, pd.to_numeric(df.get('analytical_skills', 0), errors='coerce').fillna(0.0), 0.0)
    df['medicine_gpa_interaction'] = 0.0  # Medicine domain removed; kept for schema compatibility
    
    # 5. Skills Intelligence Score (NEW)
    df['skills_intelligence_score'] = create_skills_intelligence_score(df)
    
    # 6. Education Alignment Score (NEW)
    df['education_alignment_score'] = create_education_alignment_score(df)
    
    # 7. Interest Compatibility Score (NEW)
    df['interest_compatibility_score'] = create_interest_compatibility_score(df)
    
    # 8. Career Suitability Index (NEW)
    df['career_suitability_index'] = create_career_suitability_index(df)
    
    return df
