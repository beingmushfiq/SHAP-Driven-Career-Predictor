"""
Feature Engineering Module for Career Predictor.

Adds analytical composites, communication composites, experience score,
and domain-specific interaction features.
"""

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the dataset.
    Works for both training DataFrame and single-row inference DataFrame.
    """
    df = df.copy()
    
    # 1. Analytical Composite (analytical + problem solving + coding)
    # Ensure columns exist, lowercase matching df columns
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
    # Since 'field' might be string or encoded, handle both
    is_cs = df.get('field', '').astype(str).str.lower().str.strip() == 'computer science'
    is_finance = df.get('field', '').astype(str).str.lower().str.strip() == 'finance'
    is_medicine = df.get('field', '').astype(str).str.lower().str.strip() == 'medicine'
    
    df['cs_coding_interaction'] = np.where(is_cs, pd.to_numeric(df.get('coding_skills', 0), errors='coerce').fillna(0.0), 0.0)
    df['finance_analytical_interaction'] = np.where(is_finance, pd.to_numeric(df.get('analytical_skills', 0), errors='coerce').fillna(0.0), 0.0)
    df['medicine_gpa_interaction'] = np.where(is_medicine, pd.to_numeric(df.get('gpa', 0.0), errors='coerce').fillna(0.0), 0.0)
    
    return df
