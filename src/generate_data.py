"""
Synthetic Dataset Generator for Career Predictor.

Generates a realistic career prediction dataset with 17 features
(mix of numerical and categorical) and 15 career labels.
Uses weighted probabilities so features correlate with careers.

Run: python -m src.generate_data
"""

import numpy as np
import pandas as pd
from src.config import Config
from src.utils import set_seeds

set_seeds(Config.RANDOM_SEED)


def generate_career_data(n_samples: int = 800) -> pd.DataFrame:
    """Generate synthetic career prediction dataset."""

    data = {
        'logical_quotient': np.random.randint(1, 11, n_samples),
        'hackathons': np.random.randint(0, 11, n_samples),
        'coding_skills': np.random.randint(1, 11, n_samples),
        'public_speaking': np.random.randint(1, 11, n_samples),
        'self_learning': np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35]),
        'extra_courses': np.random.choice(['Yes', 'No'], n_samples, p=[0.55, 0.45]),
        'certifications': np.random.choice(Config.CATEGORICAL_OPTIONS['certifications'], n_samples),
        'workshops': np.random.choice(Config.CATEGORICAL_OPTIONS['workshops'], n_samples),
        'reading_writing_skills': np.random.choice(['Poor', 'Medium', 'Excellent'], n_samples, p=[0.15, 0.50, 0.35]),
        'memory_capability': np.random.choice(['Poor', 'Medium', 'Excellent'], n_samples, p=[0.10, 0.50, 0.40]),
        'interested_subjects': np.random.choice(Config.CATEGORICAL_OPTIONS['interested_subjects'], n_samples),
        'interested_career': np.random.choice(Config.CATEGORICAL_OPTIONS['interested_career'], n_samples),
        'company_type': np.random.choice(Config.CATEGORICAL_OPTIONS['company_type'], n_samples),
        'senior_elder_advise': np.random.choice(['Yes', 'No'], n_samples, p=[0.60, 0.40]),
        'book_general_genre': np.random.choice(
            ['Science fiction', 'Science', 'Self help', 'Guide', 'Health',
             'Mystery', 'Drama', 'Comics', 'Action and Adventure', 'Fiction',
             'Math', 'History', 'Poetry', 'Autobiographies', 'Travel'],
            n_samples
        ),
        'management_technical': np.random.choice(['Management', 'Technical'], n_samples, p=[0.35, 0.65]),
        'hard_smart_worker': np.random.choice(['Hard Worker', 'Smart Worker', 'Both'], n_samples, p=[0.30, 0.35, 0.35]),
    }

    df = pd.DataFrame(data)
    careers = []

    for _, row in df.iterrows():
        weights = np.ones(len(Config.CAREER_LABELS))

        # High coding + hackathons → Software Developer / Web Developer
        if row['coding_skills'] >= 7 and row['hackathons'] >= 4:
            weights[9] += 4  # Software Developer
            weights[14] += 3  # Web Developer
            weights[0] += 3  # Applications Developer
        # High logic + math interest → Data Scientist
        if row['logical_quotient'] >= 7 and row['interested_subjects'] == 'Mathematics':
            weights[5] += 5  # Data Scientist
        # High public speaking + management → Project Manager / Business Analyst
        if row['public_speaking'] >= 7 and row['management_technical'] == 'Management':
            weights[8] += 5  # Project Manager
            weights[1] += 4  # Business Analyst
        # Security interest → Cyber Security
        if row['interested_career'] == 'Security' or row['workshops'] == 'Hacking':
            weights[4] += 5  # Cyber Security
        # Cloud interest → Cloud Computing Engineer
        if row['interested_career'] == 'Cloud Computing' or row['workshops'] == 'Cloud Computing':
            weights[3] += 5  # Cloud Computing
        # Testing interest → Software Tester
        if row['interested_career'] == 'Testing' or row['workshops'] == 'Testing':
            weights[10] += 5  # Software Tester
        # Database interest → Database Administrator
        if row['interested_career'] == 'Database Developer':
            weights[6] += 5  # Database Administrator
        # Networks → Network Engineer
        if row['interested_subjects'] == 'Networks':
            weights[7] += 4  # Network Engineer
        # Reading/writing excellent → Technical Writer / UX Designer
        if row['reading_writing_skills'] == 'Excellent':
            weights[12] += 3  # Technical Writer
            weights[13] += 2  # UX Designer
        # Systems architecture
        if row['interested_subjects'] == 'Computer Architecture' and row['coding_skills'] >= 6:
            weights[11] += 4  # Systems Architect
        # ML certification → Data Scientist
        if row['certifications'] == 'Machine Learning':
            weights[5] += 4
        # Web technologies → Web Developer
        if row['workshops'] == 'Web Technologies':
            weights[14] += 4
        # CRM
        if row['interested_career'] == 'Business Process Analyst':
            weights[2] += 4  # CRM Technical Developer

        # Normalize and sample
        weights = weights / weights.sum()
        career = np.random.choice(Config.CAREER_LABELS, p=weights)
        careers.append(career)

    df['career_label'] = careers

    # Add some missing values (~2% randomly)
    mask = np.random.random(df.shape) < 0.02
    mask[:, -1] = False  # Never null target
    df = df.mask(mask)

    return df


if __name__ == '__main__':
    Config.ensure_directories()
    df = generate_career_data(800)
    path = Config.DATA_RAW_DIR / 'career_data.csv'
    df.to_csv(path, index=False)
    print(f"Dataset saved: {path}")
    print(f"Shape: {df.shape}")
    print(f"\nCareer distribution:\n{df['career_label'].value_counts()}")
