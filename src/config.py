"""
Centralized configuration for the Career Predictor ML pipeline.

All paths, hyperparameters, and constants are managed here.
Uses environment variables with sensible defaults for portability.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Immutable configuration object for the entire pipeline."""
    
    # ─── UI & Calibration ───────────────────────────────────────────
    PREDICTION_SHARPENING = float(os.environ.get('PREDICTION_SHARPENING', 3.5))

    # ─── Project Root ───────────────────────────────────────────────
    PROJECT_ROOT = Path(os.environ.get(
        'PROJECT_ROOT',
        Path(__file__).resolve().parent.parent
    ))

    # ─── Data Paths ─────────────────────────────────────────────────
    DATA_RAW_DIR = PROJECT_ROOT / os.environ.get('DATA_RAW_DIR', 'data/raw')
    DATA_PROCESSED_DIR = PROJECT_ROOT / os.environ.get('DATA_PROCESSED_DIR', 'data/processed')
    DATA_EXTERNAL_DIR = PROJECT_ROOT / 'data' / 'external'

    # ─── Model Paths ────────────────────────────────────────────────
    MODELS_DIR = PROJECT_ROOT / os.environ.get('MODELS_DIR', 'models')
    MODEL_PATH = MODELS_DIR / 'xgb_model.pkl'
    ENCODERS_PATH = MODELS_DIR / 'label_encoders.pkl'
    SCALER_PATH = MODELS_DIR / 'feature_scaler.pkl'
    SHAP_BACKGROUND_PATH = MODELS_DIR / 'shap_background.npy'
    FEATURE_SCHEMA_PATH = MODELS_DIR / 'feature_schema.json'
    METADATA_PATH = MODELS_DIR / 'metadata.json'
    TARGET_ENCODER_PATH = MODELS_DIR / 'target_encoder.pkl'

    # ─── Media Paths ────────────────────────────────────────────────
    MEDIA_DIR = PROJECT_ROOT / os.environ.get('MEDIA_DIR', 'webapp/media')
    SHAP_PLOTS_DIR = MEDIA_DIR / 'shap_plots'

    # ─── ML Configuration ───────────────────────────────────────────
    RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 42))
    TEST_SIZE = float(os.environ.get('TEST_SIZE', 0.2))
    SHAP_BACKGROUND_SIZE = int(os.environ.get('SHAP_BACKGROUND_SIZE', 200))

    # ─── Hyperparameter Tuning ──────────────────────────────────────
    TUNING_N_ITER = int(os.environ.get('TUNING_N_ITER', 50))
    TUNING_CV_FOLDS = int(os.environ.get('TUNING_CV_FOLDS', 5))

    PARAM_GRID = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2],
    }

    # ─── XGBoost Base Parameters ────────────────────────────────────
    XGB_BASE_PARAMS = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbosity': 0,
    }

    # ─── Feature Definitions ────────────────────────────────────────
    # These define the expected features and their types/ranges.
    # The actual schema is locked after training via feature_schema.json.

    NUMERICAL_FEATURES = [
        'gpa',
        'extracurricular_activities',
        'internships',
        'projects',
        'leadership_positions',
        'field_specific_courses',
        'research_experience',
        'coding_skills',
        'communication_skills',
        'problem_solving_skills',
        'teamwork_skills',
        'analytical_skills',
        'presentation_skills',
        'networking_skills',
        'industry_certifications',
    ]

    CATEGORICAL_FEATURES = [
        'field',
    ]

    TARGET_COLUMN = 'career'

    # ─── Feature Metadata (for form validation) ────────────────────
    NUMERICAL_RANGES = {
        'gpa': (2.0, 4.0),
        'extracurricular_activities': (0, 10),
        'internships': (0, 5),
        'projects': (0, 10),
        'leadership_positions': (0, 1),
        'field_specific_courses': (0, 10),
        'research_experience': (0, 1),
        'coding_skills': (0, 5),
        'communication_skills': (0, 5),
        'problem_solving_skills': (0, 5),
        'teamwork_skills': (0, 5),
        'analytical_skills': (0, 5),
        'presentation_skills': (0, 5),
        'networking_skills': (0, 5),
        'industry_certifications': (0, 1),
    }

    CATEGORICAL_OPTIONS = {
        'field': [
            'Architecture', 'Art', 'Biology', 'Business', 'Chemistry',
            'Computer Science', 'Education', 'Engineering', 'Finance',
            'Law', 'Marketing', 'Medicine', 'Music', 'Physics', 'Psychology'
        ],
    }

    # ─── Career Labels ──────────────────────────────────────────────
    CAREER_LABELS = [
        'AI Researcher', 'Accountant', 'Acoustics Specialist', 'Actuary',
        'Advertising Manager', 'Aerospace Engineer', 'Analytical Chemist',
        'Animator', 'Architect', 'Architectural Technologist', 'Art Director',
        'Art Therapist', 'Artist', 'Astronomer', 'Biochemist', 'Biologist',
        'Biomedical Engineer', 'Biotechnologist', 'Brand Manager',
        'Chemical Engineer', 'Chemist', 'Civil Engineer', 'Clinical Psychologist',
        'Composer', 'Conductor', 'Construction Manager', 'Counselor',
        'Credit Analyst', 'Curriculum Developer', 'Cybersecurity Analyst',
        'Data Scientist', 'Dentist', 'Digital Marketing Specialist', 'Doctor',
        'Ecologist', 'Education Administrator', 'Electrical Engineer',
        'Entrepreneur', 'Financial Advisor', 'Financial Analyst',
        'Financial Controller', 'Fluid Mechanics Engineer', 'Forensic Psychologist',
        'Game Developer', 'Geneticist', 'Graphic Designer',
        'Human Resources Specialist', 'Illustrator',
        'Industrial-Organizational Psychologist', 'Inorganic Chemist',
        'Interior Designer', 'Investment Banker', 'Judge', 'Landscape Architect',
        'Lawyer', 'Legal Analyst', 'Legal Consultant', 'Legal Secretary',
        'Manager', 'Market Research Analyst', 'Marketing Manager',
        'Marketing Specialist', 'Mechanical Engineer', 'Microbiologist',
        'Music Teacher', 'Music Therapist', 'Musician', 'Nuclear Physicist',
        'Nurse', 'Organic Chemist', 'Paralegal', 'Pharmacist', 'Physical Chemist',
        'Physician Assistant', 'Physicist', 'Principal', 'Psychologist',
        'Quantum Physicist', 'Risk Analyst', 'School Counselor',
        'School Psychologist', 'Social Media Manager', 'Software Developer',
        'Sound Engineer', 'Special Education Teacher', 'Surgeon', 'Teacher',
        'Urban Planner', 'Web Developer', 'Zoologist'
    ]

    # ─── Logging ────────────────────────────────────────────────────
    LOG_DIR = PROJECT_ROOT / 'logs'
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s'

    @classmethod
    def ensure_directories(cls):
        """Create all required directories if they don't exist."""
        dirs = [
            cls.DATA_RAW_DIR, cls.DATA_PROCESSED_DIR, cls.DATA_EXTERNAL_DIR,
            cls.MODELS_DIR, cls.MEDIA_DIR, cls.SHAP_PLOTS_DIR, cls.LOG_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_all_features(cls):
        """Return ordered list of all features (numerical + categorical)."""
        return cls.NUMERICAL_FEATURES + cls.CATEGORICAL_FEATURES

    @classmethod
    def validate(cls):
        """Validate configuration integrity."""
        assert cls.TEST_SIZE > 0 and cls.TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
        assert cls.RANDOM_SEED >= 0, "RANDOM_SEED must be non-negative"
        assert cls.SHAP_BACKGROUND_SIZE > 0, "SHAP_BACKGROUND_SIZE must be positive"
        return True
