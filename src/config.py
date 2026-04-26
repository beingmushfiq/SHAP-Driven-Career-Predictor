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
        'logical_quotient',
        'hackathons',
        'coding_skills',
        'public_speaking',
        'cgpa',
    ]

    CATEGORICAL_FEATURES = [
        'self_learning',
        'extra_courses',
        'certifications',
        'workshops',
        'reading_writing_skills',
        'memory_capability',
        'interested_subjects',
        'interested_career',
        'company_type',
        'senior_elder_advise',
        'book_general_genre',
        'management_technical',
        'hard_smart_worker',
    ]

    TARGET_COLUMN = 'career_label'

    # ─── Feature Metadata (for form validation) ────────────────────
    NUMERICAL_RANGES = {
        'logical_quotient': (1, 10),
        'hackathons': (0, 10),
        'coding_skills': (1, 10),
        'public_speaking': (1, 10),
        'cgpa': (0, 10),
    }

    CATEGORICAL_OPTIONS = {
        'self_learning': ['Yes', 'No'],
        'extra_courses': ['Yes', 'No'],
        'certifications': [
            'App Development', 'Distro Making', 'Full Stack',
            'Hadoop', 'Information Security', 'Machine Learning',
            'Python', 'R Programming', 'Shell Programming',
        ],
        'workshops': [
            'Cloud Computing', 'Data Science', 'Database Security',
            'Game Development', 'Hacking', 'System Designing',
            'Testing', 'Web Technologies',
        ],
        'reading_writing_skills': ['Poor', 'Medium', 'Excellent'],
        'memory_capability': ['Poor', 'Medium', 'Excellent'],
        'interested_subjects': [
            'Computer Architecture', 'IOT', 'Management',
            'Mathematics', 'Networks', 'Parallel Computing',
            'Programming', 'Software Engineering',
        ],
        'interested_career': [
            'Business Process Analyst', 'Cloud Computing',
            'Database Developer', 'Developer', 'Security',
            'System Developer', 'Testing', 'Web Developer',
        ],
        'company_type': [
            'BPA', 'Cloud Services', 'Finance',
            'Product Development', 'SAaS Services',
            'Sales and Marketing', 'Service Based',
            'Testing and Maintenance', 'Web Services',
        ],
        'senior_elder_advise': ['Yes', 'No'],
        'book_general_genre': [
            'Action and Adventure', 'Autobiographies',
            'Comedies', 'Comics', 'Cookbooks', 'Diaries',
            'Drama', 'Encyclopedias', 'Fantasy', 'Fiction',
            'Guide', 'Health', 'History', 'Horror',
            'Journals', 'Math', 'Maths', 'Mystery',
            'Poetry', 'Prayer books', 'Religion-Spirituality',
            'Romance', 'Satire', 'Science', 'Science fiction',
            'Self help', 'Series', 'Travel', 'Trilogy',
        ],
        'management_technical': ['Management', 'Technical'],
        'hard_smart_worker': ['Hard Worker', 'Smart Worker', 'Both'],
    }

    # ─── Career Labels ──────────────────────────────────────────────
    CAREER_LABELS = [
        'Applications Developer',
        'Business Analyst',
        'CRM Technical Developer',
        'Cloud Computing Engineer',
        'Cyber Security Specialist',
        'Data Scientist',
        'Database Administrator',
        'Network Engineer',
        'Project Manager',
        'Software Developer',
        'Software Tester',
        'Systems Architect',
        'Technical Writer',
        'UX Designer',
        'Web Developer',
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
