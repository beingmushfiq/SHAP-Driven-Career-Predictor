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

    # ─── Random Forest Model Paths ──────────────────────────────────
    RF_MODEL_PATH = MODELS_DIR / 'rf_model.pkl'
    RF_METADATA_PATH = MODELS_DIR / 'rf_metadata.json'
    RF_SHAP_BACKGROUND_PATH = MODELS_DIR / 'rf_shap_background.npy'
    RF_ENCODERS_PATH = MODELS_DIR / 'rf_label_encoders.pkl'
    RF_TARGET_ENCODER_PATH = MODELS_DIR / 'rf_target_encoder.pkl'
    RF_FEATURE_SCHEMA_PATH = MODELS_DIR / 'rf_feature_schema.json'

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
    TUNING_METHOD = os.environ.get('TUNING_METHOD', 'bayesian')  # 'random', 'grid', 'bayesian'
    OPTUNA_N_TRIALS = int(os.environ.get('OPTUNA_N_TRIALS', 100))

    PARAM_GRID = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [3, 5, 7, 9, 12],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.3, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1, 1.5, 2, 3],
    }

    # ─── Random Forest Param Grid ───────────────────────────────────
    RF_PARAM_GRID = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None],
    }

    # ─── Ensemble Configuration ─────────────────────────────────────
    ENSEMBLE_METHOD = os.environ.get('ENSEMBLE_METHOD', 'soft_voting')  # soft_voting, hard_voting, weighted_voting, stacking
    ENSEMBLE_WEIGHTS = None  # Auto-computed from CV scores if None

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
        'analytical_composite',
        'communication_composite',
        'experience_score',
        'cs_coding_interaction',
        'finance_analytical_interaction',
        'medicine_gpa_interaction',
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

    # ─── Career Clustering ──────────────────────────────────────────
    # Merge 90 fine-grained careers into ~20 high-density clusters
    USE_CAREER_CLUSTERS = os.environ.get('USE_CAREER_CLUSTERS', 'true').lower() == 'true'

    CAREER_CLUSTERS = {
        # Technology
        'Software Engineer': ['Software Developer', 'Web Developer', 'Game Developer'],
        'Data & AI Specialist': ['Data Scientist', 'AI Researcher', 'Cybersecurity Analyst'],
        # Engineering
        'Engineer': ['Civil Engineer', 'Mechanical Engineer', 'Electrical Engineer',
                     'Chemical Engineer', 'Aerospace Engineer', 'Biomedical Engineer',
                     'Fluid Mechanics Engineer', 'Acoustics Specialist'],
        # Healthcare
        'Doctor & Surgeon': ['Doctor', 'Surgeon', 'Physician Assistant'],
        'Healthcare Specialist': ['Nurse', 'Pharmacist', 'Dentist'],
        # Business & Management
        'Business Manager': ['Manager', 'Entrepreneur', 'Construction Manager',
                             'Human Resources Specialist'],
        # Finance
        'Finance Professional': ['Financial Analyst', 'Financial Advisor', 'Financial Controller',
                                 'Accountant', 'Credit Analyst', 'Risk Analyst'],
        'Investment & Insurance': ['Investment Banker', 'Actuary'],
        # Marketing
        'Marketing Professional': ['Marketing Manager', 'Marketing Specialist',
                                   'Digital Marketing Specialist', 'Social Media Manager',
                                   'Market Research Analyst'],
        'Brand & Advertising': ['Brand Manager', 'Advertising Manager'],
        # Law
        'Legal Professional': ['Lawyer', 'Judge', 'Legal Consultant', 'Legal Analyst'],
        'Legal Support': ['Paralegal', 'Legal Secretary'],
        # Education
        'Educator': ['Teacher', 'Special Education Teacher', 'Music Teacher',
                     'Principal', 'Education Administrator', 'Curriculum Developer'],
        # Psychology
        'Psychologist': ['Psychologist', 'Clinical Psychologist', 'Forensic Psychologist',
                         'School Psychologist', 'Industrial-Organizational Psychologist'],
        'Counselor & Therapist': ['Counselor', 'School Counselor', 'Art Therapist', 'Music Therapist'],
        # Sciences
        'Biologist': ['Biologist', 'Microbiologist', 'Geneticist', 'Biochemist',
                      'Biotechnologist', 'Ecologist', 'Zoologist'],
        'Chemist': ['Chemist', 'Analytical Chemist', 'Organic Chemist',
                    'Inorganic Chemist', 'Physical Chemist'],
        'Physicist': ['Physicist', 'Nuclear Physicist', 'Quantum Physicist', 'Astronomer'],
        # Architecture & Design
        'Architect & Planner': ['Architect', 'Urban Planner', 'Landscape Architect',
                                'Interior Designer', 'Architectural Technologist'],
        # Creative Arts
        'Visual Artist': ['Artist', 'Art Director', 'Graphic Designer',
                          'Illustrator', 'Animator'],
        'Musician & Audio': ['Musician', 'Composer', 'Conductor', 'Sound Engineer'],
    }

    # Build reverse mapping: original_career → cluster_name
    _CAREER_TO_CLUSTER = {}
    for cluster, careers in CAREER_CLUSTERS.items():
        for career in careers:
            _CAREER_TO_CLUSTER[career] = cluster

    @classmethod
    def get_cluster_for_career(cls, career: str) -> str:
        """Map a fine-grained career label to its cluster name."""
        return cls._CAREER_TO_CLUSTER.get(career, career)

    @classmethod
    def get_all_clusters(cls) -> list:
        """Return sorted list of all cluster names."""
        return sorted(cls.CAREER_CLUSTERS.keys())

    # ─── Career Labels (original fine-grained) ──────────────────────
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
