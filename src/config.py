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
    
    # ─── Data Preprocessing & Quality Flags ─────────────────────────
    ENABLE_SCALING = os.environ.get('ENABLE_SCALING', 'false').lower() == 'true'  # RobustScaler for preprocessing
    ENABLE_KNN_IMPUTATION = os.environ.get('ENABLE_KNN_IMPUTATION', 'true').lower() == 'true'  # k=5 KNN imputation
    ENABLE_ISOLATION_FOREST = os.environ.get('ENABLE_ISOLATION_FOREST', 'true').lower() == 'true'  # Outlier detection
    ENABLE_DATA_QUALITY_REPORT = os.environ.get('ENABLE_DATA_QUALITY_REPORT', 'true').lower() == 'true'  # Quality metrics

    
    # ─── Class Imbalance Handling ──────────────────────────────────
    REBALANCE_METHOD = os.environ.get('REBALANCE_METHOD', 'smote')  # 'smote', 'adasyn', 'borderline_smote'
    
    # ─── Monotone Constraints (XGBoost) ────────────────────────────
    # Maps feature names to monotone constraint direction: 1=increasing, -1=decreasing, 0=no constraint
    MONOTONE_CONSTRAINTS = {
        'gpa': 1,
        'coding_skills': 1,
        'analytical_skills': 1,
        'problem_solving_skills': 1,
        'leadership_positions': 1,
        'experience_score': 1,
        'analytical_composite': 1,
        'communication_composite': 1,
    }

    # ─── Hyperparameter Tuning ──────────────────────────────────────
    TUNING_N_ITER = int(os.environ.get('TUNING_N_ITER', 50))
    TUNING_CV_FOLDS = int(os.environ.get('TUNING_CV_FOLDS', 3))
    TUNING_METHOD = os.environ.get('TUNING_METHOD', 'bayesian')  # 'random', 'grid', 'bayesian'
    OPTUNA_N_TRIALS = int(os.environ.get('OPTUNA_N_TRIALS', 10))

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
    
    # ─── Career Validation & Alignment ────────────────────────────────
    # Map academic fields to strictly field-appropriate, model-valid career clusters!
    FIELD_CAREER_ALIGNMENT = {
        'Computer Science': [
            'Software Engineer', 'Data & AI Specialist', 'Engineer',
        ],
        'Engineering': [
            'Engineer', 'Architect & Planner', 'Software Engineer', 'Educator',
        ],
        'Business': [
            'Business Manager', 'Marketing Professional', 'Finance Professional',
        ],
        'Finance': [
            'Finance Professional', 'Business Manager', 'Marketing Professional',
        ],
        'Psychology': [
            'Psychologist', 'Educator',
        ],
        'Chemistry': [
            'Chemist', 'Educator',
        ],
        'Physics': [
            'Physicist', 'Educator',
        ],
        'Marketing': [
            'Marketing Professional', 'Business Manager', 'Educator',
        ],
        'Education': [
            'Educator', 'Business Manager',
        ],
        'Art': [
            'Visual Artist', 'Musician & Audio', 'Educator',
        ],
        'Architecture': [
            'Architect & Planner', 'Engineer', 'Educator',
        ],
        'Music': [
            'Musician & Audio', 'Educator',
        ],
    }
    
    # Map careers to typical interests/domains
    CAREER_INTERESTS = {
        'Software Engineer': [
            'Technology', 'Coding', 'Web Development', 'Mobile App Development', 'Game Development',
            'Artificial Intelligence', 'Problem-solving', 'Cybersecurity', 'Innovation'
        ],
        'Data & AI Specialist': [
            'Technology', 'Artificial Intelligence', 'Data Science', 'Machine Learning',
            'Research', 'Problem-solving', 'Analytics', 'Innovation'
        ],
        'Engineer': [
            'Technology', 'Design', 'Innovation', 'Problem-solving', 'Engineering'
        ],
        'Psychologist': [
            'Psychology', 'Helping People', 'Research', 'Human Behavior'
        ],
        'Marketing Professional': [
            'Business', 'Communication', 'Creativity', 'People', 'Marketing', 'Advertising'
        ],
        'Finance Professional': [
            'Finance', 'Numbers', 'Analysis', 'Strategy', 'Accounting'
        ],
        'Educator': [
            'Teaching', 'Communication', 'Helping People', 'Knowledge', 'Education'
        ],
        'Architect & Planner': [
            'Design', 'Innovation', 'Creativity', 'Engineering', 'Architecture'
        ],
        'Business Manager': [
            'Leadership', 'Strategy', 'People', 'Business', 'Management'
        ],
        'Visual Artist': [
            'Creativity', 'Art', 'Design', 'Innovation', 'UI/UX', 'Graphic Design'
        ],
        'Musician & Audio': [
            'Music', 'Creativity', 'Performance', 'Art', 'Audio'
        ],
        'Chemist': [
            'Science', 'Research', 'Problem-solving', 'Innovation', 'Chemistry'
        ],
        'Physicist': [
            'Science', 'Research', 'Problem-solving', 'Innovation', 'Physics'
        ],
    }
    
    # Minimum skill requirements per career (for validation checks)
    CAREER_SKILL_REQUIREMENTS = {
        'Software Engineer': {
            'coding_skills': 3.5, 'problem_solving_skills': 3.5, 'analytical_skills': 3.0,
            'projects': 2.0
        },
        'Data & AI Specialist': {
            'coding_skills': 3.0, 'analytical_skills': 4.0, 'problem_solving_skills': 3.5,
            'projects': 2.5
        },
        'Engineer': {
            'problem_solving_skills': 3.0, 'analytical_skills': 3.0, 'teamwork_skills': 2.5,
            'projects': 2.0
        },
        'Psychologist': {
            'communication_skills': 3.5, 'analytical_skills': 3.0, 'teamwork_skills': 3.0
        },
        'Marketing Professional': {
            'communication_skills': 3.5, 'presentation_skills': 3.0, 'teamwork_skills': 3.0
        },
        'Finance Professional': {
            'analytical_skills': 3.5, 'problem_solving_skills': 3.0, 'communication_skills': 2.5
        },
        'Educator': {
            'communication_skills': 3.5, 'presentation_skills': 3.5, 'teamwork_skills': 3.0
        },
        'Architect & Planner': {
            'analytical_skills': 3.0, 'problem_solving_skills': 3.0, 'presentation_skills': 2.5
        },
        'Business Manager': {
            'communication_skills': 3.5, 'leadership_positions': 1, 'teamwork_skills': 3.5
        },
        'Visual Artist': {
            'communication_skills': 3.0, 'presentation_skills': 2.5
        },
        'Musician & Audio': {
            'communication_skills': 3.0, 'presentation_skills': 3.0
        },
        'Chemist': {
            'analytical_skills': 3.5, 'problem_solving_skills': 3.0
        },
        'Physicist': {
            'analytical_skills': 3.5, 'problem_solving_skills': 3.0
        },
    }

    # ─── XGBoost Base Parameters ────────────────────────────────────
    # Features to exclude from training (field-derived / leaky)
    DROP_FEATURES = [
        'education_alignment_score',
        'interest_compatibility_score',
        'medicine_gpa_interaction',
        'finance_analytical_interaction',
        'cs_coding_interaction',
        'leadership_creativity_interaction',
    ]

    XGB_BASE_PARAMS = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist',
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 0.5,
    }
    EARLY_STOPPING_ROUNDS = 20

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
        'skills_intelligence_score',        # NEW: Weighted skill representation
        'education_alignment_score',        # NEW: Field-career compatibility
        'interest_compatibility_score',     # NEW: Interest-field alignment
        'career_suitability_index',         # NEW: Composite suitability score
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
            'Architecture', 'Art', 'Business', 'Chemistry',
            'Computer Science', 'Education', 'Engineering', 'Finance',
            'Marketing', 'Music', 'Physics', 'Psychology'
        ],
    }

    # ─── Career Clustering ──────────────────────────────────────────
    # Merge 90 fine-grained careers into ~20 high-density clusters
    USE_CAREER_CLUSTERS = os.environ.get('USE_CAREER_CLUSTERS', 'true').lower() == 'true'

    CAREER_CLUSTERS = {
        # Technology
        'Software Engineer': [
            'Software Development and Engineering', 'Development', 'Security',
            'Software Developer', 'Web Developer', 'Game Developer'
        ],
        'Data & AI Specialist': [
            'Data Science', 'Artificial Intelligence',
            'Data Scientist', 'AI Researcher', 'Cybersecurity Analyst'
        ],
        # Engineering
        'Engineer': [
            'Civil Engineer', 'Mechanical Engineer', 'Electrical Engineer',
            'Chemical Engineer', 'Aerospace Engineer', 'Biomedical Engineer',
            'Fluid Mechanics Engineer', 'Acoustics Specialist'
        ],
        # Business & Management
        'Business Manager': [
            'Manager', 'Entrepreneur', 'Construction Manager',
            'Human Resources Specialist'
        ],
        # Finance (merged Investment & Insurance into Finance Professional)
        'Finance Professional': [
            'Financial Analyst', 'Financial Advisor', 'Financial Controller',
            'Accountant', 'Credit Analyst', 'Risk Analyst',
            'Investment Banker', 'Actuary'
        ],
        # Marketing (merged Brand & Advertising into Marketing Professional)
        'Marketing Professional': [
            'Marketing Manager', 'Marketing Specialist',
            'Digital Marketing Specialist', 'Social Media Manager',
            'Market Research Analyst',
            'Brand Manager', 'Advertising Manager'
        ],
        # Education
        'Educator': [
            'Teacher', 'Special Education Teacher', 'Music Teacher',
            'Principal', 'Education Administrator', 'Curriculum Developer'
        ],
        # Psychology (merged Counselor & Therapist into Psychologist)
        'Psychologist': [
            'Psychologist', 'Clinical Psychologist', 'Forensic Psychologist',
            'School Psychologist', 'Industrial-Organizational Psychologist',
            'Counselor', 'School Counselor', 'Art Therapist', 'Music Therapist'
        ],
        # Sciences
        'Chemist': [
            'Chemist', 'Analytical Chemist', 'Organic Chemist',
            'Inorganic Chemist', 'Physical Chemist'
        ],
        'Physicist': [
            'Physicist', 'Nuclear Physicist', 'Quantum Physicist', 'Astronomer'
        ],
        # Architecture & Design
        'Architect & Planner': [
            'Architect', 'Urban Planner', 'Landscape Architect',
            'Interior Designer', 'Architectural Technologist'
        ],
        # Creative Arts
        'Visual Artist': [
            'User Experience (UX) and User Interface (UI) Design',
            'Artist', 'Art Director', 'Graphic Designer',
            'Illustrator', 'Animator'
        ],
        'Musician & Audio': [
            'Musician', 'Composer', 'Conductor', 'Sound Engineer'
        ],
    }

    # Build reverse mapping: original_career → cluster_name
    _CAREER_TO_CLUSTER = {}
    for cluster, careers in CAREER_CLUSTERS.items():
        for career in careers:
            _CAREER_TO_CLUSTER[career] = cluster

    # ─── Unsupported Domains (filtered at inference time) ────────────
    # These clusters may exist in legacy model artifacts but must never
    # be shown to users or returned as recommendations.
    UNSUPPORTED_CAREER_CLUSTERS = {
        'Doctor & Surgeon',
        'Healthcare Specialist',
        'Legal Professional',
        'Legal Support',
        'Biologist',
    }

    # Minimum Career Alignment Score to accept a recommendation
    MIN_ALIGNMENT_SCORE = 40.0

    # Weights for the Career Alignment Score
    ALIGNMENT_WEIGHT_EDUCATION = 0.40
    ALIGNMENT_WEIGHT_SKILL = 0.35
    ALIGNMENT_WEIGHT_INTEREST = 0.25

    # Weight for combining model confidence vs alignment score in final ranking
    RANK_WEIGHT_CONFIDENCE = 0.50
    RANK_WEIGHT_ALIGNMENT = 0.50

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
        'Art Therapist', 'Artist', 'Astronomer', 'Biomedical Engineer',
        'Brand Manager',
        'Chemical Engineer', 'Chemist', 'Civil Engineer', 'Clinical Psychologist',
        'Composer', 'Conductor', 'Construction Manager', 'Counselor',
        'Credit Analyst', 'Curriculum Developer', 'Cybersecurity Analyst',
        'Data Scientist', 'Digital Marketing Specialist',
        'Education Administrator', 'Electrical Engineer',
        'Entrepreneur', 'Financial Advisor', 'Financial Analyst',
        'Financial Controller', 'Fluid Mechanics Engineer', 'Forensic Psychologist',
        'Game Developer', 'Graphic Designer',
        'Human Resources Specialist', 'Illustrator',
        'Industrial-Organizational Psychologist', 'Inorganic Chemist',
        'Interior Designer', 'Investment Banker', 'Landscape Architect',
        'Manager', 'Market Research Analyst', 'Marketing Manager',
        'Marketing Specialist', 'Mechanical Engineer',
        'Music Teacher', 'Music Therapist', 'Musician', 'Nuclear Physicist',
        'Organic Chemist', 'Physical Chemist',
        'Physicist', 'Principal', 'Psychologist',
        'Quantum Physicist', 'Risk Analyst', 'School Counselor',
        'School Psychologist', 'Social Media Manager', 'Software Developer',
        'Sound Engineer', 'Special Education Teacher', 'Teacher',
        'Urban Planner', 'Web Developer'
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
