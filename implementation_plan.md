# 🎯 Implementation Plan: SHAP-Driven Career Prediction System

## SHAP-Driven Feature Importance Analysis of XGBoost for Explainable Localized Career Prediction Using Academic and Soft-Skill Data

---

## 1. Project Overview

### 1.1 Objective
Build a production-ready, academically rigorous web application that:
- Trains an XGBoost classifier on academic + soft-skill features to predict career paths
- Provides global and local explainability via SHAP TreeExplainer
- Serves predictions through a polished Django web interface
- Maintains strict separation between training and inference pipelines

### 1.2 Key Design Principles
| Principle | Implementation |
|-----------|---------------|
| **Modularity** | Separate `src/` (ML core) from `webapp/` (Django serving layer) |
| **Reproducibility** | Fixed random seeds, versioned models, schema-locked feature order |
| **Scalability** | Cached SHAP explainer, background sampling, lazy model loading |
| **Explainability** | SHAP global summary + per-prediction waterfall plots |
| **Production-readiness** | Logging, error handling, input validation, model versioning |

---

## 2. System Architecture

```
d:\Career Predictor (ML Project)\
│
├── data/
│   ├── raw/                    # Original datasets (CSV)
│   ├── processed/              # Cleaned, merged, encoded data
│   └── external/               # Any supplementary data
│
├── models/
│   ├── xgb_model.pkl           # Trained XGBoost model (joblib)
│   ├── label_encoders.pkl      # Fitted LabelEncoders (joblib)
│   ├── feature_scaler.pkl      # CGPA scaler (joblib)
│   ├── shap_background.npy     # Background sample for SHAP
│   ├── feature_schema.json     # Locked feature order + types
│   └── metadata.json           # Training metadata & metrics
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration
│   ├── processor.py            # Data cleaning + encoding
│   ├── trainer.py              # Model training pipeline
│   ├── predictor.py            # Inference logic (singleton pattern)
│   ├── explain.py              # SHAP explainability layer
│   └── utils.py                # Logging, helpers
│
├── webapp/
│   ├── career_predictor/       # Django project
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
│   │   └── asgi.py
│   │
│   ├── predictor_app/          # Django app
│   │   ├── views.py
│   │   ├── forms.py
│   │   ├── urls.py
│   │   ├── apps.py
│   │   ├── templatetags/
│   │   ├── templates/
│   │   │   └── predictor_app/
│   │   │       ├── base.html
│   │   │       ├── index.html       # Landing page
│   │   │       ├── predict.html     # Input form
│   │   │       ├── result.html      # Prediction + SHAP
│   │   │       ├── global_analysis.html  # Global SHAP
│   │   │       └── about.html
│   │   └── static/
│   │       └── predictor_app/
│   │           ├── css/
│   │           ├── js/
│   │           └── img/
│   │
│   └── media/                  # Generated SHAP plots
│       └── shap_plots/
│
├── tests/
│   ├── __init__.py
│   ├── test_processor.py
│   ├── test_predictor.py
│   └── test_integration.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── manage.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 3. Data Engineering Strategy

### 3.1 Expected Feature Schema (11+ features)

| # | Feature | Type | Range/Values | Category |
|---|---------|------|-------------|----------|
| 1 | `logical_quotient` | int | 1–10 | Cognitive |
| 2 | `hackathons` | int | 0–10 | Experience |
| 3 | `coding_skills` | int | 1–10 | Technical |
| 4 | `public_speaking` | int | 1–10 | Soft Skill |
| 5 | `self_learning` | categorical | Yes/No | Soft Skill |
| 6 | `extra_courses` | categorical | Yes/No | Academic |
| 7 | `certifications` | categorical | e.g., Python, ML, etc. | Academic |
| 8 | `workshops` | categorical | e.g., Web Dev, ML, etc. | Experience |
| 9 | `reading_writing_skills` | categorical | Poor/Medium/Excellent | Soft Skill |
| 10 | `memory_capability` | categorical | Poor/Medium/Excellent | Cognitive |
| 11 | `interested_subjects` | categorical | e.g., Math, CS, etc. | Academic |
| 12 | `interested_career` | categorical | e.g., Developer, Designer | Academic |
| 13 | `company_type` | categorical | e.g., Product, Service | Preference |
| 14 | `senior_elder_advise` | categorical | Yes/No | Soft Skill |
| 15 | `book_general_genre` | categorical | e.g., Technical, Fiction | Soft Skill |
| 16 | `management_technical` | categorical | Management/Technical | Preference |
| 17 | `hard_smart_worker` | categorical | Hard/Smart/Both | Soft Skill |

> [!NOTE]
> The exact feature list will be finalized based on the dataset placed in `data/raw/`. The schema above is based on common career prediction datasets. The processor will auto-detect and validate features.

### 3.2 Preprocessing Pipeline
1. **Load & Merge**: Read all CSVs from `data/raw/`, merge if multiple
2. **Clean**: Strip whitespace, normalize casing, handle NaN
3. **Impute**: Mode imputation for categorical, median for numerical (explicit strategy)
4. **Encode**: LabelEncoder for ordinal/categorical features, stored for inference
5. **Scale**: MinMaxScaler for CGPA normalization (if present)
6. **Schema Lock**: Save feature order to `feature_schema.json` — never changes after training
7. **Split**: 80/20 train-test with stratification on target

### 3.3 Data Validation Rules
- Reject rows with >50% missing values
- Validate categorical values against known sets
- Log warnings for unknown categories (map to "Other")

---

## 4. Model Training Strategy

### 4.1 XGBoost Configuration
```python
base_params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}
```

### 4.2 Hyperparameter Search Space
```python
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}
```

### 4.3 Tuning Strategy
- **Method**: `RandomizedSearchCV` with 50 iterations
- **CV**: 5-fold StratifiedKFold
- **Scoring**: `f1_weighted` (handles class imbalance)
- **Class Imbalance**: `scale_pos_weight` or `sample_weight` based on class distribution

### 4.4 Evaluation Metrics (all saved to metadata.json)
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrix (saved as image)
- Classification report (saved as text)

### 4.5 Model Artifacts
| File | Contents |
|------|----------|
| `xgb_model.pkl` | Trained XGBoost model |
| `label_encoders.pkl` | Dict of {feature_name: fitted_encoder} |
| `feature_scaler.pkl` | Fitted scaler (if CGPA present) |
| `shap_background.npy` | 200-row background sample |
| `feature_schema.json` | Ordered feature names + types |
| `metadata.json` | Training params, metrics, timestamp, version |

---

## 5. SHAP Explainability Layer

### 5.1 Architecture
```
┌─────────────────────┐
│   SHAPExplainer      │  ← Singleton, lazy-loaded
│   (cached)           │
├─────────────────────┤
│ - explainer          │  TreeExplainer(model, background)
│ - background_data    │  200-row sample from training set
│ - feature_names      │  From feature_schema.json
├─────────────────────┤
│ + global_summary()   │  → SHAP summary bar plot (saved PNG)
│ + local_waterfall()  │  → Per-prediction waterfall (saved PNG)
│ + get_shap_values()  │  → Raw SHAP values for custom use
└─────────────────────┘
```

### 5.2 Key Decisions
| Decision | Rationale |
|----------|-----------|
| Background sampling (200 rows) | Full dataset SHAP is O(n²) — sampling keeps it fast |
| TreeExplainer (not KernelExplainer) | XGBoost-native, exact, fast |
| Cached explainer | Avoid reinitializing on every request (~500ms saved) |
| Save plots as PNG to media/ | Django serves them statically; no recomputation |
| Unique filenames per prediction | `shap_waterfall_{uuid}.png` prevents collisions |

### 5.3 Thesis Analysis Outputs
- **Global**: SHAP summary plot showing mean |SHAP value| per feature
- **Comparison**: XGBoost built-in `feature_importances_` vs SHAP importance ranking
- **Insight**: Analysis of soft-skill vs academic feature contribution

---

## 6. Django Web Application

### 6.1 URL Structure
| URL | View | Purpose |
|-----|------|---------|
| `/` | `IndexView` | Landing page |
| `/predict/` | `PredictView` | Input form (GET) + prediction (POST) |
| `/result/<uuid>/` | `ResultView` | Prediction result + SHAP plot |
| `/analysis/` | `AnalysisView` | Global SHAP summary |
| `/about/` | `AboutView` | Project information |

### 6.2 Form Validation
- All fields required
- Numerical fields: min/max validators
- Categorical fields: choices limited to known values
- Custom `clean_*` methods for cross-field validation
- CSRF protection enabled

### 6.3 Prediction Flow
```
User submits form
    → Django validates input
    → Forms cleaned data → dict
    → predictor.predict(input_dict)
        → Encode using saved encoders
        → Align to feature schema
        → model.predict_proba()
        → Return (prediction, probabilities)
    → explain.local_waterfall(input_vector)
        → Generate SHAP plot → save to media/
    → Render result.html with prediction + plot path
```

### 6.4 Frontend Design
- **Design System**: Dark theme with accent gradients (deep navy → electric blue)
- **Typography**: Inter font from Google Fonts
- **Animations**: Subtle hover effects, card transitions, loading states
- **Components**: Glass-morphic cards, gradient buttons, animated progress indicators
- **Responsive**: Mobile-first, 3 breakpoints (mobile/tablet/desktop)
- **SHAP Visualization**: Embedded matplotlib PNG or interactive Plotly chart

---

## 7. Testing Strategy

### 7.1 Unit Tests
| Test | Module | What it validates |
|------|--------|-------------------|
| `test_processor.py` | `processor.py` | Encoding, schema lock, missing value handling |
| `test_predictor.py` | `predictor.py` | Prediction output shape, label mapping, edge cases |

### 7.2 Integration Tests
| Test | Flow |
|------|------|
| `test_integration.py` | Full HTTP POST → prediction → SHAP plot → rendered response |

### 7.3 Edge Cases to Cover
- Unknown category values in input
- All-null input row
- Extreme numerical values (0, max)
- Concurrent prediction requests
- Missing model files (graceful error)

---

## 8. Production & Deployment

### 8.1 Configuration
- All paths via `src/config.py` using `os.environ` with defaults
- Django settings split: `settings/base.py`, `settings/dev.py`, `settings/prod.py` (optional)
- Secrets in `.env` file (never committed)

### 8.2 Deployment Stack
```
Client → Nginx → Gunicorn → Django → src/ ML pipeline
                                   ↓
                              models/ (artifacts)
                              media/ (SHAP plots)
```

### 8.3 Logging
- `logging` module with rotating file handler
- Levels: INFO (predictions), WARNING (unknown categories), ERROR (failures)
- Log format: `[%(asctime)s] %(levelname)s [%(name)s] %(message)s`

---

## 9. Reproducibility Guarantees

| Mechanism | Implementation |
|-----------|---------------|
| Random seed | `np.random.seed(42)`, `random_state=42` everywhere |
| Feature schema lock | `feature_schema.json` — immutable after training |
| Model versioning | `metadata.json` with version string, timestamp |
| Dependency lock | `requirements.txt` with pinned versions |
| Data snapshot | Hash of training data saved in metadata |

---

## 10. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| ML Framework | XGBoost | latest |
| Explainability | SHAP | latest |
| Data Processing | pandas, numpy (2.1.0), scikit-learn | specified |
| Web Framework | Django | latest stable |
| Visualization | matplotlib + SHAP plots | latest |
| Serialization | joblib | latest |
| Server | Gunicorn (prod) | latest |
| Python | 3.10+ | required |

---

> [!IMPORTANT]
> **Dataset Requirement**: The user must place their career prediction dataset(s) as CSV files in `data/raw/` before running the training pipeline. The system will auto-detect columns and build the schema from the data.

> [!TIP]
> **Training vs Serving**: Training (`python -m src.trainer`) is a one-time batch operation. The Django app only does inference using pre-trained artifacts. They share NO runtime dependencies.
