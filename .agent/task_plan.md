# SHAP Career Predictor - Kaggle Thesis Deployment

## Project Goal
Deploy the SHAP-Driven Career Predictor ML pipeline to Kaggle as a reproducible, interactive thesis presentation showcasing Explainable AI for career guidance.

## Phase Overview

### Phase 1: Code Integration ✓ COMPLETE
**Status**: Complete (Commits: bb06277, 7e3a236)  
**Objective**: Integrate planning-with-files skill for project management  
**Owner**: AI Agent  

- [x] Clone planning-with-files repository
- [x] Create .agent directory structure  
- [x] Initialize task_plan.md, progress.md, findings.md
- [x] Commit changes to GitHub
- [x] Create KAGGLE_NOTEBOOK_SETUP.md (Phase 2 guide)

**Blockers**: None  
**Duration**: ~15 minutes  
**Notes**: Planning-with-files provides persistent markdown-based workflow management. All integration code pushed successfully.

---

### Phase 2: Kaggle Dataset & Notebook Preparation ⏳ READY TO START
**Status**: Ready to Begin (Setup guide complete)  
**Objective**: Set up Kaggle notebook with modular, executable cells  
**Target Time**: 30-45 minutes (mostly manual Kaggle platform interaction)  
**Owner**: Human (manual Kaggle setup), AI Agent provides detailed guide  
**Guide**: See KAGGLE_NOTEBOOK_SETUP.md for step-by-step instructions

#### Step 2A: Dataset Upload
- [ ] Navigate to Kaggle Dashboard
- [ ] Create new dataset: `student-career-dataset`
- [ ] Upload `data/raw/career_dataset_student.csv`
- [ ] Make dataset searchable (public or shared)
- [ ] Add dataset to Kaggle notebook

#### Step 2B: Notebook Cell Structure
- [ ] Cell 1: Environment setup (paths, imports, Kaggle compatibility)
- [ ] Cell 2: Dataset import from /kaggle/input
- [ ] Cell 3: Install dependencies (numpy<2.0 fix for SHAP)
- [ ] Cell 4: Directory initialization
- [ ] Cell 5: Data cleaning & relabeling
- [ ] Cell 6: XGBoost training & hyperparameter tuning
- [ ] Cell 7: SHAP background computation
- [ ] Cell 8: Model evaluation & metrics
- [ ] Cell 9: Demo prediction + waterfall visualization
- [ ] Cell 10: Results & thesis narrative

**Blockers**: Requires manual Kaggle dataset upload  
**Notes**: All cells should be independent and re-executable

---

### Phase 3: Model Training & Validation 📊 PENDING
**Status**: Pending Phase 2  
**Objective**: Execute ML pipeline and generate SHAP artifacts  
**Target Time**: 20-30 minutes  
**Owner**: Kaggle Notebook

#### Step 3A: Data Processing
- [ ] Load processed dataset
- [ ] Feature scaling & encoding verification
- [ ] Data quality checks

#### Step 3B: Model Training
- [ ] Run RandomizedSearchCV for hyperparameter tuning
- [ ] Train final XGBoost model on optimal params
- [ ] Compute model accuracy (target: 98%+)
- [ ] Save model artifacts

#### Step 3C: SHAP Computation
- [ ] Generate SHAP background samples (200)
- [ ] Initialize TreeExplainer
- [ ] Compute SHAP values for demo cases
- [ ] Generate waterfall plots

**Blockers**: Depends on Phase 2 notebook setup  
**Notes**: Pipeline handles 1000+ samples, optimized for Kaggle resources

---

### Phase 4: Thesis Demo & Presentation 🎓 PENDING
**Status**: Pending Phase 3  
**Objective**: Create compelling, reproducible demonstration  
**Target Time**: 15-20 minutes  
**Owner**: Kaggle Notebook markdown cells

#### Step 4A: Problem Narrative
- [ ] Add markdown: Problem statement (black-box vs XAI)
- [ ] Add markdown: Methodology (17 features, XGBoost, SHAP)
- [ ] Add markdown: Key results & insights

#### Step 4B: Interactive Demo
- [ ] Create sample student profile (code cell)
- [ ] Generate prediction (code cell)
- [ ] Render SHAP waterfall (code cell)
- [ ] Highlight top 3 career drivers
- [ ] Show decision transparency logic

#### Step 4C: Thesis Highlights
- [ ] Reproducibility: Every cell executable end-to-end
- [ ] Scalability: Pipeline tested with 1K+ students
- [ ] Interpretability: SHAP explains each prediction
- [ ] Production-Ready: Error handling, schema validation

#### Step 4D: Finalization
- [ ] Make notebook public/shareable
- [ ] Add version control marker
- [ ] Document deployment steps
- [ ] Create README for viewers

**Blockers**: None  
**Notes**: Notebook serves as both thesis artifact and interactive demo

---

## Decision Log

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2026-05-30 | Use Kaggle notebook (not Colab/local) | Self-contained, shareable, reproducible for thesis | ✓ Approved |
| 2026-05-30 | Integrate planning-with-files | Enables structured project management for complex deployment | ✓ Approved |
| 2026-05-30 | Focus on demo narrative over production server | Thesis graders expect interactive visualization, not web app | ✓ Approved |

---

## Risk Mitigation

| Risk | Impact | Mitigation | Status |
|------|--------|-----------|--------|
| SHAP numpy 2.x incompatibility | 🔴 Critical | Pin numpy<2.0.0 in requirements | ✓ Addressed |
| Kaggle notebook timeout (>60min) | 🟡 Medium | Split into shorter cells, avoid long loops | ⏳ Monitor |
| Dataset size limits | 🟢 Low | Dataset is only 60KB, well within limits | ✓ OK |
| Git history pollution | 🟢 Low | Commit as atomic, well-documented change | ✓ Plan |

---

## Definition of Done

- [x] Code integrated and committed to GitHub
- [ ] Kaggle dataset created and accessible
- [ ] Notebook cells 1-10 implemented and tested
- [ ] Model trained with 98%+ accuracy logged
- [ ] SHAP waterfall plots generating successfully  
- [ ] Demo walkthrough executes end-to-end without errors
- [ ] Notebook is public and has README
- [ ] Team has visibility of deployment status
