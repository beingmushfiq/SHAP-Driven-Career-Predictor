# 🚀 Kaggle Notebook Execution Guide

This guide provides the complete step-by-step procedure to run the **SHAP-Driven Career Predictor** (both the ML training pipeline and the Django web app interface) on Kaggle.

---

## 📂 Step 1: Prepare the Kaggle Dataset
Because Kaggle's input directories are read-only, you must first upload the dataset to Kaggle:
1. Go to your [Kaggle Dashboard](https://www.kaggle.com/) and click **Create** -> **New Dataset**.
2. Name the dataset **`student-career-dataset`**.
3. Upload `career_dataset_student.csv` (located in your local `data/raw/` directory).
4. Create the dataset.
5. In your Kaggle Notebook, click **+ Add Input** in the right-hand panel, search for `student-career-dataset`, and add it to your notebook.

---

## 💻 Step 2: Notebook Code Cells

Create a new Kaggle Notebook and create the following code cells.

### Cell 1: Environment & Portability Setup
This cell initializes directory paths and points the project root to Kaggle's writable `/kaggle/working` directory.

```python
import os
import sys
from pathlib import Path

# Set environment variables for Kaggle compatibility
os.environ['PROJECT_ROOT'] = '/kaggle/working'
os.environ['DATA_RAW_DIR'] = 'data/raw'
os.environ['DATA_PROCESSED_DIR'] = 'data/processed'
os.environ['MODELS_DIR'] = 'models'
os.environ['MEDIA_DIR'] = 'webapp/media'

# Append project root to sys.path so we can import the local src package
sys.path.append('/kaggle/working')

print("✓ Kaggle environment paths initialized.")
```

### Cell 2: Git Clone / Upload Source Code
You need to copy your project codebase to `/kaggle/working`. You can do this by using a Kaggle Utility script or clone it from a git repository:

```bash
%%bash
# Clone the repository directly to /kaggle/working (or upload it as a dataset/zip and extract here)
# Example clone:
# git clone https://github.com/YOUR_USERNAME/SHAP-Driven-Career-Predictor.git /tmp/repo
# cp -r /tmp/repo/* /kaggle/working/
# rm -rf /tmp/repo
```
*Note: If you uploaded the source files directly as a Kaggle dataset, copy the files from `/kaggle/input/your-source-code-dataset` to `/kaggle/working`.*

### Cell 3: Install Dependencies
This cell upgrades and pins package dependencies matching the project specifications, preventing Numpy 2.x compile errors with SHAP.

```bash
%%bash
pip install --quiet "numpy<2.0.0" pandas scikit-learn xgboost shap django matplotlib joblib python-dotenv scipy pyngrok
```

### Cell 4: Initialize Directory Structuring
Create all output folders within the writeable working directory.

```python
from src.config import Config
Config.ensure_directories()
print("✓ Pipeline directory trees created successfully.")
```

### Cell 5: Import and Copy Dataset
Copy the raw dataset from the read-only input folder to the writeable project raw data directory.

```python
import shutil

source_path = '/kaggle/input/student-career-dataset/career_dataset_student.csv'
dest_path = '/kaggle/working/data/raw/career_dataset_student.csv'

shutil.copy(source_path, dest_path)
print("✓ Dataset imported to writeable working path.")
```

### Cell 6: Run Data Cleaning and Relabeling
Execute the relabeling process to inject logical correlations into the data.

```python
from scripts.clean_and_relabel import clean_and_relabel
clean_and_relabel()
print("✓ Data cleaned, balanced, and saved.")
```

### Cell 7: Train the XGBoost Model & Compute SHAP Values
Execute the hyperparameter search and extract local/global attributions.

```python
from src.trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Execute complete pipeline (CV Search, Training, Evaluator, Matrix Plot, and SHAP background setup)
model = trainer.run_pipeline()
print("✓ Model training and evaluation complete.")
```

### Cell 8: Start Django Web Server & Expose via Ngrok
To view the web app interface, sign up for a free account at [ngrok.com](https://ngrok.com/) to obtain an Authtoken. Replace `"YOUR_NGROK_AUTHTOKEN_HERE"` below.

```python
from pyngrok import ngrok
import subprocess
import time

# 1. Authtoken configuration
NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"
ngrok.set_auth_token(NGROK_AUTHTOKEN)

# 2. Expose the port
public_url = ngrok.connect(8000)
print(f"★ DJANGO WEB APP RUNNING AT: {public_url} ★")

# 3. Apply SQLite migrations
print("Running database migrations...")
subprocess.run(["python", "webapp/manage.py", "migrate"], cwd="/kaggle/working")
time.sleep(2)

# 4. Start server
django_process = subprocess.Popen(
    ["python", "webapp/manage.py", "runserver", "0.0.0.0:8000", "--noreload"],
    cwd="/kaggle/working"
)
```

---

## 🛠️ Verification & Troubleshooting
* **Is Django throwing a Host header error?**
  We have updated `ALLOWED_HOSTS = ['*']` in `webapp/career_predictor/settings.py` so that requests tunneled via Ngrok are accepted.
* **Is SHAP taking too long?**
  Ensure `SHAP_BACKGROUND_SIZE` is set to a low value (like `20` or `50`) in your configurations if calculations take too long.
