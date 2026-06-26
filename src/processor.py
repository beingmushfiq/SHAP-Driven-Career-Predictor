"""
Data Processing Pipeline for Career Predictor.

Handles:
- Data loading and merging
- Cleaning (whitespace, casing, NaN)
- Encoding (LabelEncoder for categorical features)
- Schema locking (feature order is immutable after training)
- Train/test splitting with stratification

Design Decisions:
- LabelEncoder chosen over OneHotEncoder to keep feature count manageable for SHAP
- Feature order locked in feature_schema.json after first training
- Encoders persisted per-feature for inference consistency
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from difflib import SequenceMatcher

from src.config import Config
from src.utils import get_logger, set_seeds
from src.feature_engineer import engineer_features

logger = get_logger(__name__, Config.LOG_DIR)


class DataProcessor:
    """
    End-to-end data processing pipeline.

    Usage:
        processor = DataProcessor()
        X, y, feature_names = processor.process_pipeline()
    """

    def __init__(self):
        """Initialize processor with config and empty encoder storage."""
        self.config = Config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder: Optional[LabelEncoder] = None
        self.feature_names: list = []
        set_seeds(Config.RANDOM_SEED)

    def load_data(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load raw data from CSV files in the data directory.

        If multiple CSVs exist, they are concatenated vertically.

        Args:
            path: Specific file path. If None, loads all CSVs from DATA_RAW_DIR.

        Returns:
            Raw DataFrame.

        Raises:
            FileNotFoundError: If no CSV files found.
        """
        if path is not None:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            logger.info(f"Loading data from {path}")
            return pd.read_csv(path)

        main_data_file = Config.DATA_RAW_DIR / 'career_dataset_student.csv'
        if not main_data_file.exists():
            raise FileNotFoundError(
                f"Main dataset not found at {main_data_file}."
            )

        logger.info(f"Loading main dataset: {main_data_file.name}")
        df = pd.read_csv(main_data_file)
        logger.info(f"Dataset shape: {df.shape}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data.

        Steps:
            1. Strip whitespace from column names
            2. Normalize column names to snake_case
            3. Strip whitespace from string values
            4. Normalize string casing (title case for categoricals)
            5. Drop rows with >50% missing values
            6. Drop duplicate rows (exact + fuzzy match on key columns)
            7. Report data quality metrics

        Args:
            df: Raw DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        original_shape = df.shape
        logger.info(f"Cleaning data (shape: {original_shape})")

        # Normalize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True)
            .str.strip('_')
        )

        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include='object').columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()

        # Drop rows with >50% missing
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=int(threshold))

        # Replace 'nan' strings with actual NaN
        df = df.replace({'nan': np.nan, 'NaN': np.nan, '': np.nan})

        # Drop exact duplicates
        duplicates_exact = df.duplicated().sum()
        df = df.drop_duplicates()

        # Fuzzy duplicate detection (skip for large datasets — exact dedup already ran)
        if Config.TARGET_COLUMN in df.columns and len(df) <= 2000:
            duplicate_indices = self._detect_fuzzy_duplicates(df)
            if len(duplicate_indices) > 0:
                df = df.drop(index=duplicate_indices).reset_index(drop=True)
                logger.info(f"  Detected and removed {len(duplicate_indices)} fuzzy duplicates")
        
        logger.info(
            f"Cleaning complete: {original_shape} -> {df.shape} "
            f"(dropped {original_shape[0] - df.shape[0]} rows, exact duplicates: {duplicates_exact})"
        )
        return df.reset_index(drop=True)

    def _filter_unsupported_careers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows whose career target maps to an unsupported domain
        or whose field of study is no longer supported."""
        target_col = Config.TARGET_COLUMN
        unsupported = Config.UNSUPPORTED_CAREER_CLUSTERS
        original_len = len(df)

        # Filter unsupported career targets
        if target_col in df.columns:
            def _is_unsupported(career: str) -> bool:
                career = str(career).strip()
                cluster = Config.get_cluster_for_career(career)
                if cluster in unsupported:
                    return True
                if career in unsupported:
                    return True
                return False

            mask = df[target_col].astype(str).apply(_is_unsupported)
            df = df[~mask].reset_index(drop=True)

        # Filter unsupported fields of study
        supported_fields = set(f.lower() for f in Config.CATEGORICAL_OPTIONS.get('field', []))
        if 'field' in df.columns and supported_fields:
            field_mask = df['field'].astype(str).str.strip().str.lower().isin(supported_fields)
            df = df[field_mask].reset_index(drop=True)

        dropped = original_len - len(df)
        if dropped > 0:
            logger.info(f"Filtered {dropped} rows with unsupported career domains or fields")
        return df

    def _detect_fuzzy_duplicates(self, df: pd.DataFrame, similarity_threshold: float = 0.95) -> list:
        """
        Detect fuzzy duplicates by comparing ALL feature columns.
        
        Uses blocking by target column to limit comparisons to within the same
        career class. Compares both string columns (SequenceMatcher) and numeric
        columns (normalised absolute difference).
        
        Args:
            df: DataFrame to check
            similarity_threshold: Threshold for considering rows as duplicates (0-1)
            
        Returns:
            List of indices to drop
        """
        indices_to_drop = []
        
        target_col = Config.TARGET_COLUMN
        compare_cols = [c for c in df.columns if c != target_col]
        if not compare_cols:
            return indices_to_drop
        
        str_cols = [c for c in compare_cols if pd.api.types.is_string_dtype(df[c])]
        num_cols = [c for c in compare_cols if pd.api.types.is_numeric_dtype(df[c])]
        
        # Pre-compute numeric ranges for normalisation
        num_ranges = {}
        for col in num_cols:
            col_range = df[col].max() - df[col].min()
            num_ranges[col] = col_range if col_range > 0 else 1.0
        
        # Block by target column
        if target_col in df.columns:
            groups = df.groupby(target_col).groups
        else:
            groups = {'all': df.index.tolist()}
        
        for group_label, group_indices in groups.items():
            group_indices = list(group_indices)
            n = len(group_indices)
            if n < 2:
                continue
            
            max_comparisons = 100_000
            pairs_done = 0
            
            for i_pos in range(n):
                if pairs_done >= max_comparisons:
                    break
                idx_i = group_indices[i_pos]
                for j_pos in range(i_pos + 1, n):
                    if pairs_done >= max_comparisons:
                        break
                    pairs_done += 1
                    idx_j = group_indices[j_pos]
                    
                    similarities = []
                    # String column similarity
                    for col in str_cols:
                        val_i = str(df.at[idx_i, col]).lower()
                        val_j = str(df.at[idx_j, col]).lower()
                        similarities.append(SequenceMatcher(None, val_i, val_j).ratio())
                    # Numeric column similarity (1 - normalised absolute diff)
                    for col in num_cols:
                        diff = abs(df.at[idx_i, col] - df.at[idx_j, col]) / num_ranges[col]
                        similarities.append(1.0 - min(diff, 1.0))
                    
                    avg_similarity = np.mean(similarities) if similarities else 0
                    if avg_similarity >= similarity_threshold:
                        indices_to_drop.append(idx_j)
        
        return list(set(indices_to_drop))

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using multiple strategies.

        Strategy:
            - Numerical features: KNN imputation (k=5) if enabled, else median
            - Categorical features: mode imputation

        Args:
            df: DataFrame with potential missing values.

        Returns:
            DataFrame with no missing values.
        """
        missing_before = df.isnull().sum().sum()
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        # Numerical imputation
        if numerical_cols and missing_before > 0:
            if Config.ENABLE_KNN_IMPUTATION and len(df) > 5:
                logger.info("  Using KNN imputation (k=5) for numerical features")
                knn_imputer = KNNImputer(n_neighbors=5)
                df[numerical_cols] = knn_imputer.fit_transform(df[numerical_cols])
            else:
                logger.info("  Using median imputation for numerical features")
                for col in numerical_cols:
                    if df[col].isnull().sum() > 0:
                        fill_val = df[col].median()
                        df[col] = df[col].fillna(fill_val)
                        logger.info(f"    Imputed {df[col].isnull().sum()} missing values in '{col}' using median")
        
        # Categorical imputation
        for col in categorical_cols:
            if df[col].isnull().sum() == 0:
                continue
            missing_count = df[col].isnull().sum()
            fill_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(fill_val)
            logger.info(f"  Imputed {missing_count} missing values in '{col}' using mode (value={fill_val})")
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.

        Each feature gets its own LabelEncoder, stored in self.label_encoders.
        Unknown categories during inference are mapped to the most frequent class.

        Args:
            df: Cleaned DataFrame with raw categorical values.

        Returns:
            DataFrame with all features encoded as integers.
        """
        logger.info("Encoding categorical features...")

        # Identify categorical columns (excluding target)
        target_col = Config.TARGET_COLUMN
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"  Encoded '{col}': {len(le.classes_)} classes")

        # Encode target variable separately
        if target_col in df.columns and not pd.api.types.is_numeric_dtype(df[target_col]):
            if Config.USE_CAREER_CLUSTERS:
                df[target_col] = df[target_col].apply(Config.get_cluster_for_career)
                logger.info("Mapped careers to career clusters")
            self.target_encoder = LabelEncoder()
            df[target_col] = self.target_encoder.fit_transform(df[target_col].astype(str))
            logger.info(
                f"  Encoded target '{target_col}': "
                f"{len(self.target_encoder.classes_)} classes"
            )

        return df

    def create_schema(self, feature_names: list, save: bool = True) -> dict:
        """
        Create and optionally save the feature schema.

        The schema locks the feature order — it MUST NOT change after training.

        Args:
            feature_names: Ordered list of feature column names.
            save: Whether to save to disk.

        Returns:
            Schema dictionary.
        """
        schema = {
            'version': '1.0.0',
            'feature_count': len(feature_names),
            'feature_names': feature_names,
            'feature_types': {},
        }

        for name in feature_names:
            if name in Config.NUMERICAL_FEATURES:
                schema['feature_types'][name] = 'numerical'
            else:
                schema['feature_types'][name] = 'categorical'

        if save:
            Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            with open(Config.FEATURE_SCHEMA_PATH, 'w') as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Feature schema saved: {Config.FEATURE_SCHEMA_PATH}")

        self.feature_names = feature_names
        return schema

    def prepare_for_training(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare data for model training.

        Separates features from target, validates shape.

        Args:
            df: Fully encoded DataFrame.

        Returns:
            Tuple of (X array, y array, feature_names list).

        Raises:
            ValueError: If target column not found.
        """
        target_col = Config.TARGET_COLUMN
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        drop = set(getattr(Config, 'DROP_FEATURES', []))
        feature_cols = [c for c in df.columns if c != target_col and c not in drop]
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.int64)

        logger.info(f"Prepared data: X={X.shape}, y={y.shape}, features={len(feature_cols)}")
        return X, y, feature_cols

    def save_encoders(self) -> None:
        """Save all fitted encoders to disk for inference use."""
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.label_encoders, Config.ENCODERS_PATH)
        logger.info(f"Label encoders saved: {Config.ENCODERS_PATH}")

        if self.target_encoder is not None:
            joblib.dump(self.target_encoder, Config.TARGET_ENCODER_PATH)
            logger.info(f"Target encoder saved: {Config.TARGET_ENCODER_PATH}")

    def _generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality metrics report.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values_count': df.isnull().sum().sum(),
            'missing_values_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'class_distribution': {},
        }
        
        # Class distribution for target
        if Config.TARGET_COLUMN in df.columns:
            report['class_distribution'] = df[Config.TARGET_COLUMN].value_counts().to_dict()
        
        if Config.ENABLE_DATA_QUALITY_REPORT:
            logger.info("=" * 60)
            logger.info("DATA QUALITY REPORT")
            logger.info("=" * 60)
            logger.info(f"Total Rows: {report['total_rows']}")
            logger.info(f"Total Columns: {report['total_columns']}")
            logger.info(f"Missing Values: {report['missing_values_count']} ({report['missing_values_pct']:.2f}%)")
            logger.info(f"Duplicate Rows: {report['duplicate_rows']}")
            if report['class_distribution']:
                logger.info("Class Distribution:")
                for cls, count in sorted(report['class_distribution'].items(), key=lambda x: x[1], reverse=True):
                    pct = (count / report['total_rows']) * 100
                    logger.info(f"  {cls}: {count} ({pct:.1f}%)")
            logger.info("=" * 60)
        
        return report

    def process_pipeline(
        self, data_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Execute the full data processing pipeline.

        Steps:
            1. Load data
            2. Clean data
            3. Impute missing values
            4. Encode categorical features
            5. Create and save feature schema
            6. Save encoders
            7. Prepare X, y arrays

        Args:
            data_path: Optional specific data file path.

        Returns:
            Tuple of (X, y, feature_names).
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("=" * 60)

        # 1. Load
        df = self.load_data(data_path)

        # 2. Clean
        df = self.clean_data(df)
        
        # 2.5 Filter unsupported career domains
        df = self._filter_unsupported_careers(df)
        
        # 2.6 Data Quality Report
        quality_report = self._generate_data_quality_report(df)

        # 3. Impute
        df = self._impute_missing(df)

        # 3.5 Feature Engineering
        df = engineer_features(df)
        logger.info(f"Engineered features added. Shape: {df.shape}")

        # 4. Encode
        df = self.encode_features(df)

        # 5. Prepare
        X, y, feature_names = self.prepare_for_training(df)

        # 6. Schema
        self.create_schema(feature_names)

        # 7. Save encoders
        self.save_encoders()

        # 8. Save processed data
        Config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        processed_path = Config.DATA_PROCESSED_DIR / 'processed_data.csv'
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved: {processed_path}")

        logger.info("DATA PROCESSING PIPELINE COMPLETE")
        logger.info("=" * 60)
        return X, y, feature_names


def encode_single_input(
    input_dict: dict,
    label_encoders: dict,
    feature_schema: dict,
) -> np.ndarray:
    """
    Encode a single input dictionary for inference.

    This function is used by the predictor to transform form data
    into a feature vector matching the training schema.

    Args:
        input_dict: Dictionary of feature_name → raw_value.
        label_encoders: Fitted LabelEncoders from training.
        feature_schema: Feature schema with ordered names.

    Returns:
        1D numpy array matching the training feature order.

    Raises:
        ValueError: If required features are missing.
    """
    # Apply feature engineering
    cleaned_dict = {}
    for k, v in input_dict.items():
        k_clean = k.strip().lower().replace(' ', '_')
        cleaned_dict[k_clean] = [v]
    
    single_df = pd.DataFrame(cleaned_dict)
    single_df = engineer_features(single_df)
    
    input_dict = {col: single_df[col].iloc[0] for col in single_df.columns}

    feature_names = feature_schema['feature_names']
    encoded = np.zeros(len(feature_names), dtype=np.float32)

    for i, feat_name in enumerate(feature_names):
        if feat_name not in input_dict:
            raise ValueError(f"Missing required feature: '{feat_name}'")

        raw_value = input_dict[feat_name]

        if feat_name in label_encoders:
            # Categorical feature — use encoder
            le = label_encoders[feat_name]
            str_value = str(raw_value).strip()

            if str_value in le.classes_:
                encoded[i] = le.transform([str_value])[0]
            else:
                # Unknown category: map to first class (most conservative)
                logger.warning(
                    f"Unknown category '{str_value}' for feature '{feat_name}'. "
                    f"Mapping to '{le.classes_[0]}'."
                )
                encoded[i] = 0
        else:
            # Numerical feature — direct cast
            try:
                encoded[i] = float(raw_value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid numerical value for '{feat_name}': {raw_value}"
                ) from e

    return encoded
