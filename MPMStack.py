"""
MPMStack: Mineral Prospectivity Mapping with Stacking Ensemble
Code for paper under review:
“Ensemble Machine Learning with Uncertainty Quantification for 
Mineral Prospectivity Mapping in Laojunshan Mineral Concentrated Area, Southwest China ”

Author: Li Zhou (ZhouLiGIS)
License: MIT
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import joblib
from tqdm import tqdm

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb

warnings.filterwarnings('ignore')

class MPMStack:
    """
    Mineral Prospectivity Mapping with Stacking Ensemble
    
    A comprehensive machine learning pipeline for geological exploration
    that combines multiple algorithms using stacking ensemble method.
    """
    
    def __init__(self, data_dir='data', results_dir='results', fast_mode=False):
        """
        Initialize MPMStack pipeline
        
        Args:
            data_dir (str): Directory containing input data
            results_dir (str): Directory for output results
            fast_mode (bool): Enable fast mode for quick testing
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.fast_mode = fast_mode
        
        # Create directories
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data containers
        self.raster_data = {}
        self.raster_meta = None
        self.mineral_points = None
        self.dataset = None
        self.models = {}
        self.feature_names = []
        self.selected_features = []
        self.scaler = None
        
        self.logger.info("MPMStack initialized successfully")
    
    def _create_directories(self):
        """Create necessary directories for the pipeline"""
        dirs = [
            self.results_dir,
            os.path.join(self.results_dir, 'models'),
            os.path.join(self.results_dir, 'predictions'),
            os.path.join(self.results_dir, 'reports')
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.results_dir, 'mpmstack.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_raster_data(self, raster_dir=None):
        """
        Load and align raster data from directory
        
        Args:
            raster_dir (str): Directory containing raster files (.tif)
        """
        if raster_dir is None:
            raster_dir = os.path.join(self.data_dir, 'raster')
        
        self.logger.info(f"Loading raster data from {raster_dir}")
        
        if not os.path.exists(raster_dir):
            raise FileNotFoundError(f"Raster directory not found: {raster_dir}")
        
        raster_files = [f for f in os.listdir(raster_dir) if f.endswith('.tif')]
        
        if not raster_files:
            raise FileNotFoundError(f"No raster files found in {raster_dir}")
        
        self.logger.info(f"Found {len(raster_files)} raster files")
        
        # Load reference raster for alignment
        first_file = os.path.join(raster_dir, raster_files[0])
        with rasterio.open(first_file) as src:
            self.raster_meta = src.meta.copy()
            reference_transform = src.transform
            reference_shape = src.shape
            reference_crs = src.crs
        
        # Load all raster data
        for raster_file in tqdm(raster_files, desc="Loading rasters"):
            file_path = os.path.join(raster_dir, raster_file)
            feature_name = os.path.splitext(raster_file)[0]
            
            try:
                with rasterio.open(file_path) as src:
                    # Check alignment
                    if (src.transform != reference_transform or 
                        src.shape != reference_shape or 
                        src.crs != reference_crs):
                        self.logger.warning(f"Skipping misaligned raster: {raster_file}")
                        continue
                    
                    data = src.read(1).astype(np.float32)
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan
                    
                    self.raster_data[feature_name] = data
                    self.feature_names.append(feature_name)
                    
            except Exception as e:
                self.logger.error(f"Error loading {raster_file}: {e}")
                continue
        
        self.logger.info(f"Successfully loaded {len(self.raster_data)} raster features")
    
    def load_mineral_points(self, shapefile_path=None, target_column='Group', scale_column='Scale'):
        """
        Load mineral points from shapefile
        
        Args:
            shapefile_path (str): Path to shapefile
            target_column (str): Column name for target variable
            scale_column (str): Column name for scale information
        """
        if shapefile_path is None:
            shapefile_path = os.path.join(self.data_dir, 'shape', 'mineral_points.shp')
        
        self.logger.info(f"Loading mineral points from {shapefile_path}")
        
        if not os.path.exists(shapefile_path):
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        
        # Try different encodings
        for encoding in ['utf-8', 'gbk', None]:
            try:
                self.mineral_points = gpd.read_file(shapefile_path, encoding=encoding)
                break
            except:
                continue
        
        if self.mineral_points is None:
            raise ValueError(f"Failed to load shapefile: {shapefile_path}")
        
        self.logger.info(f"Loaded {len(self.mineral_points)} mineral points")
        
        # Check required columns
        if target_column not in self.mineral_points.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Log distribution
        target_dist = self.mineral_points[target_column].value_counts()
        self.logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        self.target_column = target_column
        self.scale_column = scale_column
    
    def extract_features_from_points(self):
        """
        Extract raster feature values at mineral point locations
        """
        self.logger.info("Extracting features from mineral points")
        
        if self.mineral_points is None:
            raise ValueError("Mineral points not loaded")
        
        if not self.raster_data:
            raise ValueError("Raster data not loaded")
        
        # Get point coordinates
        coords = [(point.x, point.y) for point in self.mineral_points.geometry]
        
        feature_data = []
        
        for i, (x, y) in enumerate(tqdm(coords, desc="Extracting features")):
            point_features = {'point_id': i, 'x': x, 'y': y}
            
            # Extract values from each raster
            for feature_name in self.feature_names:
                raster_array = self.raster_data[feature_name]
                
                try:
                    # Convert geographic coordinates to pixel coordinates
                    row, col = rasterio.transform.rowcol(self.raster_meta['transform'], x, y)
                    
                    # Check if coordinates are within raster bounds
                    if (0 <= row < raster_array.shape[0] and 0 <= col < raster_array.shape[1]):
                        value = raster_array[row, col]
                        point_features[feature_name] = value if not np.isnan(value) else 0
                    else:
                        point_features[feature_name] = 0
                        
                except Exception as e:
                    self.logger.warning(f"Error extracting feature {feature_name} for point {i}: {e}")
                    point_features[feature_name] = 0
            
            feature_data.append(point_features)
        
        # Create feature DataFrame
        features_df = pd.DataFrame(feature_data)
        
        # Merge with mineral point attributes
        mineral_attrs = self.mineral_points.drop('geometry', axis=1).reset_index(drop=True)
        self.dataset = pd.concat([features_df, mineral_attrs], axis=1)
        
        self.logger.info(f"Feature extraction completed. Dataset shape: {self.dataset.shape}")
    
    def feature_selection(self, vif_threshold=10.0, info_gain_threshold=0.01):
        """
        Perform feature selection using VIF and information gain
        
        Args:
            vif_threshold (float): VIF threshold for multicollinearity removal
            info_gain_threshold (float): Information gain threshold
        """
        self.logger.info("Starting feature selection")
        
        if self.dataset is None:
            raise ValueError("Dataset not available. Run extract_features_from_points first.")
        
        # Prepare feature matrix and target
        X = self.dataset[self.feature_names].copy()
        y = self.dataset[self.target_column].copy()
        
        # Encode target if necessary
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Remove features with too many missing values
        missing_threshold = 0.5
        missing_ratios = X.isnull().sum() / len(X)
        features_to_keep = missing_ratios[missing_ratios <= missing_threshold].index.tolist()
        X = X[features_to_keep]
        
        self.logger.info(f"Removed {len(self.feature_names) - len(features_to_keep)} features with >50% missing values")
        
        # Fill remaining missing values
        X = X.fillna(X.mean())
        
        # Standardize features for VIF analysis
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # VIF-based feature removal
        self.logger.info(f"Performing VIF analysis with threshold {vif_threshold}")
        
        removed_by_vif = []
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            if len(X_scaled.columns) <= 1:
                break
                
            vif_data = []
            for i, feature in enumerate(X_scaled.columns):
                try:
                    vif_value = variance_inflation_factor(X_scaled.values, i)
                    vif_data.append({'Feature': feature, 'VIF': vif_value})
                except:
                    vif_data.append({'Feature': feature, 'VIF': np.inf})
            
            vif_df = pd.DataFrame(vif_data)
            max_vif = vif_df['VIF'].max()
            
            if max_vif <= vif_threshold:
                break
            
            # Remove feature with highest VIF
            feature_to_remove = vif_df.loc[vif_df['VIF'].idxmax(), 'Feature']
            removed_by_vif.append(feature_to_remove)
            X_scaled = X_scaled.drop(columns=[feature_to_remove])
            
            self.logger.info(f"Removed high VIF feature: {feature_to_remove} (VIF: {max_vif:.2f})")
            iteration += 1
        
        # Information gain-based feature selection
        self.logger.info(f"Performing information gain selection with threshold {info_gain_threshold}")
        
        if len(X_scaled.columns) > 0:
            info_gain_scores = mutual_info_classif(X_scaled, y, random_state=42)
            info_gain_df = pd.DataFrame({
                'feature': X_scaled.columns,
                'info_gain': info_gain_scores
            }).sort_values('info_gain', ascending=False)
            
            # Select features above threshold
            selected_features = info_gain_df[info_gain_df['info_gain'] >= info_gain_threshold]['feature'].tolist()
            removed_by_info_gain = info_gain_df[info_gain_df['info_gain'] < info_gain_threshold]['feature'].tolist()
            
            self.logger.info(f"Removed {len(removed_by_info_gain)} features with low information gain")
            self.logger.info(f"Selected {len(selected_features)} features for modeling")
            
            self.selected_features = selected_features
        else:
            self.logger.warning("No features remaining after VIF analysis")
            self.selected_features = []
        
        # Save feature selection results
        self._save_feature_selection_results(removed_by_vif, removed_by_info_gain)
    
    def _save_feature_selection_results(self, removed_by_vif, removed_by_info_gain):
        """Save feature selection results to files"""
        results_dir = os.path.join(self.results_dir, 'reports')
        
        # Save selected features
        with open(os.path.join(results_dir, 'selected_features.txt'), 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        # Save feature selection report
        with open(os.path.join(results_dir, 'feature_selection_report.txt'), 'w') as f:
            f.write("Feature Selection Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original features: {len(self.feature_names)}\n")
            f.write(f"Features removed by VIF: {len(removed_by_vif)}\n")
            f.write(f"Features removed by information gain: {len(removed_by_info_gain)}\n")
            f.write(f"Final selected features: {len(self.selected_features)}\n\n")
            
            f.write("Removed by VIF:\n")
            for feature in removed_by_vif:
                f.write(f"  - {feature}\n")
            
            f.write("\nRemoved by information gain:\n")
            for feature in removed_by_info_gain:
                f.write(f"  - {feature}\n")
            
            f.write("\nSelected features:\n")
            for feature in self.selected_features:
                f.write(f"  - {feature}\n")
    
    def prepare_training_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Prepare training, validation, and test datasets
        
        Args:
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Preparing training data")
        
        if not self.selected_features:
            raise ValueError("No features selected. Run feature_selection first.")
        
        # Prepare feature matrix and target
        X = self.dataset[self.selected_features].copy()
        y = self.dataset[self.target_column].copy()
        
        # Encode target if necessary
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Split into train and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Training set: {X_train_scaled.shape}")
        self.logger.info(f"Validation set: {X_val_scaled.shape}")
        self.logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Train machine learning models with hyperparameter optimization
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target vectors
        """
        self.logger.info("Starting model training")
        
        # Calculate sample weights for imbalanced data
        sample_weights = compute_sample_weight('balanced', y_train)
        
        # Define base models
        base_models = {
            'SVM': SVC(probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'NaiveBayes': GaussianNB()
        }
        
        # Define parameter grids for hyperparameter tuning
        param_grids = {
            'SVM': {
                'C': [0.1, 1, 10] if self.fast_mode else [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'] if self.fast_mode else ['scale', 'auto', 0.001, 0.01]
            },
            'RandomForest': {
                'n_estimators': [50, 100] if self.fast_mode else [50, 100, 200],
                'max_depth': [5, 10] if self.fast_mode else [5, 10, 15, None],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [50, 100] if self.fast_mode else [50, 100, 200],
                'max_depth': [3, 6] if self.fast_mode else [3, 6, 9],
                'learning_rate': [0.1, 0.2] if self.fast_mode else [0.01, 0.1, 0.2]
            },
            'NaiveBayes': {}
        }
        
        # Train and optimize each model
        cv_folds =  5
        
        for model_name, model in base_models.items():
            self.logger.info(f"Training {model_name}")
            
            if param_grids[model_name]:  # If hyperparameters to tune
                grid_search = GridSearchCV(
                    model, param_grids[model_name], 
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring='f1_weighted', n_jobs=-1
                )
                
                if model_name == 'NaiveBayes':
                    grid_search.fit(X_train, y_train)
                else:
                    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
                
                self.models[model_name] = grid_search.best_estimator_
                self.logger.info(f"{model_name} best params: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
                self.models[model_name] = model
        
        # Create stacking ensemble
        self.logger.info("Creating stacking ensemble")
        
        estimators = [(name, model) for name, model in self.models.items()]
        
        self.models['Stacking'] = StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            n_jobs=-1
        )
        
        self.models['Stacking'].fit(X_train, y_train, sample_weight=sample_weights)
        
        self.logger.info("Model training completed")
    
    def evaluate_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target vectors
        
        Returns:
            dict: Evaluation results for all models
        """
        self.logger.info("Evaluating models")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Evaluating {model_name}")
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # Probabilities (if available)
            try:
                y_train_proba = model.predict_proba(X_train)[:, 1] if len(np.unique(y_train)) == 2 else None
                y_val_proba = model.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) == 2 else None
                y_test_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
            except:
                y_train_proba = y_val_proba = y_test_proba = None
            
            # Calculate metrics
            results = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
                'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
                'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
                'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
                'val_precision': precision_score(y_val, y_val_pred, average='weighted'),
                'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
                'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
                'val_recall': recall_score(y_val, y_val_pred, average='weighted'),
                'test_recall': recall_score(y_test, y_test_pred, average='weighted')
            }
            
            # Add AUC for binary classification
            if y_train_proba is not None:
                results.update({
                    'train_auc': roc_auc_score(y_train, y_train_proba),
                    'val_auc': roc_auc_score(y_val, y_val_proba),
                    'test_auc': roc_auc_score(y_test, y_test_proba)
                })
            
            evaluation_results[model_name] = results
        
        # Save evaluation results
        self._save_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _save_evaluation_results(self, results):
        """Save evaluation results to file"""
        results_df = pd.DataFrame(results).T
        results_path = os.path.join(self.results_dir, 'reports', 'model_evaluation.csv')
        results_df.to_csv(results_path)
        
        # Create detailed report
        report_path = os.path.join(self.results_dir, 'reports', 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, metrics in results.items():
                f.write(f"{model_name}:\n")
                f.write("-" * 20 + "\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        self.logger.info(f"Evaluation results saved to {results_path}")
    
    def save_models(self):
        """Save trained models to disk"""
        models_dir = os.path.join(self.results_dir, 'models')
        
        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Saved scaler to {scaler_path}")
        
        # Save label encoder if exists
        if hasattr(self, 'label_encoder'):
            encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
            joblib.dump(self.label_encoder, encoder_path)
            self.logger.info(f"Saved label encoder to {encoder_path}")
    
    def load_models(self):
        """Load trained models from disk"""
        models_dir = os.path.join(self.results_dir, 'models')
        
        # Load models
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '')
            model_path = os.path.join(models_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
            self.logger.info(f"Loaded {model_name} model from {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Loaded scaler from {scaler_path}")
        
        # Load label encoder
        encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
            self.logger.info(f"Loaded label encoder from {encoder_path}")
    
    def predict_raster(self, model_name='Stacking', output_filename=None, chunk_size=1000):
        """
        Generate predictions for the entire raster area
        
        Args:
            model_name (str): Name of the model to use for prediction
            output_filename (str): Output filename for prediction raster
            chunk_size (int): Number of pixels to process at once
        
        Returns:
            numpy.ndarray: Prediction array
        """
        self.logger.info(f"Generating raster predictions using {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        if not self.selected_features:
            raise ValueError("No selected features available")
        
        model = self.models[model_name]
        
        # Get raster dimensions
        first_raster = list(self.raster_data.values())[0]
        height, width = first_raster.shape
        
        # Initialize prediction array
        predictions = np.full((height, width), np.nan, dtype=np.float32)
        probabilities = np.full((height, width), np.nan, dtype=np.float32)
        
        # Create feature stack
        feature_stack = np.stack([self.raster_data[feature] for feature in self.selected_features], axis=-1)
        
        # Process in chunks to manage memory
        total_pixels = height * width
        processed_pixels = 0
        
        with tqdm(total=total_pixels, desc="Predicting") as pbar:
            for i in range(0, height, chunk_size):
                for j in range(0, width, chunk_size):
                    # Define chunk boundaries
                    i_end = min(i + chunk_size, height)
                    j_end = min(j + chunk_size, width)
                    
                    # Extract chunk
                    chunk = feature_stack[i:i_end, j:j_end, :]
                    chunk_shape = chunk.shape[:2]
                    
                    # Reshape for prediction
                    chunk_reshaped = chunk.reshape(-1, len(self.selected_features))
                    
                    # Find valid pixels (no NaN values)
                    valid_mask = ~np.isnan(chunk_reshaped).any(axis=1)
                    
                    if valid_mask.sum() > 0:
                        # Scale features
                        chunk_scaled = self.scaler.transform(chunk_reshaped[valid_mask])
                        
                        # Make predictions
                        chunk_pred = model.predict(chunk_scaled)
                        
                        # Get probabilities if available
                        try:
                            chunk_proba = model.predict_proba(chunk_scaled)
                            if chunk_proba.shape[1] == 2:  # Binary classification
                                chunk_proba = chunk_proba[:, 1]
                            else:  # Multi-class - use max probability
                                chunk_proba = chunk_proba.max(axis=1)
                        except:
                            chunk_proba = chunk_pred.astype(float)
                        
                        # Fill results
                        pred_chunk = np.full(chunk_reshaped.shape[0], np.nan)
                        proba_chunk = np.full(chunk_reshaped.shape[0], np.nan)
                        
                        pred_chunk[valid_mask] = chunk_pred
                        proba_chunk[valid_mask] = chunk_proba
                        
                        # Reshape back to spatial dimensions
                        pred_chunk = pred_chunk.reshape(chunk_shape)
                        proba_chunk = proba_chunk.reshape(chunk_shape)
                        
                        # Store in output arrays
                        predictions[i:i_end, j:j_end] = pred_chunk
                        probabilities[i:i_end, j:j_end] = proba_chunk
                    
                    # Update progress
                    chunk_pixels = (i_end - i) * (j_end - j)
                    processed_pixels += chunk_pixels
                    pbar.update(chunk_pixels)
        
        # Save predictions
        if output_filename is None:
            output_filename = f'{model_name}_predictions.tif'
        
        self._save_prediction_raster(predictions, output_filename, 'predictions')
        self._save_prediction_raster(probabilities, output_filename.replace('.tif', '_probabilities.tif'), 'probabilities')
        
        self.logger.info(f"Raster prediction completed. Saved to {output_filename}")
        
        return predictions, probabilities
    
    def _save_prediction_raster(self, data, filename, data_type):
        """Save prediction array as raster file"""
        output_dir = os.path.join(self.results_dir, 'predictions')
        output_path = os.path.join(output_dir, filename)
        
        # Update metadata
        meta = self.raster_meta.copy()
        meta.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': np.nan
        })
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data.astype(np.float32), 1)
        
        self.logger.info(f"Saved {data_type} raster to {output_path}")
    
    def run_complete_pipeline(self, raster_dir=None, shapefile_path=None, 
                            target_column='Group', scale_column='Scale'):
        """
        Run the complete MPMStack pipeline
        
        Args:
            raster_dir (str): Directory containing raster files
            shapefile_path (str): Path to mineral points shapefile
            target_column (str): Target variable column name
            scale_column (str): Scale column name
        
        Returns:
            dict: Evaluation results
        """
        self.logger.info("Starting complete MPMStack pipeline")
        
        try:
            # Step 1: Load data
            self.load_raster_data(raster_dir)
            self.load_mineral_points(shapefile_path, target_column, scale_column)
            
            # Step 2: Extract features
            self.extract_features_from_points()
            
            # Step 3: Feature selection
            self.feature_selection()
            
            # Step 4: Prepare training data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data()
            
            # Step 5: Train models
            self.train_models(X_train, X_val, X_test, y_train, y_val, y_test)
            
            # Step 6: Evaluate models
            results = self.evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)
            
            # Step 7: Save models
            self.save_models()
            
            # Step 8: Generate predictions
            self.predict_raster()
            
            self.logger.info("MPMStack pipeline completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """
    Main function to run MPMStack pipeline
    """
    # Initialize pipeline
    mpm = MPMStack(fast_mode=False)
    
    # Run complete pipeline
    results = mpm.run_complete_pipeline()
    
    # Print results summary
    print("\nModel Performance Summary:")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test F1-Score: {metrics['test_f1']:.4f}")
        if 'test_auc' in metrics:
            print(f"  Test AUC: {metrics['test_auc']:.4f}")

if __name__ == "__main__":
    main()