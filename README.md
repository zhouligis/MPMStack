# 🚀MPMStack: Mineral Prospectivity Mapping with Stacking Ensemble

A comprehensive machine learning pipeline for geological exploration that combines multiple algorithms using stacking ensemble method for mineral prospectivity mapping.

## Features

- **Multi-Algorithm Ensemble**: Combines SVM, Random Forest, XGBoost, and Naive Bayes using stacking
- **Automated Feature Selection**: VIF-based multicollinearity removal and information gain filtering
- **Raster Data Processing**: Efficient handling of large geospatial raster datasets
- **Imbalanced Data Handling**: Built-in class weighting and stratified sampling
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1-score, precision, recall, and AUC
- **Raster Prediction**: Generate full-coverage mineral prospectivity maps
- **Modular Design**: Easy to customize and extend for different geological contexts

## Quick Start

### Prerequisites

- Python 3.7 or higher
- GDAL/OGR libraries for geospatial data processing

### Installation

1. Clone the repository:
```bash
git clone https://github.com/zhouligis/MPMStack.git
cd MPMStack
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from MPMStack import MPMStack

# Initialize the pipeline
mpm = MPMStack(data_dir='data', results_dir='results')

# Run complete pipeline
results = mpm.run_complete_pipeline(
    raster_dir='data/raster',
    shapefile_path='data/shape/mineral_points.shp',
    target_column='Group'
)

# Print results
for model_name, metrics in results.items():
    print(f"{model_name} - Test Accuracy: {metrics['test_accuracy']:.4f}")
```

## 📁 Data Structure

Organize your data in the following structure:

```
project/
├── data/
│   ├── raster/           # Raster files (.tif)
│   │   ├── feature1.tif
│   │   ├── feature2.tif
│   │   └── ...
│   └── shape/            # Vector files
│       └── mineral_points.shp
├── results/              # Output directory (auto-created)
│   ├── models/          # Trained models
│   ├── predictions/     # Prediction rasters
│   └── reports/         # Evaluation reports
└── MPMStack.py
```

### Input Data Requirements

#### Raster Data
- **Format**: GeoTIFF (.tif)
- **Projection**: All rasters must have the same CRS, extent, and resolution
- **Features**: Geological, geochemical, geophysical, or remote sensing data
- **NoData**: Properly defined nodata values

#### Vector Data (Mineral Points)
- **Format**: Shapefile (.shp)
- **Geometry**: Point features
- **Required Attributes**:
  - Target column (e.g., 'Group'): Mineral occurrence classification
  - Optional scale column: For multi-scale analysis

## 🔧 Advanced Usage

### Custom Pipeline Configuration

```python
# Initialize with custom settings
mpm = MPMStack(
    data_dir='custom_data',
    results_dir='custom_results',
    fast_mode=True  # For quick testing
)

# Load data
mpm.load_raster_data('path/to/rasters')
mpm.load_mineral_points('path/to/points.shp', target_column='Mineral_Type')

# Extract features
mpm.extract_features_from_points()

# Custom feature selection
mpm.feature_selection(vif_threshold=5.0, info_gain_threshold=0.02)

# Prepare data with custom split
X_train, X_val, X_test, y_train, y_val, y_test = mpm.prepare_training_data(
    test_size=0.3, val_size=0.2
)

# Train models
mpm.train_models(X_train, X_val, X_test, y_train, y_val, y_test)

# Evaluate
results = mpm.evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)

# Generate predictions
predictions, probabilities = mpm.predict_raster(model_name='Stacking')
```

### Model Loading and Prediction

```python
# Load pre-trained models
mpm = MPMStack(results_dir='existing_results')
mpm.load_models()
mpm.load_raster_data('new_raster_data')

# Generate predictions for new area
predictions, probabilities = mpm.predict_raster(
    model_name='RandomForest',
    output_filename='new_area_predictions.tif'
)
```

## Output Files

### Models
- `{ModelName}_model.pkl`: Trained model files
- `scaler.pkl`: Feature scaler
- `label_encoder.pkl`: Target label encoder

### Predictions
- `{ModelName}_predictions.tif`: Class predictions
- `{ModelName}_probabilities.tif`: Prediction probabilities

### Reports
- `model_evaluation.csv`: Comprehensive model metrics
- `evaluation_report.txt`: Detailed evaluation summary
- `feature_selection_report.txt`: Feature selection results
- `selected_features.txt`: List of selected features
- `mpmstack.log`: Processing log

## Key Parameters

### Feature Selection
- `vif_threshold` (default: 10.0): Variance Inflation Factor threshold for multicollinearity
- `info_gain_threshold` (default: 0.01): Information gain threshold for feature importance

### Data Splitting
- `test_size` (default: 0.2): Proportion of data for testing
- `val_size` (default: 0.2): Proportion of training data for validation
- `random_state` (default: 42): Random seed for reproducibility

### Prediction
- `chunk_size` (default: 1000): Pixels processed simultaneously (adjust based on memory)
- `model_name`: Model to use for prediction ('SVM', 'RandomForest', 'XGBoost', 'NaiveBayes', 'Stacking')

## Methodology

### Machine Learning Pipeline

1. **Data Loading**: Automated raster alignment and point data integration
2. **Feature Extraction**: Spatial sampling of raster values at point locations
3. **Feature Selection**: 
   - VIF analysis for multicollinearity removal
   - Mutual information for relevance filtering
4. **Model Training**: 
   - Individual model optimization with GridSearchCV
   - Stacking ensemble with cross-validation
5. **Evaluation**: Comprehensive metrics on stratified test set
6. **Prediction**: Memory-efficient raster processing

### Supported Algorithms

- **Support Vector Machine (SVM)**: Non-linear classification with RBF kernel
- **Random Forest**: Ensemble of decision trees with bootstrap aggregating
- **XGBoost**: Gradient boosting with advanced regularization
- **Naive Bayes**: Probabilistic classifier assuming feature independence
- **Stacking Ensemble**: Meta-learning approach combining all base models

## Performance Optimization

### Memory Management
- Chunked raster processing for large datasets
- Efficient data types (float32 for rasters)
- Garbage collection between processing steps

### Speed Optimization
- Parallel processing with joblib
- Fast mode for quick prototyping
- Optimized feature selection algorithms

## API Reference

### MPMStack Class

#### Initialization
```python
MPMStack(data_dir='data', results_dir='results', fast_mode=False)
```

#### Core Methods

- `load_raster_data(raster_dir=None)`: Load and align raster datasets
- `load_mineral_points(shapefile_path=None, target_column='Group', scale_column='Scale')`: Load mineral occurrence points
- `extract_features_from_points()`: Extract raster values at point locations
- `feature_selection(vif_threshold=10.0, info_gain_threshold=0.01)`: Automated feature selection
- `prepare_training_data(test_size=0.2, val_size=0.2, random_state=42)`: Split and scale data
- `train_models(X_train, X_val, X_test, y_train, y_val, y_test)`: Train ensemble models
- `evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)`: Comprehensive evaluation
- `predict_raster(model_name='Stacking', output_filename=None, chunk_size=1000)`: Generate predictions
- `save_models()`: Save trained models to disk
- `load_models()`: Load pre-trained models
- `run_complete_pipeline()`: Execute full pipeline

## Example Workflows

### Workflow 1: Complete Pipeline
```python
from MPMStack import MPMStack

# Run everything in one go
mpm = MPMStack()
results = mpm.run_complete_pipeline()
print(f"Best model: {max(results.keys(), key=lambda k: results[k]['test_f1'])}")
```

### Workflow 2: Step-by-Step Analysis
```python
from MPMStack import MPMStack

# Initialize
mpm = MPMStack(fast_mode=True)

# Load and explore data
mpm.load_raster_data()
mpm.load_mineral_points()
print(f"Loaded {len(mpm.feature_names)} features")
print(f"Loaded {len(mpm.mineral_points)} mineral points")

# Feature engineering
mpm.extract_features_from_points()
mpm.feature_selection(vif_threshold=5.0)
print(f"Selected {len(mpm.selected_features)} features")

# Model training and evaluation
data = mpm.prepare_training_data()
mpm.train_models(*data)
results = mpm.evaluate_models(*data)

# Generate predictions
mpm.predict_raster(model_name='Stacking')
```

### Workflow 3: Model Comparison
```python
from MPMStack import MPMStack
import pandas as pd

# Train models
mpm = MPMStack()
results = mpm.run_complete_pipeline()

# Compare models
comparison = pd.DataFrame(results).T[['test_accuracy', 'test_f1', 'test_auc']]
print(comparison.sort_values('test_f1', ascending=False))

# Generate predictions with best model
best_model = comparison.sort_values('test_f1', ascending=False).index[0]
mpm.predict_raster(model_name=best_model)
```

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/MPMStack.git
cd MPMStack

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## Citation

If you use MPMStack in your research, please cite:

```bibtex
@software{mpmstack2025,
  title={MPMStack: Mineral Prospectivity Mapping with Stacking Ensemble},
  author={Li Zhou},
  year={2025},
  url={https://github.com/zhouligis/MPMStack}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/MPMStack/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/MPMStack/discussions)
- **Email**: zhouligis@163.com

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [rasterio](https://rasterio.readthedocs.io/)
- Inspired by advances in ensemble learning for geoscience applications
- Thanks to the open-source geospatial community

## 📋 Dependencies

Create a `requirements.txt` file with the following dependencies:

```
numpy>=1.19.0
pandas>=1.2.0
geopandas>=0.9.0
rasterio>=1.2.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.0.0
tqdm>=4.60.0
statsmodels>=0.12.0
```
