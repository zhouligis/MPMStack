# 🚀MPMStack: Ensemble Machine Learning with Uncertainty Quantification for Mineral Prospectivity Mapping

Code for the research paper currently under review:
“Ensemble Machine Learning with Uncertainty Quantification for 
Mineral Prospectivity Mapping in Laojunshan Mineral Concentrated Area, Southwest China ”.

Manuscript Number:ACAGS-D-25-00145

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

- Python 3.9 or higher
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

## Data Structure

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

-- **Naive Bayes**: Probabilistic classifier assuming feature independence
- **Support Vector Machine (SVM)**: Non-linear classification with RBF kernel
- **Random Forest**: Ensemble of decision trees with bootstrap aggregating
- **XGBoost**: Gradient boosting with advanced regularization
- **Stacking Ensemble**: Meta-learning approach combining all base models

## Example Workflows

```python
from MPMStack import MPMStack

# Run everything in one go
mpm = MPMStack()
results = mpm.run_complete_pipeline()
print(f"Best model: {max(results.keys(), key=lambda k: results[k]['test_f1'])}")
```

## Citation

If you use MPMStack in your research, please cite:

```bibtex
@software{mpmstack2025,
  title={Ensemble Machine Learning with Uncertainty Quantification for Mineral Prospectivity Mapping in Laojunshan Mineral Concentrated Area, Southwest China},
  author={Li Zhou},
  year={2025},
  url={https://github.com/zhouligis/MPMStack}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/zhouligis/MPMStack/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zhouligis/MPMStack/discussions)
- **Email**: zhouligis@163.com

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [rasterio](https://rasterio.readthedocs.io/)
- Thanks to the open-source geospatial community

