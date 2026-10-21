# Seeds Dataset Clustering

An unsupervised machine learning project that applies **K-Means clustering** to a wheat seeds dataset using geometric seed measurements.

The goal is to group seeds based on similarities in their physical characteristics without using the original class labels during clustering.

## Dataset Features

The dataset includes:

- Area
- Perimeter
- Compactness
- Kernel length
- Kernel width
- Asymmetry coefficient
- Kernel groove length

The original class column is removed before clustering.

## Workflow

The project includes:

- Dataset loading and preprocessing
- Feature selection
- Feature standardization using `StandardScaler`
- K-Means clustering with three clusters
- Dimensionality reduction using PCA
- 2D visualization of the resulting clusters
- Comparison of average feature values across clusters

Three clusters are used because the original dataset contains three wheat varieties, while the class labels themselves are not used as clustering features.

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Files

- `Assignment2SeedsDatasetClustering.py` — clustering and visualization workflow
- `seeds_dataset.txt` — dataset used by the project
