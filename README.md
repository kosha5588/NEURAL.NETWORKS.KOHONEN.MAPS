# SOM Clustering Analysis

This project performs cluster analysis using a Self-Organizing Map (SOM) neural network on multidimensional data. The implementation visualizes both the PCA projection of the data and the average feature values for each identified cluster.

## Features

- **Data Loading & Validation**: Loads learning data and PCA-transformed data with consistency checks
- **Data Preprocessing**: Normalizes features using MinMax scaling
- **SOM Clustering**: Implements a 2x2 Kohonen map for unsupervised clustering
- **Visualization**: 
  - PCA projection colored by cluster assignment
  - Average feature values per cluster
- **Comprehensive Reporting**: Detailed statistics about cluster sizes and characteristics

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn minisom
```

## Project Structure

```
├── som_analysis.py          # Main analysis script
├── Learning_data7.txt       # Input learning data (multidimensional)
├── PCA_data7.txt           # PCA-transformed data (2D projection)
└── analysis_results.png    # Generated visualization output
```

## Usage

1. Prepare your input files:
   - `Learning_data7.txt`: Raw multidimensional data (space-separated)
   - `PCA_data7.txt`: 2D PCA projection of the same data

2. Run the analysis:
```bash
python som_analysis.py
```

3. The script will generate:
   - `analysis_results.png`: Combined visualization plot
   - Console output with cluster statistics

## Output Description

### Visualization
- **Left plot**: PCA projection showing cluster assignments using a color code
- **Right plot**: Average feature values for each cluster, showing characteristic patterns

### Console Output
- Total number of objects processed
- Number of clusters identified
- Size and mean feature values for each cluster

## Parameters

The SOM is configured with:
- Grid size: 2×2 (4 clusters)
- Gaussian neighborhood function
- Sigma: 1.0
- Learning rate: 0.5
- 100 training iterations
- PCA-based weight initialization

## Customization

To adjust the analysis:
- Modify grid dimensions (`x` and `y` parameters)
- Adjust training parameters (`sigma`, `learning_rate`, iterations)
- Change visualization styles (colormaps, markers, etc.)

## Example Output

```
Результаты анализа:
Всего объектов: 150
Выявлено кластеров: 4
Кластер 1: 42 объектов
Средние значения: [5.01 3.42 1.46 0.24]
----------------------------------------
...
```

## Applications

- Pattern recognition in multidimensional data
- Customer segmentation
- Feature analysis and dimensionality reduction
- Data exploration and visualization
