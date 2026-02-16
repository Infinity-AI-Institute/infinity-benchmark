# Infinity Benchmark

A comprehensive benchmark suite for classification algorithms with the original 20 Infinity datasets plus additional complex datasets.

## Overview

This repository provides a standardized dataset loader and baseline benchmarking tools for evaluating classification models across built-in sklearn, OpenML, and sklearn fetched datasets.

### Datasets

**Built-in (scikit-learn):**
- Iris
- Wine
- Breast Cancer
- Digits

**OpenML Datasets:**
- Balance Scale
- Blood Transfusion
- Haberman
- Seeds
- Teaching Assistant
- Zoo
- Planning Relax
- Ionosphere
- Sonar
- Glass
- Vehicle
- Liver Disorders
- Heart Statlog
- Pima Indians Diabetes
- Australian
- Monks-1

**Additional Complex Datasets:**
- Adult Census Income (OpenML)
- Credit-g (OpenML)
- Bank Marketing (OpenML)
- Electricity (OpenML)
- Phoneme (OpenML)
- Satimage (OpenML)
- Madelon (OpenML)
- Amazon Employee Access (OpenML)
- Covertype (sklearn fetcher)
- KDDCup99 (sklearn fetcher)

## Installation

```bash
pip install -r requirements.txt
```

## Usage
see ```example.ipynb``` for executable code example.

### Load and Access Specific Datasets

```python
from data_loader import load

# Load all available datasets
datasets = load()

# Load specific datasets
datasets = load(["Iris", "Wine", "Breast Cancer"])

# Access specific dataset details
X_train, X_test, y_train, y_test = datasets["Iris"]
```

### Benchmark Your Model

```python
from data_loader import (
    test_on_all_datasets,
    test_on_datasets,
    test_on_infinity_benchmark,
)

model = YourModel()  # Your model here

# Test on the fixed Infinity 20 datasets
scores = test_on_infinity_benchmark(model)

# Test on all available datasets (Infinity + additional)
scores = test_on_all_datasets(model)

# Test on specific datasets
scores = test_on_datasets(model, ["Iris", "Wine", "Covertype"])
```

### Manual Per Dataset Benchmarking 

```python
from data_loader import load_classification_datasets
from sklearn.metrics import accuracy_score
import numpy as np

datasets = load_classification_datasets([...], logging=True)

for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
    model = YourModel()  # Your model here
    model.fit(np.asarray(X_train), y_train)
    y_pred = model.predict(np.asarray(X_test))
    score = accuracy_score(y_test, y_pred)
    print(f"{dataset_name}: {score:.4f}")
```

## API Reference

### `load(dataset_names=None, test_size=0.2, random_state=42, logging=False)`

Load datasets from the Infinity Benchmark.

**Parameters:**
- `dataset_names` (list, optional): Names of datasets to load. If None or empty, loads all available datasets.
- `test_size` (float): Train/test split ratio (default: 0.2)
- `random_state` (int): Random seed (default: 42)
- `logging` (bool): Print dataset loading info (default: False)

**Returns:** Dictionary mapping dataset names to (X_train, X_test, y_train, y_test) tuples

### `test_on_infinity_benchmark(model)`

Test a model on the fixed Infinity Benchmark datasets (original 20).

**Parameters:**
- `model`: A scikit-learn compatible model with `.fit()` and `.predict()` methods

**Returns:** Dictionary mapping dataset names to accuracy scores

### `test_on_all_datasets(model)`

Test a model on all available datasets.

**Parameters:**
- `model`: A scikit-learn compatible model with `.fit()` and `.predict()` methods

**Returns:** Dictionary mapping dataset names to accuracy scores

### `test_on_datasets(model, dataset_names)`

Test a model on a specific dataset list.

**Parameters:**
- `model`: A scikit-learn compatible model with `.fit()` and `.predict()` methods
- `dataset_names` (list): Explicit dataset names to test

**Returns:** Dictionary mapping dataset names to accuracy scores

## Notes

- The built-in datasets (Iris, Wine, etc.) are downloaded automatically by scikit-learn
- OpenML datasets are fetched on first use and cached locally
- All data is converted to float32 for numerical stability
- Missing features/categorical variables are handled automatically
