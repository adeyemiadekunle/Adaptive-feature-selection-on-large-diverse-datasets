# Adaptive Feature Selection on Large Diverse Datasets

This repository contains assessment and research materials for an adaptive heuristic feature selection framework designed for large and diverse machine learning datasets.

## Contents

- `Heuristic-Framework-v2.1/` - Python implementation, notebooks, requirements, experiment outputs, and statistical test results.
- `ARP Presentation.pptx` and `Heuristic_Adaptive_FS_Presentation.pptx` - presentation materials.
- `LDS7007M - 240254665.docx` and `Publication-Draft.docx` - written research and publication draft materials.

## Project Summary

The framework profiles dataset characteristics such as dimensionality ratio, class imbalance, feature correlation, variance structure, and sparsity. It then applies rule-based heuristics to choose an appropriate feature selection strategy, including filter, wrapper, and embedded approaches.

## Setup

For the framework:

```bash
cd Heuristic-Framework-v2.1
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
from heuristic_feature_selector import HeuristicFeatureSelector

selector = HeuristicFeatureSelector(verbose=True)
result = selector.fit_transform(X, y)

print(result.selected_features)
print(result.meta_profile)
```

## Tests

```bash
cd Heuristic-Framework-v2.1
jupyter notebook experiment.ipynb
```

## Notes

The root `.gitignore` excludes local virtual environments, Python caches, build artifacts, and raw or processed generated data. Recreate datasets locally before running experiments that depend on them.
