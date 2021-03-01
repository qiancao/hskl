# HSKL: Hyperspectral-scikit-learn

Hyperspectral image analysis using *scikit-learn*

## Installation

`pip install hskl`

## Usage

Training a pixel-level classifier for segmentation:

```python
import hskl.classification as classification

# List method names
methods = classification.list_methods()

# Load training, testing, and label images
train, test, label = ...

# Train a classifier and predict test image labels
cl = classification.HyperspectralClassifier(
         method_name=”RandomForest”)
cl.fit(train, label)
prediction = cl.predict(test)

```
Notes:
1. Shape of `train` and `test` arrays are (DimX, DimY, SpectralChannels).
2. Shape of `label` and `prediction` arrays are (DimX, DimY).
3. Labeling convention for classifiers:
         (a) Datatype: `label.dtype == np.uint8`.
         (b) Labeled classes start from integer 1. Pixels with `label == 0` are ignored (masked out).
5. Dimension(s) of `train` and `label` must be consistent: `train.shape[0] == label.shape[0]` and `train.shape[1] == label.shape[1]`.
6. Inputs: `train`, `test`, and `label` can also be lists of `np.ndarray`s with each element satisfying the preceeding requirements.

## Planned Features

In the near-term:
* Example scripts and data
* Grid search cross validation

In the long-term, support for:
* Pipeline API
* Patch-based featurizer
* Dask-enabled parallelism
