# HSKL: Hyperspectral-scikit-learn

Hyperspectral image analysis using *scikit-learn*

## Installation

The package can be installed from `pip`:

`pip install hskl`

## Usage

Training a pixel-level classifier for segmentation:

```python
import hskl.classification as classification
import hskl.utils as utils

# List method names
methods = classification.list_methods()

# Load training, testing, and label images (numpy.ndarray)
train, test, label = ...

# Dimensional reduction using PCA, retain 80% image variance
pca = utils.pca_fit(train)
train = utils.pca_apply(train, pca, 0.8)
test = utils.pca_apply(test, pca, 0.8)

# Train a classifier and predict test image labels
cl = classification.HyperspectralClassifier(
         method_name=”RandomForest”,
         method_params={"max_depth": 2})
cl.fit(train, label)
prediction = cl.predict(test)

# Visualization of classification result overlaid with original image
fig_objs = utils.overlay(test,prediction)

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
* Test scripts and data
* Grid search cross validation

In the long-term, support for:
* Pipelines
* Patch-based featurizer
* Dask-enabled parallelism


## Acknowledgement

Some functionalities in this package are provided by Spectral Python (SPy): https://github.com/spectralpython/spectral
