# HSKL: Hyperspectral-scikit-learn

Hyperspectral image analysis using *scikit-learn*

## Paper

Qian Cao, Deependra Mishra, John Wang, Steven Wang, Helena Hurbon and Mikhail Berezin. HSKL:  A Machine Learning Framework For Hyperspectral Image Analysis. *Proc. IEEE WHISPERS*. IEEE, 2021.

## Installation

The package can be installed from `pip`:

`pip install hskl`

It is also possible to install a latest pre-release version of HSKL directly from the GitHub repository:

1. Verify that git is installed:

    `git --version`

2. Install HSKL:

    `pip install git+https://github.com/qiancao/hskl.git`

## Usage

Training a pixel-level classifier for segmentation:

```python
import os

from hskl.demo import dl_hyrank, load_hyrank
import hskl.classification as classification
import hskl.utils as utils

# Download, unpack, and load HyRANK dataset from current directory.
path = os.getcwd()
if not os.path.exists("HyRANK_satellite"):
    dl_hyrank(path)    
images, labels, _ = load_hyrank(path)

# Dimensional reduction using PCA, retain 99.9% image variance
pca = utils.pca_fit(images[0])
train, _ = utils.pca_apply(images[0], pca, 0.999)
test, _ = utils.pca_apply(images[1], pca, 0.999)
label = labels[0]
test_mask = labels[1]>0

# Train a classifier and predict test image labels
cl = classification.HyperspectralClassifier(
         method_name="LinearDiscriminantAnalysis")
cl.fit(train, label)
prediction = cl.predict(test)

# Visualization of training data, test prediction, and test ground truth
fig_objs_train = utils.overlay(train,label)
utils.save_overlay(fig_objs_train, "hyrank_train.png")

fig_objs_predict = utils.overlay(test,prediction*test_mask)
utils.save_overlay(fig_objs_predict, "hyrank_predict.png")

fig_objs_test = utils.overlay(test,labels[1])
utils.save_overlay(fig_objs_test, "hyrank_test.png")
```
Output:

Training image and ground truth labels:

![Training](https://raw.githubusercontent.com/qiancao/hskl/main/examples/hyrank_train.png)

Test image and ground truth labels:

![Testing Ground Truth](https://raw.githubusercontent.com/qiancao/hskl/main/examples/hyrank_test.png)

Test image and predicted labels:

![Testing Prediction](https://raw.githubusercontent.com/qiancao/hskl/main/examples/hyrank_predict.png)

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
* Deep learning (PyTorch) models

## References

Karantzalos, Konstantinos, Karakizi, Christina, Kandylakis, Zacharias, & Antoniou, Georgia. (2018). HyRANK Hyperspectral Satellite Dataset I (Version v001). Zenodo. http://doi.org/10.5281/zenodo.1222202

Some functionalities in this package are provided by Spectral Python (SPy): https://github.com/spectralpython/spectral
