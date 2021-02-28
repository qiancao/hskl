# HSKL: hyperspectral-scikit-learn

Hyperspectral image analysis using *scikit-learn*

## Usage

The package can be imported as `hskl`

```python
import hskl.classification as classification

# Load training, testing, and label images
train, test, label = ...

# Train a classifier and predict test image
cl = classification.HyperspectralClassifier(
         method_name=”RandomForest”)
cl.fit(train, label)
prediction = cl.predict(test)
}
```

1. Shape of `train` and `test` arrays are (DimX,DimY,SpectralChannels)
2. Shape of `label` and `prediction` arrays are (DimX,DimY)
3. Dimension(s) of `train` and `label` must be consistent: `train.shape[0] == label.shape[0]` and `train.shape[1] == label.shape[1]`
4. Inputs: `train`, `test`, and `label` can also be a list of `np.ndarray`s with each element satisfying 1, 2 and 3.

# Planned Features

Support for:
* Pipeline API
* Grid Search Cross Validation
