# HSKL: hyperspectral-sklearn

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

