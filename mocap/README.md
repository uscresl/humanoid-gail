# Feature Extraction from MoCap Data
This code computes the 5 3D-endeffector features that are the normalized direction vectors from the humanoid's root to head, hands and feet.

The features are computed as follows:
```python
from load_mocap import load_features

features = load_features("examples/12.asf", "examples/02_01.amc")
# This is a np.array with one row for every frame in the AMC animation file.
# Each row stores the flattened 5x3 features (15-dimensional).
```