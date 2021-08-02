import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA

X, _ = load_digits(return_X_y=True)

print(X.shape)
X = np.rot90(X)
print(X.shape)


transformer = KernelPCA(n_components=1, kernel='linear')
X_transformed = transformer.fit_transform(X)

print(X_transformed.shape)