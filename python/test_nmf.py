import numpy as np
from sklearn.decomposition import NMF

X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

ans = np.matmul(W, H)
print(X)
print()
print(ans)