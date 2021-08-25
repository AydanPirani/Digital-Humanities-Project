import numpy as np

m = np.array([[1,2],[3,4]])

print(m)

m = np.rot90(m, k=1, axes=(1,0))

print(m)