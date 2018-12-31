import sys
import numpy as np

xy = np.loadtxt(sys.argv[1], delimiter=',', dtype=np.int)
x, y = xy[:, :54], xy[:, 54]
y = y - 1
np.save('x', x.astype(np.float32))
np.save('y', y.astype(np.int))
