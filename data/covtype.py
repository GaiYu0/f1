import sys
import numpy as np

xy = np.loadtxt(sys.argv[1], delimiter=',', dtype=np.int)
x, y = xy[:, :54], xy[:, 54]
np.save('x', x)
np.save('y', y)
