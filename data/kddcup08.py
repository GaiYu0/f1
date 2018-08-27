import sys
import numpy as np

x = np.loadtxt(sys.argv[1], delimiter='\t ')
y = np.loadtxt(sys.argv[2], delimiter='\t  ')[:, 0]
np.save('x', x.astype(np.float32))
np.save('y', y.astype(np.int))
