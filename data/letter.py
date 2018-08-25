import sys
import numpy as np

lnn = open(sys.argv[1]).readlines()
A = ord('A')
converter = lambda x : ord(x.decode("utf-8")) - A
xy = np.loadtxt(sys.argv[1], converters={0 : converter}, delimiter=',', dtype=np.int)
x, y = xy[:, 1:], xy[:, 0]
np.save('x', x)
np.save('y', y)
