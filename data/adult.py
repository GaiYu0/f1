from itertools import groupby
import sys
import numpy as np

lnn = [ln.strip().split(', ') \
       for ln in open(sys.argv[1]).readlines() + open(sys.argv[2]).readlines()]
roww = list(zip(*lnn))
continuouss = [True, False, True, False, True, False, False, \
              False, False, False, True, True, True, False, False]
ll = [(None if continuous else list(zip(*groupby(sorted(row))))[0]) \
      for row, continuous in zip(roww, continuouss)]
roww = [(list(map(l.index, row)) if l else list(map(float, row))) \
        for row, l in zip(roww, ll)]
roww = [np.eye(len(l))[np.array(row)] if l else np.array(row).reshape([-1, 1]) \
        for row, l in zip(roww, ll)]
x = np.concatenate(roww[:-1], 1)
y = np.argmax(roww[-1], 1)
np.save('x', x.astype(np.float32))
np.save('y', y.astype(np.int))
