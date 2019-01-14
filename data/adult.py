import sys
import numpy as np
import pandas as pds

df = pds.read_csv('adult.data', engine='python', header=None, sep=', ', verbose=True)
n_rows = df.shape[0]
columns = []
for k in df.keys():
    v = df[k]
    if v.dtype == np.object:
        collection = set(v)
        n_classes = len(collection)
        mapping = dict(zip(collection, range(n_classes)))
        one_hot = np.zeros([n_rows, n_classes])
        idx = np.array(list(map(mapping.__getitem__, v)))
        one_hot[np.arange(n_rows), idx] = 1
        columns.append(one_hot)
    else:
        columns.append(np.expand_dims(np.array(v), axis=1))

x = np.hstack(columns[:-1])
y = np.argmax(columns[-1], axis=1)
if np.sum(y == 1) / len(y) > 0.5:
    y = np.logical_not(y)

np.save('x', x.astype(np.float32))
np.save('y', y.astype(np.int))
