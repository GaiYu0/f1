import os
import sys
from event_file_loader import EventFileLoader

def extract(path):
    loader = EventFileLoader(path)
    x = []
    for e in loader.Load():
        try:
            if e.summary.value[0].tag == sys.argv[2]:
                x.append(e.summary.value[0].simple_value)
        except (AttributeError, IndexError):
            pass
    return x

b = extract('%s/b/%s' % (sys.argv[1], os.listdir('%s/b' % sys.argv[1])[0]))
c = extract('%s/c/%s' % (sys.argv[1], os.listdir('%s/c' % sys.argv[1])[0]))
K = 1
print(sum(c[b.index(x)] for x in sorted(b, reverse=True)[:K]) / K)
