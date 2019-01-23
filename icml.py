import os
import sys

d = os.fsencode(sys.argv[1])
for f in os.listdir(d):
    print(type(f))
