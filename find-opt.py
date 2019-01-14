import sys

lnn = open(sys.argv[1]).readlines()
xx = [float(ln.strip().split(': ')[1]) for ln in lnn]
print(lnn[xx.index(max(xx))].strip())
