import sys

ll = open(sys.argv[1]).readlines()
bb = [float(l.strip().split(' ')[1]) for l in ll]
print(ll[bb.index(max(bb))].strip())
