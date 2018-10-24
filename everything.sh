mkdir result
for ds in adult cifar10 cifar100 covtype kddcup08; do
    bash old-binary.sh $ds 8
    mkdir result/$ds
    mkdir result/$ds/old/
    mv tb/* result/$ds/old/
    bash optimal.sh result/$ds/old result/$ds/old-opt 64
    bash new-binary.sh $ds 8
    mkdir result/$ds/new/
    mv tb/* result/$ds/new/
    bash optimal.sh result/$ds/new result/$ds/new-opt 64
done
for ds in adult cifar10 cifar100 covtype kddcup08; do
    echo $ds
    ipython3 sel-opt.py result/$ds/old-opt
    ipython3 sel-opt.py result/$ds/new-opt
done
