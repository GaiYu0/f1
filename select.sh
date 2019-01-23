id=$1
metric=$2
n_cpus=$(grep -c ^processor /proc/cpuinfo)
for ds in adult cifar10 cifar100 covtype kddcup08 letter mnist; do
    for method in old new; do
        bash select-iteration.sh $id/$ds/$method $metric $n_cpus $id/$ds/$method-opt
        python3 select-hyperparameters.py $id/$ds/$method-opt
    done
done
