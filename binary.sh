# bash binary.sh METRIC MODEL NUMBER_OF_PROCESSES
metric=$1
model=$2
n_procs=$3
id=$metric-$model-$(date +%y%m%d%H%M%S)
n_cpus=$(grep -c ^processor /proc/cpuinfo)
mkdir tb
mkdir $id
for ds in adult cifar10 cifar100 covtype kddcup08 letter mnist; do
    mkdir $id/$ds
    for method in old new; do
        bash $method-binary.sh $ds $metric $model $n_procs
        mkdir $id/$ds/$method/
        mv tb/* $id/$ds/$method/
    done
done
rm -r tb
