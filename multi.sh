# bash binary.sh MODEL NUMBER_OF_PROCESSES
id=$(date +%y%m%d%H%M%S)
model=$1
n_cpus=$(grep -c ^processor /proc/cpuinfo)
mkdir tb
mkdir $id
for ds in adult cifar10 cifar100 covtype kddcup08 mnist; do
    mkdir $id/$ds
    for method in old new; do
        bash $method-binary.sh $ds $1 $2
        mkdir $id/$ds/$method/
        mv tb/* $id/$ds/$method/
        bash optimal.sh $id/$ds/$method $id/$ds/$method-opt $n_cpus
    done
done
for ds in adult cifar10 cifar100 covtype kddcup08 mnist; do
    echo $ds
    ipython3 find-opt.py $id/$ds/old-opt
    ipython3 find-opt.py $id/$ds/new-opt
done
rm -r tb
