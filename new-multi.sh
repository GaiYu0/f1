# bash new-multi.sh DATASET METRIC MODEL NUMBER_OF_PROCESSES
ds=$1
metric=$2
model=$3
n_procs=$4
n_gpus=$(nvidia-smi | grep % | wc -l)
i=0
for bs in 64 128 256 512; do
    for lr in 0.1 0.01 0.001 0.0001; do
        for wd in 0.1 0.01 0.001 0.0001; do
            python3 multi.py --bsi 262144 --bst $bs --ds $ds --gpu $(($i % $n_gpus)) --id $bs-$lr-$wd --log-every 100 --lr $lr --metric $metric --model $model --ni 10000 --opt adam --ptt 6 2 2 --tb --w 0 --wd $wd &
            i=$(($i + 1))
            if (($i % $n_procs == 0)); then
                wait
            fi
        done
    done
done
wait
