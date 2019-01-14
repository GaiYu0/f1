# bash old-binary.sh DATASET MODEL NUMBER_OF_PROCESSES
ds=$1
model=$2
n_procs=$3
n_gpus=$(lspci | grep VGA | grep NVIDIA | wc -l)
i=0
for bs in 32 64 128 256; do
    for lr in 0.1 0.01 0.001 0.0001; do
        for w in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            for wd in 0.1 0.01 0.001 0.0001; do
                python3 binary.py --bsi 262144 --bs-pos $bs --bs-neg $bs --ds $ds --gpu $(($i % $n_gpus)) --id $bs-$lr-$w-$wd --log-every 100 --lr $lr --model $model --ni 1 --opt adam --ptt 6 2 2 --tb --w $w --wd $wd &
                i=$(($i + 1))
                if (($i % $n_procs == 0)); then
                    wait
                fi
            done
        done
    done
done
wait
