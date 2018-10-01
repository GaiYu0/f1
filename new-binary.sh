# bash new-binary.sh DATASET NUMBER_OF_PROCESSES
i=0
for bs in 64 128 256 512; do
    for lr in 0.1 0.01 0.001 0.0001; do
        for wd in 0.1 0.01 0.001 0.0001; do
            python3 binary.py --bsi 262144 --bs-pos $bs --bs-neg $bs --ds $1 --gpu $(($i % 4)) --id $bs-$lr-$wd --log-every 100 --lr $lr --model mlp --ni 10000 --opt adam --ptt 6 2 2 --tb --w 0 --wd $wd &
            i=$(($i + 1))
            if (($i % $2 == 0)); then
                wait
            fi
        done
    done
done
wait
