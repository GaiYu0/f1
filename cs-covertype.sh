gpu=0
n=20
for bs in 128; do
    for lr in 0.01; do
        for srgt in max exp; do
            for t in $(seq 1 $((n - 1))); do
                w_pos=$(python3 -c "print(round(2 - 2 * $t / $n, 1))")
                w_neg=$(python3 -c "print(round(2 * $t / $n, 1))")
                python3 train.py --bs $bs --bsi 262144 --ds covertype --gpu $gpu --log-every 100 --lr $lr --model mlp --ni 10000 --ptt 6 2 2 --srgt $srgt --tb --update-every 0 --w-pos $w_pos --w-neg $w_neg &
                gpu=$((gpu + 1))
                if ! (($gpu % 4)); then
                    gpu=0
                    wait
                fi
            done
        done
    done
done
