gpu=0
for bs in 128; do
    for lr in 0.01; do
        for srgt in max exp; do
            python3 train.py --bs $bs --bsi 65536 --ds covtype --gpu $gpu --log-every 100 --lr $lr --model mlp --ni 1000 --ptt 6 2 2 --srgt $srgt --update-every 1 --w-pos fn --w-neg fp &
            gpu=$((gpu + 1))
            if ! (($gpu % 4)); then
                gpu=0
                wait
            fi
        done
    done
done
