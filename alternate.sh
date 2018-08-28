gpu=0
for bs in 64 128 256 512; do
    for lr in 0.1 0.01 0.001 0.0001; do
        for srgt in max exp; do
            id=alternate#ab-$2#bs-$bs#ds-$ds#lr-$lr#model-$model#srgt-$srgt
            python3 separate.py --ab $2 \
                                --bs $bs \
                                --bsi 262144 \
                                --ds $1 \
                                --gpu $gpu \
                                --id $id \
                                --log-every 100 \
                                --lr $lr \
                                --model mlp \
                                --ni 1000 \
                                --ptt 6 2 2 \
                                --srgt $srgt \
                                --update-every 1 \
                                --w-pos "p1 + fp" \
                                --w-neg "p1 - fn" | tee log/$id &
            gpu=$((gpu + 1))
            if ! (($gpu % 4)); then
                gpu=0
                wait
            fi
        done
    done
done
