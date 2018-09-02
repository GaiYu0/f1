# bs=$1
ds=$2
# lr=$3
model=$4
ni=$5
opt=$6
ptt=$7
srgt=$8
n_proc=$9
i=0

n=10
for bs in $(echo "$1"); do
    for lr in $(echo "$3"); do
        for t in $(seq 1 $((n - 1))); do
            w_pos=$(python3 -c "print(round(2 - 2 * $t / $n, 1))")
            w_neg=$(python3 -c "print(round(2 * $t / $n, 1))")

            id="cs#bs_pos-$bs"`
                `"#bs_neg-$bs"`
                `"#ds-$ds"`
                `"#lr-$lr"`
                `"#model-$model"`
                `"#opt-$opt"`
                `"#ptt-$ptt"`
                `"#srgt-$srgt"`
                `"#w_pos-$w_pos"`
                `"#w_neg-$w_neg"

            python3 separate.py --bsi 262144 \
                                --bs_pos $bs \
                                --bs_neg $bs \
                                --ds $ds \
                                --gpu $(($i % 4)) \
                                --id "$id" \
                                --log-every 100 \
                                --lr $lr \
                                --model $model \
                                --ni $ni \
                                --opt $opt \
                                --ptt $ptt \
                                --srgt $srgt \
                                --tb \
                                --update-every 0 \
                                --w-pos "$w_pos" \
                                --w-neg "$w_neg" &

            i=$((i + 1))
            if ! (($i % $n_proc)); then
                wait
            fi
        done
    done
done
