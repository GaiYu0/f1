ab=$1
# bs=$2
ds=$3
# lr=$4
model=$5
ni=$6
opt=$7
ptt=$8
srgt=$9
w_pos=${10}
w_neg=${11}
n_proc=${12}
i=0

for bs in $(echo "$2"); do
    for lr in $(echo "$4"); do
        id="alternate#ab-$ab"`
                   `"#bs_pos-$bs"`
                   `"#bs_neg-$bs"`
                   `"#ds-$ds"`
                   `"#lr-$lr"`
                   `"#model-$model"`
                   `"#opt-$opt"`
                   `"#ptt-$ptt"`
                   `"#srgt-$srgt"`
                   `"#w_pos-$w_pos"`
                   `"#w_neg-$w_neg"

        python3 separate.py --ab $ab \
                            --bsi 262144 \
                            --bsl 1024 \
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
                            --update-every 1 \
                            --w-pos "$w_pos" \
                            --w-neg "$w_neg" &

        i=$((i + 1))
        if ! (($i % $n_proc)); then
            wait
        fi
    done
done
