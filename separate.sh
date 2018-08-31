ab=$1
# bs_pos=$2
# bs_neg=$3
ds=$4
# lr=$5
model=$6
ni=$7
opt=$8
ptt=$9
srgt=${10}
w_pos=${11}
w_neg=${12}
n_proc=${13}
i=0
for bs_pos in $(echo "$2"); do
    for bs_neg in $(echo "$3"); do
        for lr in $(echo "$5"); do
            id=separate#ab-$ab#bs_pos-$bs_pos#bs_neg-$bs_neg#ds-$ds#lr-$lr#model-$model#opt-$opt#ptt-$ptt#srgt-$srgt#w_pos-$w_pos#w_neg-$w_neg
            python3 separate.py --ab $ab \
                                --bs_pos $bs_pos \
                                --bs_neg $bs_neg \
                                --bsi 262144 \
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
done
