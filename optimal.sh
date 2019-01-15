# bash optimal.sh INPUT_PATH METRIC NUMBER_OF_PROCESSES OUTPUT_FILE
input=$1
metric=$2
n_procs=$3
output=$4
i=0
for x in $input/*; do
    echo "$x: $(python3 -W ignore optimal.py $x $metric)" >> x$output$i &
    i=$(($i + 1))
    if (($i % $n_procs == 0)); then
        wait
    fi
done
cat x$output* >> $output
rm x$output*
