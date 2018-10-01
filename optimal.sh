i=0
for x in $1/*; do
    echo "$x: $(python3 -W ignore optimal.py $x f1)" >> $2 &
    i=$(($i + 1))
    if (($i % $3 == 0)); then
        wait
    fi
done
