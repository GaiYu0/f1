for x in $1/*; do
    echo "$x: $(python3 -W ignore optimal.py $x f1)" >> $2 &
done
wait
