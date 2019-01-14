mkdir adult
cd adult
bash ../data/adult.sh
python3 ../data/adult.py adult.data adult.test
cd ..

mkdir covtype
cd covtype
bash ../data/covtype.sh
python3 ../data/covtype.py covtype.data
cd ..

mkdir kddcup08
cd kddcup08
bash ../data/kddcup08.sh
python3 ../data/kddcup08.py Features.txt Info.txt
cd ..

mkdir letter
cd letter
bash ../data/letter.sh
python3 ../data/letter.py letter-recognition.data
cd ..

python3 data/cifar10.py
python3 data/cifar100.py
python3 data/mnist.py
