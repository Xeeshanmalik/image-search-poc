#!/usr/bin/env bash

source activate image-search-poc-data-preparation

# use https://stackoverflow.com/questions/1494178/how-to-define-hash-tables-in-bash

FIRST_ID=${1}
PROBABILITY=${2:-0.8}

ID=${FIRST_ID}

# 2 classes

# 500 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar

cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford -d edge escape -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 500 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/${ID}/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford -d escape fusion -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 500 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/${ID}/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1))

# 1000 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford -d edge escape -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 1000 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/${ID}/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford -d escape fusion -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 1000 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/${ID}/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1))

# 3 classes

# 500 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford -d edge escape explorer -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 500 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/29/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet -d escape fusion slvrd1500 -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 500 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/29/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1))

# 1000 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford -d edge escape explorer -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 1000 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/${ID}/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet -d escape fusion slvrd1500 -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 1000 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/${ID}/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1))

# 5 classes

# 500 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet -d edge escape explorer equinox trax -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 500 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/${ID}/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet -d escape fusion slvrd1500 fiesta mustang -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 500 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/${ID}/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1))

# 1000 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet -d edge escape explorer equinox trax -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 1000 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/${ID}/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet -d escape fusion slvrd1500 fiesta mustang -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 1000 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/${ID}/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1))

# 10 classes

# 500 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet honda toyota -d edge escape explorer equinox trax civic accord corolla camry yaris -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 500 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/${ID}/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet honda toyota -d escape fusion slvrd1500 fiesta mustang odyssey fit rav4 venza tundra -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 500 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/${ID}/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1))

# 1000 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet honda toyota -d edge escape explorer equinox trax civic accord corolla camry yaris -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 1000 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/${ID}/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet honda toyota -d escape fusion slvrd1500 fiesta mustang odyssey fit rav4 venza tundra -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 1000 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/${ID}/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1)) 

# 2000 patterns per class
mkdir -p tmp/input/${ID}/similar
mkdir -p tmp/input/${ID}/dissimilar
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet honda toyota -d edge escape explorer equinox trax civic accord corolla camry yaris -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 2000 | tee tmp/input/${ID}/similar/dataset.json | python stats.py > tmp/input/${ID}/similar/dataset-stats.json
cat tmp/datasets/4-makes/raw-with-bb.json | python filter.py -m ford chevrolet honda toyota -d escape fusion slvrd1500 fiesta mustang odyssey fit rav4 venza tundra -p ${PROBABILITY} | python sample-ad.py --by_make --by_model -n 2000 | tee tmp/input/${ID}/dissimilar/dataset.json | python stats.py > tmp/input/${ID}/dissimilar/dataset-stats.json
cat tmp/input/${ID}/similar/dataset.json | python split.py --train_set_path tmp/input/${ID}/similar/train.json --dev_set_path tmp/input/${ID}/similar/dev.json --test_set_path tmp/input/${ID}/similar/test.json
cat tmp/input/${ID}/dissimilar/dataset.json | python split.py --train_set_path tmp/input/${ID}/dissimilar/train.json --dev_set_path tmp/input/${ID}/dissimilar/dev.json --test_set_path tmp/input/${ID}/dissimilar/test.json

cat tmp/input/${ID}/similar/train.json | python stats.py > tmp/input/${ID}/similar/train-stats.json
cat tmp/input/${ID}/similar/dev.json | python stats.py > tmp/input/${ID}/similar/dev-stats.json
cat tmp/input/${ID}/similar/test.json | python stats.py > tmp/input/${ID}/similar/test-stats.json
cat tmp/input/${ID}/dissimilar/train.json | python stats.py > tmp/input/${ID}/dissimilar/train-stats.json
cat tmp/input/${ID}/dissimilar/dev.json | python stats.py > tmp/input/${ID}/dissimilar/dev-stats.json
cat tmp/input/${ID}/dissimilar/test.json | python stats.py > tmp/input/${ID}/dissimilar/test-stats.json
ID=$((ID+1)) 

