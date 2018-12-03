#!/bin/bash

set -e

BASE_PATH=${1}
ID=${2}
COPY_TO_BUCKET=${3}

function help {
    echo "syntax: ./run-end-to-end.sh BASE_PATH ID"
}

if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH missing!"
    help
    exit 1
fi

if [ -z "$ID" ]; then
    echo "ID missing!"
    help
    exit 1
fi

export HEIGHT=${HEIGHT:-197}
export WIDTH=${WIDTH:-197}
export BATCH_SIZE=${BATCH_SIZE:-1}
export GPU_COUNT=${GPU_COUNT:-0}
export LABEL=${LABEL:-'body'}

DATASET_PATH=$BASE_PATH/$ID
echo ${DATASET_PATH}

export ROOTDIR=$(pwd)
#export BASE_IMAGE_PATH=/mnt/dev-visual-search/images/segmented/v3
export BASE_IMAGE_PATH=/Users/zeeshanmalik/dev/image-search-poc/cars/cars_train
export THUMBNAIL_OUTPUT=${ROOTDIR}/tmp/thumbnail_output
export OUTPUT_BASE_PATH=${ROOTDIR}/tmp/output
export TRAIN_PREPARATION_PATH=${ROOTDIR}/tmp/input_train
export DEV_PREPARATION_PATH=${ROOTDIR}/tmp/input_dev
export MODEL_OUTPUT=${ROOTDIR}/tmp/model
export EPOCHS=${EPOCHS:-20}

mkdir -p tmp
mkdir -p ${MODEL_OUTPUT}
mkdir -p ${OUTPUT_BASE_PATH}
mkdir -p ${THUMBNAIL_OUTPUT}
mkdir -p ${TRAIN_PREPARATION_PATH}
mkdir -p ${DEV_PREPARATION_PATH}
rm -f ${OUTPUT_BASE_PATH}/*
rm -f ${MODEL_OUTPUT}/*
rm -rf ${TRAIN_PREPARATION_PATH}/*
rm -rf ${DEV_PREPARATION_PATH}/*

# conversion to thumbnails

cd data-preparation

conda env update

source activate image-search-poc-data-preparation

python thumbnail.py \
        -th ${HEIGHT} \
        -tw ${WIDTH} \
        --base_image_path ${BASE_IMAGE_PATH} \
        --output_image_path ${THUMBNAIL_OUTPUT} \
        < ${DATASET_PATH}/dataset.json

python one-hot-coding.py \
        --label ${LABEL} \
        --input_stat_file ${DATASET_PATH}/train-stats.json \
        < ${DATASET_PATH}/train.json \
        > ${DATASET_PATH}/train-coded.json

python one-hot-coding.py \
        --label ${LABEL} \
        --input_stat_file ${DATASET_PATH}/dev-stats.json \
        < ${DATASET_PATH}/dev.json \
        > ${DATASET_PATH}/dev-coded.json

source deactivate

cd ..

#
# training
#

cd training

conda env update

source activate image-search-poc-training
echo ${BASE_IMAGE_PATH}
echo ${DATASET_PATH}

python preparation.py \
        --base_image_path ${THUMBNAIL_OUTPUT} \
        --input_stat_file ${DATASET_PATH}/train-stats.json \
        --meta_file ${DATASET_PATH}/train-coded.json \
        --preparation_set_image_path ${TRAIN_PREPARATION_PATH} \
        --label ${LABEL}

python preparation.py \
        --base_image_path ${THUMBNAIL_OUTPUT} \
        --input_stat_file ${DATASET_PATH}/dev-stats.json \
        --meta_file ${DATASET_PATH}/dev-coded.json \
        --preparation_set_image_path ${DEV_PREPARATION_PATH} \
        --label ${LABEL}

python train.py \
        --model_save_path ${MODEL_OUTPUT}/ \
        -gc ${GPU_COUNT} -ep ${EPOCHS} -bb ${BATCH_SIZE} -ih ${HEIGHT} -iw ${WIDTH} -ic 3 --sh True \
        --train_images_path ${TRAIN_PREPARATION_PATH} \
        --dev_images_path ${DEV_PREPARATION_PATH} \
        --train_meta_file ${DATASET_PATH}/train-coded.json \
        --dev_meta_file ${DATASET_PATH}/dev-coded.json \
        --train_stat_file ${DATASET_PATH}/train-stats.json \
        --dev_stat_file ${DATASET_PATH}/dev-stats.json \
        --label ${LABEL}

cp -a ${MODEL_OUTPUT}/*'ep='${EPOCHS}*.h5 ${OUTPUT_BASE_PATH}

source deactivate

cd ..

#
# feature map
#

cd evaluation

conda env update

source activate image-search-poc-evaluation
MODEL=$(find ${OUTPUT_BASE_PATH}/ -name "*.h5" -print)
echo *********************************
echo $MODEL
echo *********************************
python feature_map.py \
        --base_image_path ${THUMBNAIL_OUTPUT} \
        --model_name thumbnail \
        --model_path ${MODEL} \
        -ih ${HEIGHT} -iw ${WIDTH} -ic 3 -ll stack_1_encoded \
        <  ${DATASET_PATH}/train.json \
        > ${OUTPUT_BASE_PATH}/feature-map-train.json

python feature_map.py \
        --base_image_path ${THUMBNAIL_OUTPUT} \
        --model_name thumbnail \
        --model_path ${MODEL} \
        -ih ${HEIGHT} -iw ${WIDTH} -ic 3 -ll stack_1_encoded \
        <  ${DATASET_PATH}/dev.json \
        > ${OUTPUT_BASE_PATH}/feature-map-dev.json

#
# prediction
#

#python predict.py \
#        --distance hamming \
#        -tfm ${OUTPUT_BASE_PATH}/feature-map-train.json \
#        -n 20 -p 32 \
#        < ${OUTPUT_BASE_PATH}/feature-map-dev.json \
#        > ${OUTPUT_BASE_PATH}/prediction-hamming-dev.json

python predict.py \
        --distance euclidean \
        -tfm ${OUTPUT_BASE_PATH}/feature-map-train.json \
        -n 20 -p 4 \
        < ${OUTPUT_BASE_PATH}/feature-map-dev.json \
        > ${OUTPUT_BASE_PATH}/prediction-euclidean-dev.json

#
# evaluation
#

#python evaluate.py \
#        < ${OUTPUT_BASE_PATH}/prediction-hamming-dev.json \
#        > ${OUTPUT_BASE_PATH}/evaluation-hamming-dev.json

python evaluate.py \
        -s as_is -n 1 3 5 10 \
        < ${OUTPUT_BASE_PATH}/prediction-euclidean-dev.json \
        > ${OUTPUT_BASE_PATH}/evaluation-euclidean-dev.json

for feature in make model year body color; do
        python evaluate.py \
                -s as_is -n 1 3 5 10 \
                --by_feature ${feature} \
                < ${OUTPUT_BASE_PATH}/prediction-euclidean-dev.json \
                > ${OUTPUT_BASE_PATH}/evaluation-euclidean-dev-by-${feature}.json
done

python evaluate.py \
        -s most_frequent -n 1 2 3 -l 10 \
        < ${OUTPUT_BASE_PATH}/prediction-euclidean-dev.json \
        > ${OUTPUT_BASE_PATH}/evaluation-euclidean-dev-by-frequency.json

for feature in make model year body color; do
        python evaluate.py \
                -s most_frequent -n 1 2 3 -l 10 \
                --by_feature ${feature} \
                < ${OUTPUT_BASE_PATH}/prediction-euclidean-dev.json \
                > ${OUTPUT_BASE_PATH}/evaluation-euclidean-dev-by-frequency-by-${feature}.json
done

#
# prediction report
#

echo "Computing trainset by class report"
python report_trainset.py -o ${OUTPUT_BASE_PATH} < ${DATASET_PATH}/train.json

#echo "Computing prediction report (hamming)..."
#python report_prediction.py \
#        < ${OUTPUT_BASE_PATH}/prediction-hamming-dev.json \
#        > ${OUTPUT_BASE_PATH}/prediction-hamming-dev.html

echo "Computing prediction report (euclidean)..."
python report_prediction.py \
        --evaluation_id ${ID} \
        --base_output_path ${OUTPUT_BASE_PATH} \
        --model_evaluation_file ${OUTPUT_BASE_PATH}/evaluation-euclidean-dev-by-model.json \
        < ${OUTPUT_BASE_PATH}/prediction-euclidean-dev.json \
        > ${OUTPUT_BASE_PATH}/prediction-euclidean-dev.html

#
# viz points
#

# binarized

python visualize_clustering.py \
        --clazz make  -i ${OUTPUT_BASE_PATH} < ${OUTPUT_BASE_PATH}/feature-map-dev.json

python visualize_clustering.py \
        --clazz model  -i ${OUTPUT_BASE_PATH} < ${OUTPUT_BASE_PATH}/feature-map-dev.json

python visualize_clustering.py \
        --clazz year  -i ${OUTPUT_BASE_PATH}  < ${OUTPUT_BASE_PATH}/feature-map-dev.json

python visualize_clustering.py \
        --clazz body  -i ${OUTPUT_BASE_PATH}  < ${OUTPUT_BASE_PATH}/feature-map-dev.json

python visualize_clustering.py \
        --clazz color  -i ${OUTPUT_BASE_PATH} < ${OUTPUT_BASE_PATH}/feature-map-dev.json

# activations

python visualize_clustering.py \
        --viz_activations --clazz make -i ${OUTPUT_BASE_PATH} \
        < ${OUTPUT_BASE_PATH}/feature-map-dev.json

python visualize_clustering.py \
        --viz_activations --clazz model -i ${OUTPUT_BASE_PATH} \
        < ${OUTPUT_BASE_PATH}/feature-map-dev.json

python visualize_clustering.py \
        --viz_activations --clazz year -i ${OUTPUT_BASE_PATH}  \
        < ${OUTPUT_BASE_PATH}/feature-map-dev.json

python visualize_clustering.py \
        --viz_activations --clazz body -i ${OUTPUT_BASE_PATH}  \
        < ${OUTPUT_BASE_PATH}/feature-map-dev.json

python visualize_clustering.py \
        --viz_activations --clazz color -i ${OUTPUT_BASE_PATH} \
        < ${OUTPUT_BASE_PATH}/feature-map-dev.json

#
# viz images
#

python visualize_clustering.py \
        -i ${OUTPUT_BASE_PATH} \
        --image_base_path ${BASE_IMAGE_PATH} -pt images \
        < ${OUTPUT_BASE_PATH}/feature-map-dev.json

#
# copy output to GCP
#

if [ "$COPY_TO_BUCKET" = "--copy_to_bucket" ]; then
    echo "Copying to bucket ..."
    rm ${OUTPUT_BASE_PATH}/feature-map-train.json
    gsutil -m cp -r -n -Z ${OUTPUT_BASE_PATH}/* gs://dev_visual_search/evaluations/output/by-id/$ID/
fi
