#!/usr/bin/env bash

export IMAGE_HEIGHT=240
export IMAGE_WIDTH=320
export CHANNEL=3
export T_IMAGE_HEIGHT=128
export T_IMAGE_WIDTH=128
export T_CHANNEL=3
export NUMBER_OF_CATEGORIES=2
export IMAGES_PER_CATEGORY=1000
export BATCH_SIZE=1
export EPOCHS=80
export GPU_STATE='True'
export OUTPUT_MODEL_1='/output_model_1/'
export OUTPUT_MODEL_2='/output_model_2/'
export OUTPUT_MODEL_3='/output_model_3/'
export MODEL_SAVE_1='/save_model_1/'
export MODEL_SAVE_2='/save_model_2/'
export MODEL_SAVE_3='/save_model_3/'
export DATA='trainingset_raw/cars_train'
export PRE_MODEL_PATH='./model_data/yolo.h5'
export ANCHORS_PATH='./model_data/yolo_anchors.txt'
export CLASSES_PATH='./model_data/coco_classes.txt'
export TEST_PATH='./trainingset_obj_extracted'
export SEGMENTED_OUTPUT_PATH='./trainingset_tmp/'
export SCORE_THRESHOLD=0.3
export IOU=0.5
export FONT='./font/FiraMono-Medium.otf'
export JSON_OUTPUT='bounding_boxes_training.json'
export NUMBER_OF_PREDICTIONS=100

export SUPERSET_FILTER=10
export NUMBER_OF_CLASS_PER_FILTER=2
export COUNT_PER_FILTER=1000000
export FILTER_TYPE='make'
export FILE_NAME='cars.json'
export OUTPUT_FILE_NAME='cars-filter-training.json'
export IMAGES_PATH='/Volumes/data/images/'
export TYPE='train'


python parse_json.py \
${SUPERSET_FILTER} \
${NUMBER_OF_CLASS_PER_FILTER} \
${COUNT_PER_FILTER} \
${FILTER_TYPE} \
${FILE_NAME} \
${OUTPUT_FILE_NAME} \
${TYPE}


python model_train.py \
${IMAGE_HEIGHT} \
${IMAGE_WIDTH} \
${CHANNEL} \
${T_IMAGE_HEIGHT} \
${T_IMAGE_WIDTH} \
${T_CHANNEL} \
${NUMBER_OF_CATEGORIES} \
${IMAGES_PER_CATEGORY} \
${BATCH_SIZE} \
${EPOCHS} \
${GPU_STATE} \
${OUTPUT_MODEL_1} \
${OUTPUT_MODEL_2} \
${OUTPUT_MODEL_3} \
${MODEL_SAVE_1} \
${MODEL_SAVE_2} \
${MODEL_SAVE_3} \
${DATA} \
${PRE_MODEL_PATH} \
${ANCHORS_PATH} \
${CLASSES_PATH} \
${TEST_PATH} \
${SEGMENTED_OUTPUT_PATH} \
${SCORE_THRESHOLD} \
${IOU} \
${FONT} \
${JSON_OUTPUT} \
${IMAGES_PATH} \
${OUTPUT_FILE_NAME} \
${NUMBER_OF_PREDICTIONS} \
${FILE_NAME}

