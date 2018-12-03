import os
import sys
import skimage.io
import coco
import model as modelib
import ujson
import argparse
from toolz.functoolz import compose
from functools import partial
import numpy as np
from toolz.itertoolz import topk
import cv2
from PIL import Image
import time


# For Python, run "make" under object_segmentation/coco/PythonAPI
# For Python, run "make install" under object_segmentation/coco/PythonAPI

def load_image(item):
    try:
        image_id = item['image_id']
        full_path = os.path.join(args.base_image_path, image_id) + '.jpg'
        image = skimage.io.imread(full_path)
        return {'image': image, 'image_id': image_id}
    except IOError:
        print('image is not valid ' + full_path, file=sys.stderr)
        return None


def dir_exist(arg):
    if not os.path.exists(arg):
        os.mkdir(arg)
    else:
        return arg
    return arg


def read_classes(path):
    with open(path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def segment_images(model, image):
    new_image_path = os.path.join(args.output_image_path, image['image_id']) + '.jpg'
    if os.path.isfile(new_image_path):
        print(f'Skipping previously segmened image {image["image_id"]}.')
        return image['image_id']

    results = model.detect([image['image']], verbose=1)
    output = results[0]

    def area(box):
        top, left, bottom, right = box
        return abs(bottom - top) * abs(right - left)

    def max_bound(item):
        if set(max_bounded_box) == set(item):
            return 1
        else:
            return 0
    x = topk(1, output['rois'], key=area)
    if not x:
        print("No bounding Box detected")
    else:
        max_bounded_box = list(x[0])
        max_mask_id = 0

        for index, item in enumerate(map(max_bound, output['rois'])):
            if item == 1:
                max_mask_id = index
                break
        mask_img = output['masks'][:, :, max_mask_id]
        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
        new_image = image['image'] * (mask_rgb != 0)
        new_image = np.where(mask_rgb == 0, 255, new_image)
        new_image = Image.fromarray(new_image, 'RGB')
        new_image.save(new_image_path)

    return image['image_id']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform object segmentation using deep mask_rcnn')
    parser.add_argument('input_metadata_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('output_metadata_file', nargs='?', type=argparse.FileType('w'),
                          default=sys.stdout,
                        help='The updated output file. Writes to stdout by default.')
    parser.add_argument('--output_image_path', required=False, type=lambda x: dir_exist(x),
                        help='Output path of the images')
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--images_per_gpu', type=int, default=1)
    parser.add_argument('--rcnn_model_path', type=str, default='mask_rcnn_coco.h5',
                        help='Path to the YOLO model.')
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the images.')
    parser.add_argument('--rcnn_classes', type=str, default='coco_classes.txt',
                        help='Path to the classes of YOLO model.')

    args = parser.parse_args()

    class InferenceConfig(coco.CocoConfig):

        GPU_COUNT = args.gpu_count
        IMAGES_PER_GPU = args.images_per_gpu
        BATCH_SIZE = args.batch_size

    items = ujson.load(args.input_metadata_file)

    model_dir = os.getcwd()

    class_names = read_classes(args.rcnn_classes)

    config = InferenceConfig()

    config.BATCH_SIZE = 1
    config.IMAGE_PER_GPU = 0

    config.display()

    model = modelib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)

    model.load_weights(args.rcnn_model_path, by_name=True)

    segment_image = compose(
        partial(segment_images, model),
        load_image)

    for image in map(segment_image, items):
        print("Object Extracted Successfully", image + '.jpg')
