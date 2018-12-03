import os
import sys
import coco
import model as modelib
import ujson
import argparse
import numpy as np
from toolz.itertoolz import topk, get
import cv2
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import shutil


class BatchProcessing:

    def __init__(self, input_img_gen):

        self.generator = input_img_gen

    def image_batch_generator(self, base_image_path, batch_size):

        """
        The target size needs to be fixed afterwards (https://github.com/qdbp/keras/
        blob/9c9923ff46d667e24999ffe1e280b0488f7d4658/keras/preprocessing/image.py)
        after the changes in the link will be merged with the master if :-).

        """

        image_set = self.generator.flow_from_directory(base_image_path, batch_size=batch_size,
                                                       target_size=(1024, 1024), class_mode=None)

        return image_set


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


def segment_batch_of_images(model, list_of_images, list_of_filenames):

    results = model.detect(list(list_of_images), verbose=1)

    def area(box):
        top, left, bottom, right = box
        return abs(bottom - top) * abs(right - left)

    def max_bound(item):
        if set(max_bounded_box) == set(item):
            return 1
        else:
            return 0

    for counter, output in enumerate(results):

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
            mask_rgb = cv2.cvtColor(mask_img, cv2. COLOR_GRAY2RGB)
            img = list_of_images[counter]
            new_Image = np.asarray(img) * (mask_rgb != 0)

            # Try and run with even black background by commenting the below one line

            new_Image= np.where(new_Image == 0, 255, new_Image)
            new_Image = Image.fromarray(new_Image.astype('uint8'), 'RGB')

            if not list_of_filenames:

                new_Image.save(os.path.join(args.output_image_path, "filename_zero" + str(counter) + '.jpg'))

            else:

                new_Image.save(os.path.join(args.output_image_path, list_of_filenames[counter]))


def create_dir(path, class_nme):

    if not os.path.isdir(path):
        os.makedirs(os.path.join(path, class_nme))
    else:
        shutil.rmtree(path)
        os.makedirs(os.path.join(path, class_nme))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform object segmentation using deep mask_rcnn')
    parser.add_argument('--output_image_path', required=False, type=lambda x: dir_exist(x),
                        help='Output path of the images')
    parser.add_argument('--gpu_count', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rcnn_model_path', type=str, default='mask_rcnn_coco.h5',
                        help='Path to the YOLO model.')
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the images.')
    parser.add_argument('--rcnn_classes', type=str, default='coco_classes.txt',
                        help='Path to the classes of YOLO model.')

    args = parser.parse_args()

    class InferenceConfig(coco.CocoConfig):

        GPU_COUNT=args.gpu_count
        IMAGES_PER_GPU=args.batch_size
        BATCH_SIZE=args.batch_size

    class_name = "kijiji"

    create_dir(args.output_image_path, class_name)

    model_dir = os.getcwd()

    class_names = read_classes(args.rcnn_classes)

    config = InferenceConfig()

    config.BATCH_SIZE = args.batch_size
    config.IMAGE_PER_GPU=args.gpu_count

    config.display()

    model = modelib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)

    model.load_weights(args.rcnn_model_path, by_name=True)

    image_data_gen = ImageDataGenerator(horizontal_flip=False)

    images = BatchProcessing(image_data_gen)

    data_set_len = len([name for name in os.listdir(os.path.join(args.base_image_path, class_name))
                         if os.path.isfile(os.path.join(os.path.join(args.base_image_path, class_name),
                                                        name))])

    generator = images.image_batch_generator(args.base_image_path, args.batch_size)

    batches = 0

    for batch in generator:

        idx = (generator.batch_index - 1) * generator.batch_size
        file_names = generator.filenames[idx:idx+generator.batch_size]
        print(file_names)
        print(config.BATCH_SIZE)
        segment_batch_of_images(model, batch, file_names)

        batches += 1

        if batches >= data_set_len / args.batch_size:

            break





















