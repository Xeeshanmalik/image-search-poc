from __future__ import print_function
import argparse
import ujson
from keras.preprocessing import image as Image
import sys
from toolz.functoolz import compose
from functools import partial
import os
from multiprocessing import Pool
import numpy as np
import time
from PIL import Image as Image_PIL
import copy
import json

def load_image(item):

    image_id = item['image_id']
    image = Image.load_img(
        os.path.join(args.base_image_path, image_id) + '.jpg', grayscale=False,
        target_size=(args.image_height, args.image_width, args.image_channel))
    return [item, image]


def flip_axis(x, axis):

    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def rotate_images(axis, item):

    x = flip_axis(np.asarray(item[1]), axis)
    img = Image_PIL.fromarray(x, 'RGB')
    item = item[0]
    image_name = item['image_id']+'_R'
    img.save(os.path.join(args.output_image_path, image_name) + '.jpg')
    item_rotation = copy.deepcopy(item)
    item_rotation['image_id'] = image_name

    return [item, item_rotation]


def dir_exist(arg):

    if not os.path.exists(arg):
        os.mkdir(arg)
    else:
        return arg
    return arg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs rotation of images')
    parser.add_argument('input_metadata_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('--output_image_path', required=True, type=lambda x: dir_exist(x),
                        help='Output path of the images')
    parser.add_argument('--output_metadata_file', required=True,
                        help='Output path of the file')
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the images.')
    parser.add_argument('-th', '--image_width', type=int, default='320',
                        help='Image width in pixels.')
    parser.add_argument('-tw', '--image_height', type=int, default='240',
                        help='Image height in  pixels.')
    parser.add_argument('-tc', '--image_channel', type=int, default='3',
                        help='Image height in  pixels.')
    parser.add_argument('-p', '--number_of_processes', type=int, default='2')
    parser.add_argument('-c', '--chunk_size', type=int, default='1')

    args = parser.parse_args()
    items = ujson.load(args.input_metadata_file)
    rotate_image = compose(partial(rotate_images, 1), load_image)
    p = Pool(args.number_of_processes)

    start = time.clock()
    prev = None
    with open(args.output_metadata_file, 'w') as f:
        f.write('[')
        f.write('\n')
        for index, image in enumerate(p.imap(rotate_image, items, args.chunk_size)):
            print("Successfully Saved Rotation For", image[1]['image_id'])
            if prev is not None:
                f.write(',')
            f.write(json.dumps(image[0], indent=4, separators=(',', ': ')))
            f.write(',')
            f.write('\n')
            f.write(json.dumps(image[1], indent=4, separators=(',', ': ')))
            f.write('\n')
            prev = image
        f.write(']')
    f.close()
    print("Total Time:", time.clock() - start)


