import argparse
import ujson
import sys
import os
import time
import multiprocessing
import numpy as np

from toolz.functoolz import compose
from functools import partial
from multiprocessing import Pool
from PIL import Image


def image_exists(base_image_path, item):
    image_id = item['image_id']
    full_path = os.path.join(base_image_path, image_id) + '.jpg'
    exists = os.path.isfile(full_path)
    if not exists:
        print('image missing ' + full_path, file=sys.stderr)
        return False

    return True


def load_image(item):
    image_id = item['image_id']
    image_path = os.path.join(args.base_image_path, image_id) + '.jpg'

    try:
        image = Image.open(image_path)
    except OSError:
        print('image is not valid ' + image_path, file=sys.stderr)
        return {'image': "no_image", 'image_id': image_id}

    image = image.convert('RGB')
    return {'image': image, 'image_id': image_id}


def resize_images(height, width, image):

    if image['image'] != "no_image":

        # below commented line is for random background
        # bg_image = Image.fromarray(np.random.rand(width, height, 3) * 255, 'RGB')
        bg_image = Image.new('RGB', (width, height), 'white')
        fg_image = image['image'].copy()
        fg_image.thumbnail((width, height), Image.ANTIALIAS)

        # paste image in the center
        top_left = (int((width - fg_image.size[0]) / 2), int((height - fg_image.size[1]) / 2))
        bg_image.paste(fg_image, box=top_left)

        bg_image.save(os.path.join(args.output_image_path, image['image_id']) + '.jpg')

        return {'image_id': image['image_id'],
                'image_shape': np.asarray(bg_image).shape}
    else:
        return {'image_id': image['image_id'],
                'image_shape': np.empty([])}


def dir_exist(arg):

    if not os.path.exists(arg):
        os.mkdir(arg)
    else:
        return arg
    return arg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Performs downsizing of image to produce thumbnail',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_metadata_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('--output_image_path', required=True, type=lambda x: dir_exist(x),
                        help='Output path of the images')
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the images.')
    parser.add_argument('-th', '--tb_width', type=int, default='128',
                        help='Image width in pixels.')
    parser.add_argument('-tw', '--tb_height', type=int, default='128',
                        help='Image height in  pixels.')
    parser.add_argument('-p', '--number_of_processes', type=int,
                        default=multiprocessing.cpu_count())
    parser.add_argument('-c', '--chunk_size', type=int, default='1')

    args = parser.parse_args()

    items = filter(partial(image_exists, args.base_image_path), ujson.load(args.input_metadata_file))

    resized_images = compose(partial(resize_images, args.tb_height, args.tb_width), load_image)
    p = Pool(args.number_of_processes)

    start = time.clock()

    for image in p.imap(resized_images, items, args.chunk_size):
        print("Successfully Saved:", image['image_id'] + '.jpg')

    print("Total Time:", time.clock() - start)
