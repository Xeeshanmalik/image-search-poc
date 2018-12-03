import ujson
import sys
import argparse
import os
import multiprocessing

from functools import partial
from PIL import Image


def crop_image(image_base_path, image_output_path, item):
    image_id = item['image_id']
    bounding_box = item['bounding_box']

    print(f'Cropping image {image_id} ...')

    def to_image_path(base_path, image_id):
        return os.path.join(base_path, image_id) + '.jpg'

    image_path = to_image_path(image_base_path, image_id)
    try:
        target_size = (args.image_width, args.image_height)

        image = Image.open(image_path)
        image = image.convert('RGB')
        image.thumbnail(target_size, Image.ANTIALIAS)
        resized_image = Image.new('RGB', target_size, 'white')
        resized_image.paste(image)
        image = resized_image
    except OSError:
        return

    box = (bounding_box['left'], bounding_box['top'], bounding_box['right'], bounding_box['bottom'])
    cropped_image = image.crop(box)
    del image
    cropped_image.save(to_image_path(image_output_path, image_id), 'jpeg')
    del cropped_image

    print(f'Cropped image {image_id}')


def crop(items, image_base_path, image_output_path, number_of_processes):
    with multiprocessing.Pool(processes=number_of_processes) as pool:
        return pool.map(
                partial(crop_image, image_base_path, image_output_path),
                items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='crop image to bounding box.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The dataset file to process. Reads from stdin by default.')
    parser.add_argument('-i', '--image_base_path', type=str, default='.',
                        help='base path of the folder where images are expected to be found.')
    parser.add_argument('-o', '--image_output_path', type=str,
                        help='base path of the folder where output images will be writen to.')
    parser.add_argument('-p', '--number_of_processes', type=int, default=multiprocessing.cpu_count(),
                        help='number of processed that will be used to crop the images concurrently.')
    parser.add_argument('-wd', '--image_width', type=int, default='608',
                        help='Image width in pixels.')
    parser.add_argument('-ht', '--image_height', type=int, default='608',
                        help='Image height in  pixels.')
    args = parser.parse_args()

    items = ujson.load(args.dataset_file)
    result = crop(items, args.image_base_path, args.image_output_path, args.number_of_processes)
