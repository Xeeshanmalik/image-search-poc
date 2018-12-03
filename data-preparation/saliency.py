from __future__ import print_function
import argparse
import ijson
import sys
from toolz.functoolz import compose, pipe
from itertools import imap
import os
import pyimgsaliency as psal
from scipy.misc import imsave
import numpy
from multiprocessing import Pool
import time


def saliency_detection(item):

    try:
        mbd = psal.get_saliency_mbd([args.base_image_path + '/' + item['image_id']+'.jpg'])
        imsave(os.path.join(args.output_image_path, item['image_id']+'.jpg'), mbd)
    except numpy.linalg.linalg.LinAlgError:
        print('Unable to compute saliency for: ' + item['image_id'])

    return {'image_id': item['image_id']}


def dir_exist(arg):

    if not os.path.exists(arg):
        os.mkdir(arg)
    else:
        return arg
    return arg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs saliency detection')
    parser.add_argument('input_metadata_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
            help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('--output_image_path', required=True, type=lambda x: dir_exist(x),
                        help='Output path of the saliency images')
    parser.add_argument('--base_image_path', type=str, required=True,
            help='Path of the folder containing the thumbnail images.')
    parser.add_argument('-p', '--number_of_processes', type=int, default='2')
    parser.add_argument('-c', '--chunk_size', type=int, default='1')

    args = parser.parse_args()

    items = ijson.items(args.input_metadata_file, 'item')

    saliency_images = compose(saliency_detection)

    p = Pool(args.number_of_processes)

    start = time.clock()

    for image in p.imap_unordered(saliency_images, items, args.chunk_size):

        print("Successfully Saved:", image['image_id']+'.jpg')

    print("Total Time:", time.clock() - start)
