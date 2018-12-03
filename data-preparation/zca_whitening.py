from __future__ import print_function

import argparse
import ijson
import sys
from toolz.functoolz import compose
from functools import partial
import os
import time
import multiprocessing
from multiprocessing import Pool
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg


def load_image(item):

    image_id = item['image_id']
    path = os.path.join(args.base_image_path, image_id) + '.jpg'

    if os.path.exists(path):

        image_path = os.path.join(args.base_image_path, image_id) + '.jpg'
        image = Image.open(image_path)
        image = image.convert('RGB')

        return {'image': image, 'image_id': image_id}

    else:

        return {'image': "no_image", 'image_id': image_id}


def zca_whitening(width, height, image):

    if image['image'] != "no_image":
        regularization = 10**-5

        img = np.asarray(image['image'])

        img = img.reshape(-1, 3)

        img = img - img.mean(axis=0)

        img = img / np.sqrt((img ** 2).sum(axis=1))[:, None]

        sigma = np.dot(img.T, img) / img.shape[1]

        U, S, V = linalg.svd(sigma)

        E = np.dot(U, np.diag(1/np.sqrt(S + regularization)))

        zca_whitened = np.dot(E, U.T)

        zca_whitened = np.dot(img, zca_whitened.T)

        zca_whitened = zca_whitened.reshape(width, height, 3)

        whitened_img = Image.fromarray(np.uint8(zca_whitened * 255), 'RGB')

        whitened_img.save(os.path.join(args.output_image_path, image['image_id']) + '.jpg')

        return {'image_id': image['image_id'],
                'image_shape': np.asarray(whitened_img).shape}
    else:
        return {'image_id': image['image_id'],
            'image_shape': np.empty([])}


def pca_whitening(width, height, image):

    if image['image'] != "no_image":

        X = np.asarray(image['image'])

        X = X.reshape(-1, 3)

        X = X - X.mean(axis=0)

        X = X / np.sqrt((X ** 2).sum(axis=1))[:,None]

        pca = PCA(whiten=True)

        transformed = pca.fit_transform(X)

        pca.whiten = False

        pca = pca.inverse_transform(transformed)

        pca_whitened = pca.reshape(width, height, 3)

        whitened_img = Image.fromarray(np.uint8(pca_whitened * 255), 'RGB')

        whitened_img.save(os.path.join(args.output_image_path, image['image_id']) + '.jpg')

        return {'image_id': image['image_id'],
                'image_shape': np.asarray(whitened_img).shape}
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
        description='performs whitening of image to produce zca_whitened image',
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
    items = ijson.items(args.input_metadata_file, 'item')

    image_whitening = compose(partial(zca_whitening, args.tb_width, args.tb_height), load_image)
    p = Pool(args.number_of_processes)

    start = time.clock()

    for image in p.imap(image_whitening, items, args.chunk_size):
        print("Successfully Saved:", image['image_id'] + '.jpg')
        print("with size", image['image_shape'])

    print("Total Time:", time.clock() - start)

