import argparse
import os
import sys
import multiprocessing
import ijson
from toolz.functoolz import compose, pipe
from functools import partial
from multiprocessing import Pool
import time
import shutil


def create_sets(base_image_path, path, name, item):

    image_id = item['image_id']

    if os.path.isfile(os.path.join(base_image_path, image_id) + '.jpg'):
        new_image_name = name + '_' + image_id
        shutil.copy((os.path.join(base_image_path, image_id) + '.jpg'), os.path.join(path, name)+'/' +
                    new_image_name + '.jpg')
    else:
        print('images_' + image_id + '.jpg', "does not exist")
    return {'image_id': image_id}


def create_dir(path, class_nme):

    if not os.path.isdir(path):
        os.makedirs(os.path.join(path, class_nme))
    else:
        shutil.rmtree(path)
        os.makedirs(os.path.join(path, class_nme))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs saliency detection')
    parser.add_argument('-tt', '--meta_file', required=True, type=argparse.FileType('r'))
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the thumbnail images.')
    parser.add_argument('-p', '--number_of_processes', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-c', '--chunk_size', type=int, default=1)
    parser.add_argument('-l', '--limit', type=int, default=sys.maxsize)
    parser.add_argument('-tr', '--preparation_set_image_path', required=True, help='path to preparation '
                                                                                   'directory data generator '
                                                                                   'should follow '
                                                                                   'input/'
                                                                                   'set-type(train/test)'
                                                                                   '/class_name/'
                                                                                   'class_name_....jpg')

    class_name = 'kijiji'
    args = parser.parse_args()
    items_train = ijson.items(args.meta_file, 'item')
    create_dir(args.preparation_set_image_path, class_name)

    preparation_set = compose(partial(create_sets, args.base_image_path, args.preparation_set_image_path,
                                      class_name))

    p = Pool(args.number_of_processes)

    start = time.clock()

    for image in p.imap(preparation_set, items_train, args.chunk_size):
        print("Successfully Saved:", image['image_id'] + '.jpg')

    print("Total Time:", time.clock() - start)

