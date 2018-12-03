import re
import ijson
import json
import sys
import argparse
from itertools import islice, groupby, chain
from functools import partial, reduce
import random
import toolz
from toolz.itertoolz import mapcat, concat, take, groupby
from toolz.dicttoolz import valmap
from os.path import isfile
import os


# filters out:
# - items whose image file doesn't exist
def cleanup(items, image_base_path):
    def image_exists(image_path):
        return isfile(image_path)
    def file_size(image_path):
        return os.stat(image_path).st_size
    def image_path(item):
        image_id = item['image_id']
        return f'{image_base_path}/{image_id}.jpg'

    return filter(
        lambda image_path: image_exists(image_path) and file_size(image_path) > 1024,
        map(image_path, items))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image metadata cleaner.')
    parser.add_argument('-i', '--image_base_path', type=str, default='.',
                        help='base path of the folder where images are expected to be found.')
    args = parser.parse_args()

    items = ijson.items(sys.stdin, 'item')
    cleaned_up_items = cleanup(items, args.image_base_path)

    print('[')
    prev = None
    for item in cleaned_up_items:
        if prev is not None:
            print(',')
        print(json.dumps(item), end='')
        prev = item
    print('\n]')

