import re
import ijson
import json
import sys
import argparse
from itertools import islice
from functools import partial

def flatten(ads):
    for ad in ads:
        baseline_ad = ad.copy()
        del baseline_ad['images']
        for i, image in enumerate(ad['images']):
            new_ad = baseline_ad.copy()
            new_ad['image'] = image
            new_ad['image_index'] = i
            yield new_ad


def next_class_id(d, key):
    if key not in d:
        d[key] = len(d)
    return d[key]

def set_pid(i, ad):
    ad['pid'] = i
    return i, ad

def set_image_id(i, ad):
    image_id = str(ad['pid']) + '_' + str(ad['id'])
    ad['image_id'] = image_id
    return i, ad

regex = re.compile(r"_\d\d\.jpg", re.IGNORECASE)
def set_image_size(image_size_id, i, ad):
    ad['image'] = regex.sub('_' +str(image_size_id)+ '.JPG', ad['image'])
    return i, ad

def set_class_id(d, key, f, i, ad):
    ad[key + '_class_id'] = next_class_id(d, f(ad, key))
    return i, ad


def rcompose(f1, f2, f3, f4, f5, f6, f7):
    return lambda a: f7(*f6(*f5(*f4(*f3(*f2(*f1(*a)))))))

def run(limit):
    ads = ijson.items(sys.stdin, 'item')

    transformers = rcompose(
            set_pid,
            set_image_id,
            partial(set_image_size, '58'),
            partial(set_class_id, {}, 'make', lambda ad, key: ad[key]),
            partial(set_class_id, {}, 'model', lambda ad, key: ad['make'] + ad[key]),
            partial(set_class_id, {}, 'color', lambda ad, key: ad[key]),
            partial(set_class_id, {}, 'body', lambda ad, key: ad[key]))

    print('[')
    prev = None
    for id, ad in islice(map(transformers, enumerate(flatten(ads))), limit):
        if prev is not None:
            print(',')
        print(json.dumps(ad), end='')
        prev = ad
    print('\n]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image data generator.')
    parser.add_argument('-n', '--limit', type=int, default=sys.maxsize,
                            help='the maximum number of images to generate.')

    args = parser.parse_args()

    run(args.limit)
