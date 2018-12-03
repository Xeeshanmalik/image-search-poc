import ujson
import sys
import argparse
import random

from toolz.itertoolz import mapcat, groupby, second, concat, take
sys.path.append("..")
from common.pretty_json import pretty_json


def sample(dataset, key, limit):
    by_id = groupby(
        lambda x: (x['id'], x['make'], x['model'], x['seller'], x['color'], x['year']),
        dataset)
    classes = groupby(key, by_id.items())

    def sample_images(xs):
        ads = random.sample(xs, limit if len(xs) >= limit else len(xs))
        images = mapcat(second, ads)
        return list(take(limit, images))

    samples = map(sample_images, classes.values())

    sample = concat(samples)

    return sample


def keyfn(by_make, by_model, by_seller, by_color, by_year):
    return lambda x: (
            x[0][1] if by_make else None,
            x[0][2] if by_model else None,
            x[0][3] if by_seller else None,
            x[0][4] if by_color else None,
            x[0][5] if by_year else None)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='image data sampler.')
    p.add_argument('dataset', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                   help='Dataset to sample from. It defaults to stdin.')
    p.add_argument('-n', '--limit', type=int, default=200,
                   help='the maximum number of images per group to include in the sample.')
    p.add_argument('--by_make', default=True, action='store_true', help='sample by: make')
    p.add_argument('--by_model', default=True, action='store_true', help='sample by: model')
    p.add_argument('--by_seller', default=False, action='store_true', help='sample by: seller')
    p.add_argument('--by_color', default=False, action='store_true', help='sample by: color')
    p.add_argument('--by_year', default=False, action='store_true', help='sample by: year')
    args = p.parse_args()

    print('loading dataset...', file=sys.stderr)
    dataset = ujson.load(args.dataset)
    print('loaded dataset', file=sys.stderr)
    key = keyfn(args.by_make, args.by_model, args.by_seller, args.by_color, args.by_year)

    dataset_sample = sample(dataset, key, args.limit)

    print('[')
    prev = None
    for item in dataset_sample:
        if prev is not None:
            print(',')
        print(pretty_json(item), end='')
        prev = item
    print('\n]')
