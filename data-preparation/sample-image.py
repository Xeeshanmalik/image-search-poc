import ijson
import json
import sys
import argparse
from itertools import groupby
import random
from toolz.itertoolz import concat
from toolz.dicttoolz import valmap


def sample(stream, key, limit):
    items = ijson.items(sys.stdin, 'item')

    classes = groupby(key, items)
    samples = valmap(lambda xs: random.sample(xs, limit if len(xs) >= limit else len(xs)), classes)
    sample = list(concat(samples.values()))
    return sample


def keyfn(by_make, by_model, by_seller):
    return lambda x: (
            x['make'] if by_make else None,
            x['model'] if by_model else None,
            x['seller'] if by_seller else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image data sampler.')
    parser.add_argument('-n', '--limit', type=int, default=200,
                        help='the maximum number of images per group to include in the sample.')
    parser.add_argument('--by_make', default=True, action='store_true',
                        help='sample by: make')
    parser.add_argument('--by_model', default=True, action='store_true',
                        help='sample by: model')
    parser.add_argument('--by_seller', default=False, action='store_true',
                        help='sample by: seller')
    args = parser.parse_args()

    key = keyfn(args.by_make, args.by_model, args.by_seller)
    s = sample(sys.stdin, key, args.limit)
    print(json.dumps(s, indent=4, separators=(',', ': ')))
