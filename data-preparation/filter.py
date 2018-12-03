import sys
import argparse
import simplejson as json
import ujson

from toolz.itertoolz import groupby, concat
from toolz.dicttoolz import get_in


def match(makes, models, x_image_ids, seller, probability):
    def fn(d):
        match_make = not makes or d['make'] in makes
        match_model = not models or d['model'] in models
        match_x_image_ids = not x_image_ids or d['image_id'] not in x_image_ids
        match_seller = not seller or d['seller'] == seller
        match_probability = get_in(['bounding_box', 'probability'], d, 1.0) >= probability

        return (match_make
                and match_model
                and match_x_image_ids
                and match_seller
                and match_probability)

    return fn


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Filters a dataset.')
    p.add_argument('dataset', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                   help='Dataset to filter.')
    p.add_argument('-m', '--make', type=str, nargs='+', help='Makes to keep.')
    p.add_argument('-d', '--model', type=str, nargs='+', help='Models to keep.')
    p.add_argument('--x_image_ids', type=str, nargs='+', help='These image ids will be removed.')
    p.add_argument('-s', '--seller', type=str, choices=['COMMERCIAL', 'PRIVATE'],
                   help='Keep images matching this seller.')
    p.add_argument('-p', '--probability', type=float, default=0.6,
                   help='Keep images with probability of being a vehicle >= x.')
    p.add_argument('--make_model_year_limit', type=int,
                   help=('Keep X if count(X) >= limit where X is all sets of images '
                         'grouped by make/model/year.'))

    args = p.parse_args()

    print(args, file=sys.stderr)

    makes = set(args.make) if args.make is not None else set([])
    models = set(args.model) if args.model is not None else set([])
    x_image_ids = set(args.x_image_ids) if args.x_image_ids is not None else set([])
    seller = args.seller
    probability = args.probability

    items = ujson.load(args.dataset)
    matches = filter(match(makes, models, x_image_ids, seller, probability), items)

    if args.make_model_year_limit is not None:
        groups = groupby(lambda x: (x['make'], x['model'], x['year']), matches)
        filtered = filter(lambda x: len(x) >= args.make_model_year_limit, groups.values())
        matches = concat(filtered)

    print('[')
    prev = None
    for item in matches:
        if prev is not None:
            print(',')
        print(json.dumps(item, indent=4, separators=(',', ': ')), end='')
        prev = item
    print('\n]')
