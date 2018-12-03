import sys
import argparse
import ujson
import json
from functools import partial, reduce
from toolz.dicttoolz import valmap
from toolz.itertoolz import first
import operator as op
from collections import OrderedDict


def update_in(d, keys, fn, default):
    key = first(keys)
    if len(keys) == 1:
        if key not in d:
            d[key] = default
        d[key] = fn(d[key])
    else:
        if key not in d:
            d[key] = dict()
        nested = d[key]
        update_in(nested, keys[1:], fn, default)

    return d


def run(dataset):

    def ordered_dict_sorted_by_key(d):
        return OrderedDict(sorted(d.items(), key=lambda t: t[0], reverse=True))

    def ordered_dict_sorted_by_value(d):
        return OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))

    def accum(stats, x):
        make = x['make']
        model = x['model']
        year = x['year']
        body = x['body']
        color = x['color']
        ad_id = x['id']

        stats = update_in(stats, ['total'], partial(op.add, 1), 0)
        stats = update_in(stats, ['make', make], partial(op.add, 1), 0)
        stats = update_in(stats, ['model', make, model], partial(op.add, 1), 0)
        stats = update_in(stats, ['year', make, model, year], partial(op.add, 1), 0)
        stats = update_in(stats, ['color', make, model, color], partial(op.add, 1), 0)
        stats = update_in(stats, ['body', body], partial(op.add, 1), 0)
        stats = update_in(stats, ['ad_id', ad_id], partial(op.add, 1), 0)

        return stats

    stats = reduce(accum, dataset, {})

    stats['make'] = ordered_dict_sorted_by_value(stats['make'])
    stats['model'] = valmap(ordered_dict_sorted_by_value, stats['model'])
    stats['body'] = ordered_dict_sorted_by_value(stats['body'])

    for make, models in stats['year'].items():
        for model, years in models.items():
            stats['year'][make][model] = ordered_dict_sorted_by_value(years)

    for make, models in stats['color'].items():
        for model, colors in models.items():
            stats['color'][make][model] = ordered_dict_sorted_by_value(colors)

    stats['ad_total'] = len(stats['ad_id'])
    stats['ad_avg_patterns'] = reduce(op.add, stats['ad_id'].values(), 0) / stats['ad_total']
    del stats['ad_id']

    print(json.dumps(stats, indent=4, separators=(',', ': ')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='computes the stats of a dataset')
    parser.add_argument('dataset_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The dataset file to process. Reads from stdin by default.')

    args = parser.parse_args()

    dataset = ujson.load(args.dataset_file)
    run(dataset)
