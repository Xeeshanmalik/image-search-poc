import sys
import argparse
import ujson
import simplejson as json
import operator as op

from functools import reduce, partial
from toolz.itertoolz import take, groupby, first
from toolz.functoolz import juxt, pipe, compose
from toolz.dicttoolz import valmap
from collections import OrderedDict


def update_in(d, keys, fn, default=None):
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


def _ordered_dict_sorted_by_value(d):
    return OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))


def _log(x, label=''):
    print(f'{label}{x}', file=sys.stderr)
    return x


def predictions_as_is(label, entry):
    return entry['predictions']


def predictions_most_frequent(label, entry):
    data_keys = label['data_keys']
    data_fn = juxt(map(op.itemgetter, data_keys))

    histogram = pipe(entry['predictions'],
                     partial(groupby, data_fn),
                     partial(valmap, len),
                     _ordered_dict_sorted_by_value)

    return map(lambda values: {k: v for k, v in zip(data_keys, values)}, histogram.keys())


def evaluate_at_limit(meta, limit, dataset):

    def accuracy(predictions_fn, label, match_one):

        def evaluate_elem(elem):

            def match(p):
                data_fn = juxt(map(op.itemgetter, label['data_keys']))
                # try deleting tuple func call
                return data_fn(elem) == data_fn(p)

            predictions = list(take(limit, predictions_fn(label, elem)))
            if match_one is True:
                total = min(1, len(predictions))
                correct = min(1, len(list(filter(match, predictions))))
            else:
                total = len(predictions)
                correct = len(list(filter(match, predictions)))

            return total, correct

        def accumulate(a, total_correct):
            total, correct = total_correct
            update_in(a, ['total'], partial(op.add, total), 0)
            update_in(a, ['correct'], partial(op.add, correct), 0)
            return a

        r = reduce(accumulate, map(evaluate_elem, dataset), {})

        return r['correct'] / r['total']

    p_fn = meta['predictions_fn']

    return {
        'accuracy': {
            label['key']: accuracy(p_fn, label, meta['match_one']) for label in meta['labels']
        }
    }


def evaluate(meta, limits, dataset):
    return {
        l: evaluate_at_limit(meta, l, dataset) for l in limits
    }


# !!! modeling labels as a dict would have made this code a no brainer
def get_feature_data_keys(strategy_meta, feature):
    return pipe(strategy_meta['labels'],
                partial(filter, lambda x: x['key'] == feature),
                first,
                op.itemgetter('data_keys'))


if __name__ == '__main__':

    labels = [{'key': 'make', 'data_keys': ['make']},
              {'key': 'model', 'data_keys': ['make', 'model']},
              {'key': 'year', 'data_keys': ['make', 'model', 'year']},
              {'key': 'body', 'data_keys': ['body']},
              {'key': 'color', 'data_keys': ['color']}]

    meta = {'as_is': {'predictions_fn': predictions_as_is,
                      'match_one': False,
                      'labels': labels},
            'most_frequent': {'predictions_fn': predictions_most_frequent,
                              'labels': labels,
                              'match_one': True}}

    parser = argparse.ArgumentParser(description='Evaluates predictions.')
    parser.add_argument('input_predictions_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='Predictions to evaluate')
    parser.add_argument('output_evaluation_file', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help=("The evaluation output will be writen to this file. "
                              "Writes to stdout by default."))
    parser.add_argument('-n', '--predictions_limit', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='Limits at which to evaluate predictions.')
    parser.add_argument('-s', '--strategy', type=str,
                        choices=['as_is', 'most_frequent'], default='as_is')
    parser.add_argument('-l', '--limit', type=int, default=10,
                        help='Only limit predictions per image will be considered.')
    parser.add_argument('--by_feature', type=str,
                        choices=['make', 'model', 'year', 'color', 'body'],
                        help='Only limit predictions per image will be considered.')
    args = parser.parse_args()

    print(f'loading predictions...', file=sys.stderr)
    dataset = ujson.load(args.input_predictions_file)
    print(f'loaded {len(dataset)} predictions', file=sys.stderr)

    limited_dataset = list(map(
        lambda elem: update_in(elem, ['predictions'], compose(list, partial(take, args.limit))),
        dataset))

    if args.by_feature is not None:
        strategy_meta = meta[args.strategy]
        data_keys = get_feature_data_keys(strategy_meta, args.by_feature)
        grouped = groupby(
            compose(lambda x: '/'.join(x), juxt(map(op.itemgetter, data_keys))),
            limited_dataset)
        evaluation_fn = partial(evaluate, strategy_meta, args.predictions_limit)
        output = {
            value: evaluation_fn(section) for value, section in grouped.items()
        }
    else:
        output = evaluate(meta[args.strategy], args.predictions_limit, limited_dataset)

    print(json.dumps(output, indent=4, separators=(',', ': ')),
          end='',
          file=args.output_evaluation_file)
