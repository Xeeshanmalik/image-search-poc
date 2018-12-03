import ujson
import argparse
import sys
from toolz.dicttoolz import assoc
from toolz.functoolz import compose, pipe
from functools import partial
import json
sys.path.append("..")


def return_zero_vector(items_stats, label):

    vectors_of_zeros = []

    def return_items_per_class(item):
        return item

    def get_make_model(makes):
        models = []
        for m in makes:
            models.append(list(items_stats[label][m]))
        return models

    def get_model_vector(labels):
        coding = {}
        for i in labels:
            for labels in map(return_items_per_class, i):
                coding[labels] = 0
        return coding

    def get_body_vector(label):
        coding = {}
        for index, item in enumerate(items_stats[label].keys()):
            if index > 0 :
                coding[index] = 0
        return coding

    if label == 'model':
        make = list(items_stats['make'])
        classes = get_make_model(make)
        vectors_of_zeros = get_model_vector(classes)
    elif label == 'body':
        vectors_of_zeros = get_body_vector(label)

    return vectors_of_zeros

def compute_label(one_hot_codes, label, item):

    one_hot_codes = dict(one_hot_codes)
    one_hot_codes[item[label]]= 1

    return list(one_hot_codes.values())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs one-hot encoding')
    parser.add_argument('input_metadata_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('output_metadata_file', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='The updated output file. Writes to stdout by default.')
    parser.add_argument('--input_stat_file', type=argparse.FileType('r'), required=True, help="to compute total number of"
                        "bits required for one-hot-coding vector")

    parser.add_argument('--label', default='model', help="possible values: 'model', 'body'")


    args = parser.parse_args()


    print(f'loading input meta file...', file=sys.stderr)
    items = ujson.load(args.input_metadata_file)

    print(f'loading stats file..', file=sys.stderr)
    stats = ujson.load(args.input_stat_file)

    # create dictionary of stats

    zero_vector = return_zero_vector(stats, args.label)


    # composing from the right will make this code more readable
    compute_labels = compose(partial(compute_label, zero_vector, args.label))

    print('[')
    prev = None
    for x in map(lambda item: assoc(item, 'label', compute_labels(item)), items):
        if not x['label']:
            continue
        if prev is not None:
            print(',')
        print(json.dumps(x, indent=2, separators=(',', ': ')), end='')
        prev = x
    print('\n]')








