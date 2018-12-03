import sys
import argparse
import ijson
import json
from functools import partial, reduce
from random import random, randint
import toolz
from toolz.itertoolz import mapcat, concat, groupby
from toolz.dicttoolz import keyfilter
from yattag import Doc, indent
import csv

def run(metadata_file):
    items = ijson.items(metadata_file, 'item')

    def accum(d, item):
        d['count'] += 1
        if item['car_exterior_present'] and item['car_detected']:
            d['true_positive'] += 1
        elif not item['car_exterior_present'] and item['car_detected']:
            d['false_positive'] += 1
        elif not item['car_exterior_present'] and not item['car_detected']:
            d['true_negative'] += 1
        elif item['car_exterior_present'] and not item['car_detected']:
            d['false_negative'] += 1

        return d

    r = reduce(accum, items, {
        'count': 0,
        'true_positive': 0,
        'false_positive': 0,
        'true_negative': 0,
        'false_negative': 0})

    precision = r['true_positive'] / (r['true_positive'] + r['false_positive'])
    recall = r['true_positive'] / (r['true_positive'] + r['false_negative'])

    print(f'precision = {precision}')
    print(f'recall = {recall}')

    print(r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='computes car detection error')

    args = parser.parse_args()

    run(sys.stdin)

