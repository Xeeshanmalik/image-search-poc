import sys
import argparse
import ijson
import json
from functools import partial, reduce
from random import random, randint
import toolz
from toolz.itertoolz import mapcat, concat, groupby
from yattag import Doc, indent
import csv

def run(json_stream, csv_stream):
    reader = csv.DictReader(csv_stream) 
    car_exterior_present = {x['image_id']:(True if x['car_exterior_present']=='yes' else False) for x in reader}
    items = ijson.items(json_stream, 'item')

    print('[')
    prev = None
    for item in [x for x in items if x['image_id'] in car_exterior_present]:
        if prev is not None:
            print(',')
        image_id = item['image_id']
        item['car_exterior_present'] = car_exterior_present[image_id]
        print(json.dumps(item), end='')
        prev = item
    print('\n]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Filters json file, keepinng only rows for which there is data in
        the CrowdFlower CSV Report.
    """)
    parser.add_argument('-c', '--csv_path', type=str, default='.',
            help='path to CF csv file.')

    args = parser.parse_args()

    with open(args.csv_path) as csvfile:
        run(sys.stdin, csvfile)

