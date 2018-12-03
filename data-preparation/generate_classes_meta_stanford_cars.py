import sys
import argparse
import simplejson as json
import scipy.io as sio
import time
from functools import partial
from toolz.dicttoolz import get_in


if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Filters a dataset.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cars_meta', type=str, required=True)
    p.add_argument('--cars_annos', type=str, required=True)
    p.add_argument('classes_output_file', nargs='?', type=argparse.FileType('w'),
                   default=sys.stdout,
                   help='The prepared dataset will be writen to here.')

    args = p.parse_args()

    cars_meta = sio.loadmat(args.cars_meta)
    cars_annos = sio.loadmat(args.cars_annos)


    def to_dataset_elem(cars_meta,cars_annos):
       description = get_in([0], cars_meta)
       classes = str(get_in([0,0,0], cars_annos))

       split_desc = ''.join(description).split(" ")

       return {str(cars_meta[0]): {'id':classes,
               'make': split_desc[0],
               'model': split_desc[1],
               'body': split_desc[2],
               'year': split_desc[3]}}


    dataset = list(map(partial(to_dataset_elem), cars_meta["class_names"][0], cars_annos['annotations'][0]))

    print('[')
    prev = None
    for item in dataset:
        if prev is not None:
            print(',')
        print(json.dumps(item, indent=4, separators=(',', ': ')),
              end='',
              file=args.classes_output_file)
        prev = item
    print('\n]')