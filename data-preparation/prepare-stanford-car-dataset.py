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
    p.add_argument('dataset_output_file', nargs='?', type=argparse.FileType('w'),
                   default=sys.stdout,
                   help='The prepared dataset will be writen to here.')

    args = p.parse_args()

    cars_meta = sio.loadmat(args.cars_meta)
    cars_annos = sio.loadmat(args.cars_annos)
    # classes_meta = json.load(args.classes_meta)

    classes = list(map(partial(get_in, [0]), cars_meta['class_names'][0]))

    def return_json(klass):
        split_desc = ''.join(klass).split(" ")

        if len(split_desc)==4:
            return {'make': split_desc[0],
               'model': split_desc[1],
               'body': split_desc[2],
               'year': split_desc[3]}
        else:
            return 0

    def to_dataset_elem(x):
        image_id = get_in([5, 0], x).replace('.jpg', '').replace('car_ims/', '')
        klass = classes[get_in([4, 0, 0], x) - 1]
        meta = return_json(klass)
        if meta !=0:
            pid = int(image_id)
            id = pid

            return {'body': meta['body'],
                'color': 'n/a',
                'make': meta['make'],
                'pid': pid,
                'seller': 'n/a',
                'image_id': f'{image_id}',
                'bounding_box': {'left': int(get_in([0, 0, 0], x)),
                                 'top': int(get_in([1, 0, 0], x)),
                                 'right': int(get_in([2, 0, 0], x)),
                                 'bottom': int(get_in([3, 0, 0], x))},
                'year': meta['year'],
                'id': id,
                'model': meta['model'],
                'image': 'n/a'}
        else:
            return None

    dataset = list(map(partial(to_dataset_elem), cars_annos['annotations'][0]))

    print('[')
    prev = None
    for item in dataset:
        if item is not None:
            if prev is not None:
                print(',')
            print(json.dumps(item, indent=4, separators=(',', ': ')),
                 end='',
                file=args.dataset_output_file)
            prev = item
    print('\n]')
