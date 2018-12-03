import argparse
import sys
import ujson
import multiprocessing as mp
import operator as op

from functools import partial
from toolz.dicttoolz import keyfilter, merge, get_in
from toolz.itertoolz import take, concat, topk, first
from toolz.functoolz import pipe
from scipy.spatial import distance
from rtree import index

"""
Prerequisites:
    - libspatialindex
      - linux: sudo apt-get install libspatialindex-c4v5
      - osx: brew install spatialindex
"""
# also do BK-Tree!!!!

trainset_rtree = None
trainset_fms = None


def to_rtree_coords(activations):
    coords = list(concat([activations, activations]))
    return coords


def init_trainset_rtree(feature_maps_d):
    """
    feature_maps: {image_id: [feature_map]}
    """
    global trainset_rtree

    d = pipe(feature_maps_d.values(), first, len)
    print(f'feature map dimension = {d}', file=sys.stderr)

    p = index.Property()
    p.dimension = d
    p.storage = index.RT_Memory
    p.type = index.RT_RTree
    p.variant = index.RT_Star

    def to_index_record(t):
        image_id, feature_map = t
        return int(image_id), to_rtree_coords(feature_map), image_id

    print(f'loading index...', file=sys.stderr)
    trainset_rtree = index.Index(map(to_index_record, feature_maps_d.items()),
                                 properties=p,
                                 interleaved=True)
    print(f'loaded index!', file=sys.stderr)


def predict_euclidean(limit, distance_fn, fm):
    point = to_rtree_coords(fm)
    # we need to limit the results of nearest because it returns more than
    # limit when the model overfits to the training data
    nearest_neighbors = take(limit, trainset_rtree.nearest(point, num_results=limit, objects=True))
    return list(map(lambda x: x.object, nearest_neighbors))


def init_trainset_fms(feature_maps_d):
    global trainset_fms
    trainset_fms = list(feature_maps_d.items())


def predict_discrete(limit, distance_fn, fm):
    nearest_neighbors = topk(
        limit, trainset_fms,
        key=lambda x: - distance_fn(x[1], fm))

    return map(op.itemgetter(0), nearest_neighbors)


def pick(whitelist, d):
    return keyfilter(lambda k: k in whitelist, d)


def to_prediction(o, p, feature_map_fn, distance_fn):
    d = distance_fn(feature_map_fn(o), feature_map_fn(p))
    return merge(
            pick(['image_id', 'make', 'model', 'body', 'year', 'color'], p),
            {'distance': d})


def predict(meta, trainset, dataset, predict_limit, output_file):

    feature_map_fn = partial(get_in, [meta['data_key'], 'thumbnail'])
    d_fn = meta['distance_fn']
    fms_trainset_d = dict({element['image_id']: feature_map_fn(element) for element in trainset})
    fms_dataset = list(map(feature_map_fn, dataset))

    trainset_d = {row['image_id']: row for row in trainset}

    predict_fn = partial(meta['predict_fn'], predict_limit, d_fn)

    with mp.Pool(args.number_of_processes,
                 initializer=meta['initializer'], initargs=[fms_trainset_d]) as pool:
        print('[', file=output_file)
        prev = None
        for row, image_ids in zip(dataset,
                                  pool.imap(predict_fn, fms_dataset, args.chunk_size)):
            if prev is not None:
                print(',', file=output_file)
            predictions = map(lambda image_id: to_prediction(row, trainset_d[image_id], feature_map_fn, d_fn),
                              image_ids)
            row['predictions'] = list(predictions)
            del row['feature_maps']
            del row['activations']
            print(ujson.dumps(row, indent=4), end='', file=output_file)
            prev = row
        print('\n]', file=output_file)


if __name__ == '__main__':

    meta = {'euclidean': {'initializer': init_trainset_rtree,
                          'predict_fn': predict_euclidean,
                          'distance_fn': distance.euclidean,
                          'data_key': 'activations'},
            'hamming': {'initializer': init_trainset_fms,
                        'predict_fn': predict_discrete,
                        'distance_fn': distance.hamming,
                        'data_key': 'feature_maps'}}

    parser = argparse.ArgumentParser(
        description='Computes predictions for all element in the dataset_file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='Dataset file, with feature maps, to compute image predictioins for')
    parser.add_argument('output_file', nargs='?', type=argparse.FileType('rw'),
                        default=sys.stdout,
                        help='The output file, predictions will be writen to it')

    parser.add_argument('-tfm', '--trainset_file', required=True,
                        type=argparse.FileType('r'),
                        help='Train dataset, with feature maps. Predictions will be picked from here')
    parser.add_argument('-n', '--predict_limit', type=int, default=10,
                        help='Number of predictions to compute per image')
    parser.add_argument('-l', '--limit', type=int, default=sys.maxsize,
                        help='Limit the items for compute predictions for')
    parser.add_argument('--distance', type=str, choices=meta.keys(), default='euclidean')
    parser.add_argument('-p', '--number_of_processes', type=int, default=mp.cpu_count())
    parser.add_argument('-c', '--chunk_size', type=int, default=1)
    args = parser.parse_args()

    print(f'loading trainset...', file=sys.stderr)
    trainset = ujson.load(args.trainset_file)
    print(f'loaded trainset {len(trainset)}', file=sys.stderr)

    print(f'loading dataset...', file=sys.stderr)
    dataset = list(take(args.limit, ujson.load(args.dataset_file)))
    print(f'loaded dataset {len(dataset)}', file=sys.stderr)

    predict(meta[args.distance], trainset, dataset, args.predict_limit, args.output_file)
