import argparse
import os
import sys
from functools import partial
from toolz.dicttoolz import keyfilter
from toolz.itertoolz import take, groupby
import ujson
from scipy.spatial import distance
import numpy as np
from sklearn.manifold import TSNE
from keras.preprocessing import image as Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def pick(whitelist, d):
    return keyfilter(lambda k: k in whitelist, d)


def label(key, item):

    return item[key]


def feature_map(key, load_image, item):

    image_id = item['image_id']

    d = {'image_id': image_id,
         'feature_maps': np.array(item[key]['thumbnail'])}

    if load_image is True:
        image = Image.load_img(os.path.join(args.image_base_path, image_id) + '.jpg',
                               grayscale=False,
                               target_size=(args.tb_height, args.tb_width, args.tb_channel))
        d['image'] = np.asarray(image)

    return d


def t_sne(X):
    X_embedded = TSNE(n_components=2, metric=distance.hamming).fit_transform(X)
    return X_embedded

def plot_points(title, X, L):

    plt.figure(figsize=(6, 6))

    for label, X_label in groupby(lambda t: L[t[0]], enumerate(X)).items():
        # these 2 lines are hacks!!!!
        X_label = list(map(lambda x: x[1], X_label))
        X_label = np.array(X_label)
        plt.scatter(X_label[:, 0], X_label[:, 1], label=label)

    plt.title(title)
    plt.xlabel('t-sne 1')
    plt.ylabel('t-sne 2')
    plt.legend()
    return plt


def plot_images(X, image, ax=None, zoom=1):

    artists = []
    for i in range(len(image)):
        x, y = np.atleast_1d(X[i, 0], X[i, 1])
        im = OffsetImage(image[i], zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([X[:, 0], X[:, 1]]))
    ax.autoscale()


def display_points(items, type):
    im = []
    feature_maps = []
    if type == 'points':
        for x in items:
            feature_maps.append(x['feature_maps'])
        return feature_maps, [0]
    elif type == 'images':
        for x in items:
            feature_maps.append(x['feature_maps'])
            im.append(x['image'])
        return feature_maps, im


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs Evaluation to compute precision')
    parser.add_argument('feature_maps', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help='Feature maps to visualize')
    parser.add_argument('--image_base_path', type=str, default='.',
                        help='base path of the folder where images are expected to be found.')
    parser.add_argument('-l', '--limit', type=int, default=sys.maxsize,
                        help='Limit the items for compute predictions for')
    parser.add_argument('--clazz', type=str, default='make',
                        choices=['make', 'model', 'body', 'color', 'year'],
                        help='Class we want to evaluate the clustering on')
    parser.add_argument('-th', '--tb_width', type=int, default='128',
                        help='Image width in pixels.')
    parser.add_argument('-tw', '--tb_height', type=int, default='128',
                        help='Image height in  pixels.')
    parser.add_argument('-tc', '--tb_channel', type=int, default='3',
                        help='Image height in  pixels.')
    parser.add_argument('-pt', '--plot_type', type=str, default='points',
                        choices=['points', 'images'],
                        help='Image height in  pixels.')
    parser.add_argument('-i', '--viz_base_path', type=str,
                        help='When specified, the viz will be saved to this path.')
    parser.add_argument('--viz_activations', default=False, action='store_true',
                        help='Visualize activations pre-binarization!')

    args = parser.parse_args()
    print(args)

    items = list(take(args.limit, ujson.load(args.feature_maps)))

    data_points = map(
        partial(feature_map,
                'activations' if args.viz_activations else 'feature_maps',
                args.plot_type == 'images'),
        items)

    if args.plot_type == 'points':

        feature_maps, im = display_points(data_points, args.plot_type)
        X_2d = t_sne(np.asarray(feature_maps))
        L = np.array(list(map(partial(label, args.clazz), items)))
        viz = plot_points(args.clazz, X_2d, L)
        if args.viz_base_path is not None:
            tag = 'activations' if args.viz_activations else 'binarized'
            viz.savefig(f'{args.viz_base_path}/viz-clustering-{tag}-{args.clazz}.svg',
                        bbox_inches='tight')
        else:
            viz.show()

    elif args.plot_type == 'images':

        feature_maps, im = display_points(data_points, args.plot_type)
        X_2d = t_sne(np.asarray(feature_maps))
        im = np.asarray(im)
        fig, ax = plt.subplots()
        plt.title(args.clazz)
        plot_images(X_2d, im, ax=ax, zoom=0.12)
        if args.viz_base_path is not None:
            tag = 'activations' if args.viz_activations else 'binarized'
            plt.savefig(f'{args.viz_base_path}/viz-clustering-{tag}-images.png')
        else:
            plt.show()
