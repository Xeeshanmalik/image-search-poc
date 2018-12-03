import argparse
import os
import numpy as np
from keras.preprocessing import image as Image
from keras.models import load_model
from toolz.functoolz import compose
from toolz.itertoolz import take
from toolz.dicttoolz import assoc_in
from functools import partial
import ujson
import sys
import os.path
import matplotlib
from keras.models import Model
matplotlib.use('Agg')
from matplotlib import pyplot as plot
sys.path.append("..")
from common.pretty_json import pretty_json


def image_exists(base_image_path, item):
    image_id = item['image_id']
    full_path = os.path.join(base_image_path, image_id) + '.jpg'
    if os.path.isfile(os.path.join(base_image_path, image_id) + '.jpg') and os.stat(os.path.join(base_image_path, image_id) + '.jpg').st_size > 128:
        return os.path.isfile(full_path)
    else:
        print('image missing {full_path}', file=sys.stderr)

def flatten_score(score):
    n_data = score.shape[0]
    flatten_dim = np.prod(score.shape[1:])
    x_data_flatten = score.reshape((n_data, flatten_dim))

    return x_data_flatten[0]


def predict_score(model, image):
    image = np.expand_dims(image, 0)  # Add batch dimension
    score = model.predict(image)
    return score


def load_image(base_image_path, item):
    image_id = item['image_id']

    grayscale = True if args.image_channel == 1 else False

    image = np.asarray(Image.load_img(
        os.path.join(base_image_path, image_id) + '.jpg',
        grayscale=grayscale,
        target_size=(args.image_height, args.image_width)))

    if grayscale:
        image = np.asarray(image, dtype='float32')
        image = image.reshape([args.im_height, args.im_width, 1])

    # normalize
    if np.max(image) - np.min(image) != 0:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def update_feature_maps(model_name, scoring_fn, item):
    return assoc_in(item, ['feature_maps', model_name], scoring_fn(item))


def save_score_map(skip, base_path, image_id, score_map):
    if skip is False:
        image_data = score_map.reshape([args.image_height, args.image_width])
        im = plot.imshow(image_data, extent=(0, args.image_height, 0, args.image_width))
        im.set_cmap('hot')
        plot.axis('off')
        plot.savefig(f'{base_path}/{image_id}.png', bbox_inches='tight')
    return score_map


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs Prediction')
    parser.add_argument('input_metadata_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the thumbnail images.')
    parser.add_argument('--model_name', required=True, type=str, help='Name of model')
    parser.add_argument('--model_path', required=True, type=str, help='Path to model\'s h5 file')
    parser.add_argument('-ih', '--image_height', required=True, type=int,
                        help='Image height in pixels')
    parser.add_argument('-iw', '--image_width', required=True, type=int,
                        help='Image width in pixels.')
    parser.add_argument('-ic', '--image_channel', required=True, type=int,
                        help='Image height in pixels')
    parser.add_argument('-l', '--limit', type=int, default=2 ** 32)
    parser.add_argument('--save_score_map', default=False, action='store_true',
                        help='Save score map ?')
    parser.add_argument('--score_map_output_path', type=str, required=False,
                        help='Path of the folder score maps will be writen to')
    parser.add_argument('-ll', '--layer_name', required=True, type=str,
                        help='Name of the layer: layer_1, layer_2,... layer_20 and output')
    parser.add_argument('-a', '--activation', default=False, action='store_true',
                        help='Add layer activations to output ?')

    args = parser.parse_args()

    model = load_model(args.model_path)
    output = Model(inputs=model.input, outputs=model.get_layer(args.layer_name).output)

    items = list(take(
          args.limit,
          filter(partial(image_exists, args.base_image_path), ujson.load(args.input_metadata_file))))
    print('loaded items!', file=sys.stderr)

    compute_score = compose(
      flatten_score,
      partial(predict_score, output),
      partial(load_image, args.base_image_path))

    scores = map(compute_score, items)
    print('scoring items...', file=sys.stderr)

    def binarize(threshold, score):
        return list(map(lambda x: 1 if float(x) > threshold else 0, score))

    to_feature_map = compose(
            partial(binarize, 0.5),
            partial(save_score_map, not args.save_score_map, args.score_map_output_path))

    def augment(item, score):
        item['feature_maps'] = {args.model_name: to_feature_map(item['image_id'], score)}
        item['activations'] = {args.model_name: score.tolist()}
        return item

    print('[')
    prev = None
    for item in map(
            lambda t: augment(t[0], t[1]),
            zip(items, scores)):
        if prev is not None:
            print(',')
        print(pretty_json(item), end='')
        prev = item
    print('\n]')

