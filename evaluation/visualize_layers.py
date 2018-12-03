import sys
import argparse
from keras.models import load_model
from toolz.itertoolz import take
from functools import partial
import os
import ujson
from toolz.functoolz import compose
from keras.preprocessing import image as Image
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


def image_exists(base_image_path, item):

    image_id = item['image_id']
    full_path = os.path.join(base_image_path, image_id) + '.jpg'
    if os.path.isfile(os.path.join(base_image_path, image_id) + '.jpg') and os.stat(os.path.join(base_image_path, image_id) + '.jpg').st_size > 128:
        return os.path.isfile(full_path)
    else: 
        print('image missing {full_path}', file=sys.stderr)

def load_img(item):

    image_id = item['image_id']
    image = Image.load_img(os.path.join(args.base_image_path, image_id) + '.jpg', grayscale=False,
                                        target_size=(args.image_height, args.image_width, args.image_channel))

    return {'image': image, 'image_id': image_id}

def compute_scores(model, layer_name, image):

    output = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    print(f'score in progress', file=sys.stderr)
    arr = np.asarray(image['image'])
    arr = np.expand_dims(arr, 0)
    score = output.predict(arr)

    return score

def visualize(activations):

    print(f'Assert batch size', file=sys.stderr)
    batch_size = activations.shape[0]
    assert batch_size == 1
    print(f'Display activation map', file=sys.stderr)
    shape = activations.shape
    if len(shape) == 4:
        activations = np.hstack(np.transpose(activations[0], (2, 0, 1)))
    return activations

def stacking(arr_of_activations):

    for layer, activation in enumerate(arr_of_activations):
        img = activation * 1./args.visualize_limit_per_images
        stacked[layer*activation.shape[0]: layer*activation.shape[0] + activation.shape[0],
        0:activation.shape[1]] += img
    return stacked


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize Layers')
    parser.add_argument('input_metadata_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
            help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to model\'s h5 file')
    parser.add_argument('-n', '--visualize_limit_per_images', type=int, help='number of images to visualize', default=2)
    parser.add_argument('-ih', '--image_height', type=int, help='Image height in pixels', default=128)
    parser.add_argument('-iw', '--image_width', type=int, help='Image width in pixels.', default=128)
    parser.add_argument('--base_image_path', type=str, required=True, help='Path of the folder containing the thumbnail images.')
    parser.add_argument('-ic', '--image_channel', type=int, help='Image channel', default=3)
    parser.add_argument('-l', '--layer_name', required=True, type=str, help='Name of the layer: layer_1, layer_2,... layer_20 and output')
    parser.add_argument('--save_visualization_path', required=True, type=str, help='output path of visualization')
    parser.add_argument('--vmin', type=float, help='set the color scaling for the image: min value', default=0)
    parser.add_argument('--vmax', type=float, help='set the color scaling for the image: max value', default=1)
    args = parser.parse_args()

    print(f'loading model...', file=sys.stderr)
    model = load_model(args.model_path)

    print(f'model summary...', file=sys.stderr)
    model.summary()


    items = list(take(
          args.visualize_limit_per_images,
          filter(partial(image_exists, args.base_image_path), ujson.load(args.input_metadata_file))))

    print('loaded items!', file=sys.stderr)

    save_visualization = compose(visualize,
                         partial(compute_scores, model, args.layer_name),
                         load_img)

    activations = map(save_visualization, items)

    arr_of_activations = np.asarray(list(activations))
    new_shape = ((args.visualize_limit_per_images - 1)* arr_of_activations[0].shape[0] + arr_of_activations[0].shape[0],
                 arr_of_activations[0].shape[1])

    stacked = np.zeros(new_shape, dtype=np.float)

    stacked = stacking(arr_of_activations)

    plt.imsave(os.path.join(args.save_visualization_path,'stacked.png'), stacked, vmin=args.vmin, vmax=args.vmax)


