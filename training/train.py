from __future__ import print_function

import argparse
import sys
import os
import multiprocessing
import tensorflow as tf
import ujson
import numpy as np


from keras.applications import ResNet50
from model import autoencoder
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model, print_summary



class DelegatingCallbackWithFixedModel(Callback):

    def __init__(self, delegate, fixed_model):
        super(DelegatingCallbackWithFixedModel, self).__init__()
        self.delegate = delegate
        self.fixed_model = fixed_model

    def on_train_begin(self, logs=None):
        self.delegate.set_model(self.fixed_model)
        self.delegate.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.delegate.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.delegate.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.delegate.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        self.delegate.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.delegate.on_batch_end(batch, logs)


class ImageAugmentation:

    def __init__(self, input_img_gen):

        self.generator = input_img_gen

    def train_data_generator(self, train_set_path, batch_size, im_height, im_width, labels):

        training_set = self.generator.flow_from_directory(
            train_set_path, target_size=(im_width, im_height),
            batch_size=batch_size, shuffle=True, class_mode='categorical')

        while True:
            x_train = training_set.next()
            yield (x_train[0], x_train[1])

    def dev_data_generator(self, dev_set_path, batch_size, im_height, im_width, labels):

        dev_set = self.generator.flow_from_directory(
            dev_set_path, target_size=(im_width, im_height),
            batch_size=batch_size, shuffle=True, class_mode='categorical')

        while True:
            x_dev = dev_set.next()
            yield (x_dev[0], x_dev[1])


def train_model(gpu_count, image_shape, batch_gen, batch_gen_val, len_train_set, len_dev_set, len_class_size):

    checkpoint = ModelCheckpoint(
        filepath=args.model_save_path + args.model_name + '-sh='
        + str(args.sh) + '-ep={epoch:02d},bz=' + str(args.batch_size) + ',loss={loss:.4f}.h5',
        monitor='loss',
        verbose=1)

    if gpu_count > 1:
        with tf.device('/cpu:0'):
            base_model = autoencoder(image_shape, len_class_size)
            model = base_model.model()
            print_summary(model, print_fn=lambda x: print(x, file=sys.stderr))
            model.compile(loss='binary_crossentropy', optimizer='sgd')
            checkpoint = DelegatingCallbackWithFixedModel(checkpoint, model)

        model = multi_gpu_model(model, gpus=gpu_count)
    else:
        # base_model = autoencoder(image_shape, len_class_size)
        base_model = ResNet50(include_top=True, weights=None, input_tensor=None,
                              input_shape=image_shape, pooling=None, classes=len_class_size)
        model = base_model
        print_summary(model, print_fn=lambda x: print(x, file=sys.stderr))

    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.fit_generator(
        batch_gen,
        steps_per_epoch=len_train_set // args.batch_size,
        epochs=args.number_of_epochs,
        verbose=1,
        shuffle=True,
        callbacks=[checkpoint],
        validation_data=batch_gen_val,
        validation_steps=len_dev_set // args.batch_size)


def get_distinct_classes(items, label):

    def return_model():
        makes = list(items['make'])
        model = []
        for index in makes:
            model.append(list(items[label][index].keys()))
        return set(list([item for sublist in model for item in sublist]))

    def return_body():
        body = []
        for item in items[label]:
            body.append(item)
        return set(list(body))

    switcher = {
        'model': return_model,
        'body': return_body
    }
    choices = switcher.get(label, lambda: "nothing")

    return choices()


def get_images_labels(item):

    return {'image_id': item['image_id'], 'label': item['label']}


def get_numpy_labels(items):

    array = np.ndarray((len(items.values()[0]), len(items['label'][0])))

    for index, val in enumerate(items['label']):
        array[index] = val

    return array

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs saliency detection')
    parser.add_argument('--model_save_path', required=True,
                        type=lambda x: os.mkdir(x) if not os.path.isdir(x) else x,
                        help='Saved model path of the first modle')
    parser.add_argument('-mn', '--model_name', type=str, default='model')
    parser.add_argument('-ep', '--number_of_epochs', type=int, default=120)
    parser.add_argument('-bb', '--batch_size', type=int, default=16)
    parser.add_argument('-gp', '--gpu_state', type=bool, default=False)
    parser.add_argument('-gc', '--gpu_count', type=int, default=0)
    parser.add_argument('-ih', '--im_height', required=True, type=int, help='Image height in pixels')
    parser.add_argument('-iw', '--im_width', required=True, type=int, help='Image width in pixels.')
    parser.add_argument('-ic', '--im_channel', required=True, type=int, help='Image height in pixels')
    parser.add_argument('-p', '--number_of_processes', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-c', '--chunk_size', type=int, default=1)
    parser.add_argument('-l', '--limit', type=int, default=sys.maxsize)
    parser.add_argument('-s', '--sh', type=bool, default=False)
    parser.add_argument('-tr', '--train_images_path', required=True, help='path to the train dir')
    parser.add_argument('-te', '--dev_images_path', required=True,   help='path to the validation dir')
    parser.add_argument('--train_stat_file', type=argparse.FileType('r'), required=True,
                        help="count distinct classes")
    parser.add_argument('--dev_stat_file', type=argparse.FileType('r'), required=True,
                        help="count distinct classes")
    parser.add_argument('--train_meta_file', type=argparse.FileType('r'), required=True,
                        help="input meta file for training")
    parser.add_argument('--dev_meta_file', type=argparse.FileType('r'), required=True,
                        help="input meta file for dev")
    parser.add_argument('-t', '--label', required=True, help="possible values: 'model', 'body'")

    # Augmentation Parameters

    parser.add_argument('-ia_re', '--rescale', type=float, default=1./255,
                        help=""" rescaling factor. Defaults to None. If None or 0, no rescaling
                        is applied, otherwise we multiply the data by the value provided
                        (before applying any other transformation """)
    parser.add_argument('-ia_sr', '--shear_range', type=float, default=0.,
                        help=""" Float. Shear Intensity
                                 (Shear angle in counter-clockwise direction as radians)""")
    parser.add_argument('-ia_zm', '--zoom_range', type=float, default=0.,
                        help=""" Float or [lower, upper]. Range for random zoom.
                                 If a float,[lower, upper] = [1-zoom_range, 1+zoom_range].""")
    parser.add_argument('-ia_rr', '--rotation_range', type=int, default=0,
                        help=""" Int. Degree range for random rotations.""")
    parser.add_argument('-ia_hf', '--horizontal_flip', type=bool, default=False,
                        help=""" Boolean. Randomly flip inputs horizontally.""")
    parser.add_argument('-ia_vf', '--vertical_flip', type=bool, default=False,
                        help=""" Boolean. Randomly flip inputs vertically.""")
    parser.add_argument('-ia_zc', '--zca_whitening', type=bool, default=False,
                        help=""" Apply ZCA whitening. """)
    parser.add_argument('-ia_wsr', '--width_shift_range', type=float, default=0,
                        help=""" Float (fraction of total width). Range for random horizontal shifts. """)
    parser.add_argument('-ia_hsr', '--height_shift_range', type=float, default=0,
                        help=""" Float (fraction of total height). Range for random vertical shifts. """)

    args = parser.parse_args()

    if args.gpu_state is not True:

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    input_image_shape = [args.im_height, args.im_width, args.im_channel]

    # Get Distinct Classes

    train_stats = ujson.load(args.train_stat_file)
    train_distinct_classes = get_distinct_classes(train_stats, args.label)

    dev_stats = ujson.load(args.dev_stat_file)
    dev_distinct_classes = get_distinct_classes(dev_stats, args.label)

    # Note: Need to use the stat file to get the length of train and dev sets in the future

    train_set_len = 0
    dev_set_len = 0

    for class_name in train_distinct_classes:
        train_set_len += len([name for name in os.listdir(os.path.join(args.train_images_path, class_name))
                         if os.path.isfile(os.path.join(os.path.join(args.train_images_path, class_name),
                                                        name))])
    for class_name in dev_distinct_classes:
        dev_set_len += len([name for name in os.listdir(os.path.join(args.dev_images_path, class_name))
                       if os.path.isfile(os.path.join(os.path.join(args.dev_images_path, class_name),
                                                      name))])
    # Compute list of file_names and label

    train_input = ujson.load(args.train_meta_file)
    train_dict, dev_dict = {}, {}

    for item in map(get_images_labels, train_input):
        train_dict.setdefault('image_id', []).append(item['image_id'])
        train_dict.setdefault('label', []).append(item['label'])

    np_train_labels = get_numpy_labels(train_dict)

    print(np_train_labels.shape)
    class_size = np_train_labels.shape[1]

    dev_input = ujson.load(args.dev_meta_file)
    for item in map(get_images_labels, dev_input):
        dev_dict.setdefault('image_id', []).append(item['image_id'])
        dev_dict.setdefault('label', []).append(item['label'])

    np_dev_labels = get_numpy_labels(dev_dict)

    train_data_gen = ImageDataGenerator(rescale=args.rescale,
                                        shear_range=args.shear_range,
                                        zoom_range=args.zoom_range,
                                        rotation_range=args.rotation_range,
                                        horizontal_flip=args.horizontal_flip,
                                        vertical_flip=args.vertical_flip,
                                        zca_whitening=args.zca_whitening,
                                        width_shift_range=args.width_shift_range,
                                        height_shift_range=args.height_shift_range)

    dev_data_gen = ImageDataGenerator(rescale=args.rescale)

    train = ImageAugmentation(train_data_gen)

    generator = train.train_data_generator(args.train_images_path, args.batch_size, args.im_height,
                                           args.im_width, np_train_labels)

    dev = ImageAugmentation(dev_data_gen)

    dev_generator = dev.dev_data_generator(args.dev_images_path, args.batch_size, args.im_height,
                                           args.im_width, np_dev_labels)

    train_model(args.gpu_count,
                input_image_shape, generator, dev_generator, train_set_len, dev_set_len, class_size)
