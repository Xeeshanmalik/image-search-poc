from __future__ import print_function
import sys
import os
import argparse
import ujson
import json
import numpy as np
from keras import backend as K
from keras.models import load_model
import keras
from yad2k.models.keras_yolo import yolo_eval, yolo_head
from toolz.functoolz import compose, pipe
from toolz.itertoolz import topk, get
from toolz.dicttoolz import assoc
from functools import partial
from itertools import imap
from PIL import Image


def image_exists(base_image_path, item):
    image_id = item['image_id']
    full_path = os.path.join(base_image_path, image_id) + '.jpg'
    exists = os.path.isfile(full_path)
    if not exists:
        print('image missing ' + full_path, file=sys.stderr)
        return False

    return True


def read_classes(path):
    with open(path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def read_anchors(path):
    with open(path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def validate(args):
    # TODO
    assert True


def detect_objects(yolo_model, boxes, scores, classes, class_names, sess, image):

    if image is None:
        return []

    image_data = np.array(image, dtype='float32')
    image_data /= 255.  # Normalized
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0})

    def decode(out_box, out_score, out_class):
        predicted_class = class_names[out_class]
        label = '{}'.format(predicted_class)

        top, left, bottom, right = out_box

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        return {'left': int(left),
                'top': int(top),
                'right': int(right),
                'bottom': int(bottom),
                'label': label,
                'probability': float(out_score)}

    return map(decode, out_boxes, out_scores, out_classes)


def load_image(item):
    try:
        image_id = item['image_id']
        full_path = os.path.join(args.base_image_path, image_id) + '.jpg'
        image = Image.open(full_path)
        return image
    except IOError:
        print('image is not valid ' + full_path, file=sys.stderr)
        return None


def pre_process_image(image):
    if image is None:
        return None

    target_size = tuple(reversed(model_image_size))

    image = image.convert('RGB')
    image.thumbnail(target_size, Image.ANTIALIAS)
    resized_image = Image.new('RGB', target_size, 'white')
    resized_image.paste(image)

    return resized_image


def keep_max_bounding_box(labels, bounding_boxes):

    def area(box):
        return abs(box['bottom'] - box['top']) * abs(box['right'] - box['left'])

    return pipe(bounding_boxes,
                partial(filter, lambda x: x['label'] in labels),
                lambda x: topk(1, x, key=area),
                lambda x: get(0, x, None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs object detection using YOLO.')
    parser.add_argument('input_metadata_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('output_metadata_file', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='The updated output file. Writes to stdout by default.')
    parser.add_argument('--yolo_model_path', type=str, default='./model_data/yolo.h5',
                        help='Path to the YOLO model.')
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the images.')
    parser.add_argument('--yolo_anchors', type=str, default='./model_data/yolo_anchors.txt',
                        help='Path to the anchors of YOLO model.')
    parser.add_argument('--yolo_classes', type=str, default='./model_data/coco_classes.txt',
                        help='Path to the classes of YOLO model.')

    args = parser.parse_args()
    validate(args)

    yolo_model = load_model(args.yolo_model_path)
    anchors = read_anchors(args.yolo_anchors)
    class_names = read_classes(args.yolo_classes)

    model_image_size = yolo_model.layers[0].input_shape[1:3]
    print('max BB size: ' + str(model_image_size), file=sys.stderr)

    sess = K.get_session()

    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

    keras.utils.print_summary(yolo_model, print_fn=lambda x: print(x, file=sys.stderr))

    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
            yolo_outputs, input_image_shape,
            score_threshold=0.4, iou_threshold=0.5)

    items = filter(partial(image_exists, args.base_image_path), ujson.load(args.input_metadata_file))

    # composing from the right will make this code more readable
    detect_vehicle = compose(
            partial(keep_max_bounding_box, {'car', 'truck'}),
            partial(detect_objects, yolo_model, boxes, scores, classes, class_names, sess),
            pre_process_image,
            load_image)

    print('[')
    prev = None
    for x in imap(lambda item: assoc(item, 'bounding_box', detect_vehicle(item)), items):
        if not x['bounding_box']:
            continue
        if prev is not None:
            print(',')
        print(json.dumps(x, indent=4, separators=(',', ': ')), end='')
        prev = x
    print('\n]')
