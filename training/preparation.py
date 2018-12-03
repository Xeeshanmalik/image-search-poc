import argparse
import os
import sys
import multiprocessing
import ujson
from toolz.functoolz import compose, pipe
from functools import partial
from multiprocessing import Pool
import time
import shutil


def create_sets(base_image_path, path, label,item):

    image_id = item['image_id']
    model_name = item[label]

    if os.path.isfile(os.path.join(base_image_path, image_id) + '.jpg') and os.stat(os.path.join(base_image_path, image_id) + '.jpg').st_size > 128:
        new_image_name = model_name + '_' + image_id
        shutil.copy((os.path.join(base_image_path, image_id) + '.jpg'), os.path.join(path, model_name)+'/' +
                    new_image_name + '.jpg')
    else:
        print('images_' + image_id + '.jpg', "does not exist")
    return {'image_id': image_id}


def create_dir(path, class_names):

    for class_name in class_names:
        if not os.path.isdir(os.path.join(path,class_name)):
            os.makedirs(os.path.join(path, class_name))
        else:
            shutil.rmtree(os.path.join(path, class_name))
            os.makedirs(os.path.join(path, class_name))


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs saliency detection')
    parser.add_argument('-tt', '--meta_file', required=True, type=argparse.FileType('r'))
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the thumbnail images.')
    parser.add_argument('-p', '--number_of_processes', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-c', '--chunk_size', type=int, default=1)
    parser.add_argument('-l', '--limit', type=int, default=sys.maxsize)
    parser.add_argument('--input_stat_file', type=argparse.FileType('r'), required=True,
                        help="count distinct classes")
    parser.add_argument('-tr', '--preparation_set_image_path', required=True, help='path to preparation '
                                                                                   'directory data generator '
                                                                                   'should follow '
                                                                                   'input/'
                                                                                   'set-type(train/test)'
                                                                                   '/class_name/'
                                                                                   'class_name_....jpg')

    parser.add_argument('-t', '--label', required=True, help="possible values: 'model', 'body'")

    args = parser.parse_args()

    items_stats = ujson.load(args.input_stat_file)
    distinct_classes = get_distinct_classes(items_stats, args.label)

    items = ujson.load(args.meta_file)

    create_dir(args.preparation_set_image_path, distinct_classes)

    preparation_set = compose(partial(create_sets,
                                      args.base_image_path,
                                      args.preparation_set_image_path,
                                      args.label))

    p = Pool(args.number_of_processes)

    start = time.clock()

    for image in p.imap(preparation_set, items, args.chunk_size):
        print("Successfully Saved:", image['image_id'] + '.jpg')

    print("Total Time:", time.clock() - start)

