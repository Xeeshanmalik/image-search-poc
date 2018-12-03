import ujson
import sys
import argparse
import os.path as path

from toolz.itertoolz import groupby
from yattag import Doc, indent
from collections import OrderedDict

def ordered_dict_sorted_by_key(d):
    return OrderedDict(sorted(d.items(), key=lambda t: t[0], reverse=True))


def to_html_doc(title, items, image_base_path, image_width, image_height):
    doc, tag, text = Doc().tagtext()
    with tag('html'):
        with tag('head'):
            with tag('style'):
                doc.asis(f'img {{max-width:{image_width}px;max-height:{image_height}px;width:auto;height:auto;}}')
        with tag('body'):
            with tag('h1'):
                text(title)
            for year, year_items in ordered_dict_sorted_by_key(groupby(lambda item: item['year'], items)).items():
                with tag('h3'):
                    text(f'{year}')
                for item in year_items:
                    with tag('img',
                            src=image_base_path + '/' + item['image_id'] + '.jpg?authuser=1',
                            height=f'{image_height}', width=f'{image_width}'):
                        text('')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainset report, produces a file per make/model pair.')
    parser.add_argument('dataset_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('-i', '--image_base_path', type=str,
                        default='https://storage.cloud.google.com/dev_visual_search/images/cropped/v3',
                        help='base path of the folder where images are expected to be found.')
    parser.add_argument('-wd', '--image_width', type=int, default='128',
                        help='image width in  pixels.')
    parser.add_argument('-ht', '--image_height', type=int, default='128',
                        help='image height in  pixels.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Reports will be written to the folder at this path.')

    args = parser.parse_args()

    dataset = ujson.load(args.dataset_file)

    groups = groupby(lambda x: (x['make'], x['model']), dataset)

    for k, group in groups.items():
        make, model = k
        size = len(group)
        doc = to_html_doc(
            f'Make: {make}, Model: {model}, Size: {size}',
            group,
            args.image_base_path, args.image_width, args.image_height)

        with open(path.join(args.output_path, f'trainset-{make}-{model}.html'), mode='w') as f:
            print(indent(doc.getvalue()), file=f)
