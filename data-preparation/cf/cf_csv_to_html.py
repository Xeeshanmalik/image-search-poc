import sys
import argparse
from functools import partial, reduce
from random import random, randint
import toolz
from toolz.itertoolz import mapcat, concat, groupby
from yattag import Doc, indent
import csv

def to_html_doc(items, image_base_path, image_width, image_height):
    doc, tag, text = Doc().tagtext()
    with tag('html'):
        with tag('head'):
            with tag('style'):
                doc.asis(f'img {{max-width:{image_width}px;max-height:{image_height}px;width:auto;height:auto;}}')
        with tag('body'):
            for item in items:
                with tag('img',
                        src=image_base_path + '/' + item['image_id'] + '.jpg',
                        height=f'{image_height}', width=f'{image_width}'):
                    text('')
    return doc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrowdFlower CSV Report to Html converter.')
    parser.add_argument('-i', '--image_base_path', type=str, default='.',
            help='base path of the folder where images are expected to be found.')
    parser.add_argument('-wd', '--image_width', type=int, default='128',
            help='image width in  pixels.')
    parser.add_argument('-ht', '--image_height', type=int, default='128',
            help='image height in  pixels.')

    args = parser.parse_args()

    items = csv.DictReader(sys.stdin)

    doc = to_html_doc(items, args.image_base_path, args.image_width, args.image_height)
    print(indent(doc.getvalue()))
