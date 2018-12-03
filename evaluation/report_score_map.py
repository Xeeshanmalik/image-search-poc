import ijson
import sys
import argparse
from functools import partial, reduce
from random import random, randint
import toolz
from toolz.itertoolz import mapcat, concat, groupby
from yattag import Doc, indent

def to_html_doc(title, items, image_base_path, score_map_base_path, image_width, image_height):
    doc, tag, text = Doc().tagtext()
    with tag('html'):
        with tag('head'):
            with tag('style'):
                doc.asis(f'img {{max-width:{image_width}px;max-height:{image_height}px;width:auto;height:auto;}}')
        with tag('body'):
            with tag('h1'):
                text(title)
            for make, make_items in groupby(lambda item: item['make'], items).items():
                for model, model_items in groupby(lambda item: item['model'], make_items).items():
                    with tag('h3'):
                        text(f'{make} / {model}')
                    for item in model_items:
                        with tag('img',
                                src=image_base_path + '/' + item['image_id'] + '.jpg',
                                height=f'{image_height}', width=f'{image_width}'):
                            text()
                        with tag('img',
                                src=score_map_base_path + '/' + item['image_id'] + '.png',
                                height=f'{image_height}', width=f'{image_width}'):
                            text('')
    return doc

#image_base_path: https://storage.cloud.google.com/dev_visual_search/v0/images/
#score_map_base_path: https://storage.cloud.google.com/dev_visual_search/v0/datasets/all-2-top-makes/output/score-maps/model-thumbnail-100-bs-50-ep-1000-sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score map report')
    parser.add_argument('-i', '--image_base_path', type=str, required=True,
            help='base path of the folder where images are expected to be found.')
    parser.add_argument('-sm', '--score_map_base_path', type=str, required=True,
            help='base path of the folder where score maps are expected to be found.')
    parser.add_argument('-wd', '--image_width', type=int, default='128',
            help='image width in  pixels.')
    parser.add_argument('-ht', '--image_height', type=int, default='128',
            help='image height in  pixels.')

    args = parser.parse_args()

    items = ijson.items(sys.stdin, 'item')

    doc = to_html_doc(
            'Score map report',
            items,
            args.image_base_path, args.score_map_base_path, args.image_width, args.image_height)

    print(indent(doc.getvalue()))
