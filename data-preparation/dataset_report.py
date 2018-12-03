import ijson
import sys
import argparse
from toolz.itertoolz import groupby
from yattag import Doc, indent


# render as hint in html
def to_html_doc(title, items, image_base_path, image_width, image_height):
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
                    for seller, seller_items in groupby(lambda item: item['seller'], model_items).items():
                        with tag('h3'):
                            text(make + ' - ' + model + ' - ' + seller + ' (' +str(len(seller_items))+ ')')
                        for item in seller_items:
                            with tag('img',
                                    src=image_base_path + '/' + item['image_id'] + '.jpg?authuser=1',
                                    height=f'{image_height}', width=f'{image_width}'):
                                text('')
    return doc


filters = {
        'all': lambda x: True,
        'true_positive': lambda x: x['car_exterior_present'] and x['car_detected'],
        'false_positive': lambda x: not x['car_exterior_present'] and x['car_detected'],
        'true_negative': lambda x: not x['car_exterior_present'] and not x['car_detected'],
        'false_negative': lambda x: x['car_exterior_present'] and not x['car_detected'],
        'car_detected': lambda x: x['car_detected']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image data html generator.')
    parser.add_argument('-i', '--image_base_path', type=str,
                        default='https://storage.cloud.google.com/dev_visual_search/images/cropped/v3',
                        help='base path of the folder where images are expected to be found.')
    parser.add_argument('-wd', '--image_width', type=int, default='128',
                        help='image width in  pixels.')
    parser.add_argument('-ht', '--image_height', type=int, default='128',
                        help='image height in  pixels.')
    parser.add_argument('-f', '--filter', type=str, default='all',
                        choices=['all', 'car_detected', 'true_positive', 'false_positive',
                                 'true_negative', 'false_negative'],
                        help='Filter items before generating report.')

    args = parser.parse_args()

    items = ijson.items(sys.stdin, 'item')

    doc = to_html_doc(
            args.filter,
            filter(filters[args.filter], items),
            args.image_base_path, args.image_width, args.image_height)

    print(indent(doc.getvalue()))
