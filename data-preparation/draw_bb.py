import ijson
import sys
import argparse
import toolz
from toolz.itertoolz import mapcat, concat, take, groupby
from toolz.dicttoolz import valmap
from PIL import Image, ImageDraw, ImageFont
import os

font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)

def draw_bounding_box(items, image_base_path, image_output_path):
    def to_image_path(name):
        return os.path.join(image_base_path, name)


    for item in items:
        name = item['name']
        image_path = to_image_path(name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        if item['label'] == 'car' or item['label'] == 'truck':
            draw = ImageDraw.Draw(image)
            draw.rectangle(((item['left'], item['top']), (item['right'], item['bottom'])), outline='red')
            draw.rectangle(((item['left']-1, item['top']-1), (item['right']+1, item['bottom']+1)), outline='red')
            draw.rectangle(((item['left']-2, item['top']-2), (item['right']+2, item['bottom']+2)), outline='red')
            draw.text((item['left'], item['top']), "%.2f" % item['probability'], font=font, fill='red')
            del draw
        image.save(os.path.join(image_output_path, name), 'jpeg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw bounding box.')
    parser.add_argument('-i', '--image_base_path', type=str, default='.',
                        help='base path of the folder where images are expected to be found.')
    parser.add_argument('-o', '--image_output_path', type=str,
                        help='base path of the folder where output images will be writen to.')
    args = parser.parse_args()

    items = ijson.items(sys.stdin, 'bound.item')
    draw_bounding_box(items, args.image_base_path, args.image_output_path)
