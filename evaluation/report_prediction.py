import ujson
import sys
import os.path
import argparse

# from pprint import pprint
from functools import partial
from toolz.dicttoolz import valmap, update_in, get_in
from toolz.itertoolz import groupby, take, sliding_window, cons, first, concat
from toolz.functoolz import pipe, compose
from yattag import Doc, indent
from collections import OrderedDict


def ordered_dict_sorted_by_value(d):
    return OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))


def to_master_page(title, rows):
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        with tag('head'):
            doc.asis('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">')
            doc.asis('<link rel="stylesheet" href="https://cdn.datatables.net/1.10.16/css/jquery.dataTables.min.css">')
            doc.asis('<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>')
            doc.asis('<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>')
            doc.asis('<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>')
            doc.asis('<script src="https://cdn.datatables.net/1.10.16/js/jquery.dataTables.min.js"></script>')
            doc.asis('<script src="https://cdn.datatables.net/1.10.16/js/dataTables.bootstrap.min.js"></script>')
            with tag('style'):
                doc.asis(f'th {{ font-size: 70%; }}')
                doc.asis(f'td {{ font-size: 70%; }}')
        with tag('body'):
            doc.asis("""
                     <script>
                    $(document).ready(function() {
                        $('#accuracy-table').DataTable({
                          'pageLength': 50
                        });
                    });
                     </script>
                     """)
            with tag('h1'):
                text(title)
            with tag('table', klass='table table-striped', id='accuracy-table'):
                with tag('thead'):
                    with tag('tr'):
                        with tag('th'):
                            text('Make')
                        with tag('th'):
                            text('Model')
                        for feature in ['Make', 'Model', 'Year', 'Color', 'Body']:
                            with tag('th'):
                                text(f'{feature}')
                            for n in ['1', '3', '5', '10']:
                                with tag('th'):
                                    text(f'{n}')
                with tag('tbody'):
                    for row in rows:
                        with tag('tr'):
                            with tag('td'):
                                text(row['make'])
                            with tag('td'):
                                line('a', row['model'], href=row['url'],  target='_blank')
                            for feature in ['make', 'model', 'year', 'color', 'body']:
                                with tag('td'):
                                    text() # Feature name column
                                for value in row['accuracy'][feature]:
                                    with tag('td'):
                                        text(f'{value:.3}')
    return doc


def to_page(items, links, args):
    title, evaluation_id, image_base_path, image_width, image_height = map(
        args.get,
        ['title', 'evaluation_id', 'image_base_path', 'image_width', 'image_height'])

    link_prev, link_master, link_next = map(
        links.get,
        ['prev', 'parent', 'next'])

    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        with tag('head'):
            doc.asis('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">')
            doc.asis('<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>')
            doc.asis('<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>')
            doc.asis('<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>')
            with tag('style'):
                doc.asis(f'img {{max-width:{image_width}px;max-height:{image_height}px;width:auto;height:auto;}}')
                doc.asis(f'.correct {{border: green 4px solid; vertical-align: top}}')
                doc.asis(f'.incorrect {{border: red 4px solid; vertical-align: top}}')
        with tag('body'):
            doc.asis("""
                     <script>
                     $(function () {
                       $('[data-toggle="popover"]').popover();
                     })
                     </script>
                     """)
            with tag('h1'):
                text(title)
            for make, make_items in groupby(lambda item: item['make'], items).items():
                for model, model_items in groupby(lambda item: item['model'], make_items).items():
                    with tag('h3'):
                        with tag('span'):
                            if link_prev is not None:
                                line('a', 'Previous', href=f'{link_prev}')
                                doc.asis('&nbsp;')
                            if link_next is not None:
                                line('a', 'Next', href=f'{link_next}')
                    for item in model_items:
                        with tag('a',
                                 ('data-toggle', 'popover'),
                                 ('data-trigger', 'hover'),
                                 ('data-content', f'Body: {item["body"]}, Color: {item["color"]}, Year: {item["year"]}'),
                                 title=f'{make} / {model}',
                                 href=f'https://storage.cloud.google.com/dev_visual_search/evaluations/output/by-id/{evaluation_id}/trainset-{make}-{model}.html',
                                 target='_blank'):
                            with tag('img',
                                     src=image_base_path + '/' + item['image_id'] + '.jpg',
                                     height=f'{image_height}', width=f'{image_width}'):
                                text()

                        predictions = item['predictions']

                        for p in predictions:
                            p_image_id = p['image_id']
                            correct = (item['make'], item['model']) == (p['make'], p['model'])
                            with tag('img',
                                     ('data-toggle', 'popover'),
                                     ('data-trigger', 'hover'),
                                     ('data-content', f'Body: {p["body"]}, Color: {p["color"]}, Year: {p["year"]}'),
                                     title=f'{p["make"]} / {p["model"]}',
                                     klass='correct' if correct else 'incorrect',
                                     src=image_base_path + '/' + p_image_id + '.jpg',
                                     height=f'{image_height}', width=f'{image_width}'):
                                text('')

                        histogram = pipe(predictions,
                                         partial(groupby, lambda x: (x['make'], x['model'])),
                                         partial(valmap, len),
                                         ordered_dict_sorted_by_value)

                        for k, v in histogram.items():
                            correct = (item['make'], item['model']) == k
                            with tag('span',
                                     klass='correct' if correct else 'incorrect'):
                                text(v)

                        with tag('br'):
                            text()
                    with tag('h3'):
                        with tag('span'):
                            if link_prev is not None:
                                line('a', 'Previous', href=f'{link_prev}')
                                doc.asis('&nbsp;')
                            if link_next is not None:
                                line('a', 'Next', href=f'{link_next}')

    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prediction report',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('output_file', nargs='?', type=argparse.FileType('rw'),
                        default=sys.stdout,
                        help='The output file, the master predictions page will be writen to it')
    parser.add_argument('-i', '--image_base_path', type=str,
                        default='https://storage.cloud.google.com/dev_visual_search/images/cropped/v3',
                        help='base path of the folder where images are expected to be found.')
    parser.add_argument('-wd', '--image_width', type=int, default=128, help='image width in  pixels.')
    parser.add_argument('-ht', '--image_height', type=int, default=128, help='image height in  pixels.')
    parser.add_argument('-n', '--predictions_limit', type=int, default=10,
                        help='Number of predictions per image to render.')
    parser.add_argument('-id', '--evaluation_id', type=str,
                        help='The ID of the evaluation this report is for.')
    parser.add_argument('--base_output_path', type=str, default='tmp',
                        help='When --split is True, this is the path of the folder report parts will be writen to.')
    parser.add_argument('--model_evaluation_file', type=argparse.FileType('r'),
                        help='Path to the Make/Model evaluation file.')

    args = parser.parse_args()

    dataset = list(map(
        lambda elem: update_in(elem, ['predictions'], compose(list, partial(take, args.predictions_limit))),
        ujson.load(args.dataset_file)))

    sections = groupby(lambda x: tuple(map(x.get, ['make', 'model'])), dataset).items()

    evaluation_base_url = f'https://storage.cloud.google.com/dev_visual_search/evaluations/output/by-id/{args.evaluation_id}'

    def link_to_page(key):
        if key is None:
            return None
        make, model = key
        return f'{evaluation_base_url}/prediction-{make}-{model}.html'

    for prev, current, next in sliding_window(3, cons(None, concat([sections, [None]]))):
        key, section = current
        make, model = key

        prev_key, _ = prev if prev is not None else (None, None)
        next_key, _ = next if next is not None else (None, None)

        page = to_page(section,
                       {'prev': link_to_page(prev_key),
                        'parent': '',
                        'next': link_to_page(next_key)},
                       {'title': f'Prediction report for {make} / {model}',
                        'evaluation_id': args.evaluation_id,
                        'image_base_path': args.image_base_path,
                        'image_width': args.image_width,
                        'image_height': args.image_height})

        with open(os.path.join(args.base_output_path, f'prediction-{make}-{model}.html'), 'w') as outfile:
            print(indent(page.getvalue()), file=outfile)

    accuracy = ujson.load(args.model_evaluation_file)

    keys = map(first, sections)

    def get_accuracy(key):
        make, model = key
        features = ['make', 'model', 'year', 'color', 'body']
        return {
            feature:  [get_in([f'{make}/{model}', f'{n}', 'accuracy', feature], accuracy) for n in ["1", "3", "5", "10"]]
            for feature in features}

    def to_row(key):
        make, model = key
        return {'make': make,
                'model': model,
                'accuracy': get_accuracy(key),
                'url': link_to_page(key)}

    master_page = to_master_page(
        f'Master Prediction Report for Evaluation #{args.evaluation_id}',
        map(to_row, keys))
    print(indent(master_page.getvalue()), file=args.output_file)
