import ijson
import sys
import argparse
import csv


def write_to_csv(in_stream, out_stream):
    items = ijson.items(in_stream, 'item')
    w = csv.DictWriter(out_stream, fieldnames=['id', 'image_url', 'image_id'])
    w.writeheader()
    for item in map(lambda item: {'image_url': item['image'], 'id': item['id'], 'image_id': item['image_id']}, items):
        w.writerow(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrowdFlower data generator.')
    args = parser.parse_args()

    write_to_csv(sys.stdin, sys.stdout)

