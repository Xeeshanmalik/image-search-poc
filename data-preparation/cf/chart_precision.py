import os
import sys
import argparse
import ujson
import toolz
from toolz.itertoolz import mapcat, concat, take, groupby, sliding_window, take
from toolz.dicttoolz import valmap, get_in
from toolz.sandbox.core import unzip
from functools import reduce
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def precision_at(dataset, p_threshold):

    def accum(d, item):
        d['count'] += 1
        if item['car_exterior_present'] and get_in(['car_detected_probability'], item, 0.0) >= p_threshold:
            d['true_positive'] += 1
        elif not item['car_exterior_present'] and get_in(['car_detected_probability'], item, 0.0) >= p_threshold:
            d['false_positive'] += 1
        elif not item['car_exterior_present'] and not get_in(['car_detected_probability'], item, 0.0) >= p_threshold:
            d['true_negative'] += 1
        elif item['car_exterior_present'] and not get_in(['car_detected_probability'], item, 0.0) >= p_threshold:
            d['false_negative'] += 1

        return d

    r = reduce(accum, dataset, {
        'count': 0,
        'true_positive': 0,
        'false_positive': 0,
        'true_negative': 0,
        'false_negative': 0})

    precision = r['true_positive'] / (r['true_positive'] + r['false_positive'])

    return precision




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chart the precision of vehicle detection (all.json from CF)')
    parser.add_argument('dataset', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
        help='The dataset to process. Reads from stdin by default.')
    args = parser.parse_args()

    dataset = ujson.load(args.dataset)

    data_p0_6 = map(
            lambda n: (n, precision_at(take(n, dataset), 0.6)),
            range(100, len(dataset) + 1, 100))

    data_p0_7 = map(
            lambda n: (n, precision_at(take(n, dataset), 0.7)),
            range(100, len(dataset) + 1, 100))

    data_p0_8 = map(
            lambda n: (n, precision_at(take(n, dataset), 0.8)),
            range(100, len(dataset) + 1, 100))

    data_p0_85 = map(
            lambda n: (n, precision_at(take(n, dataset), 0.85)),
            range(100, len(dataset) + 1, 100))

    # Data for plotting
    n_p0_6, p_p0_6 = unzip(data_p0_6)
    n_p0_7, p_p0_7 = unzip(data_p0_7)
    n_p0_8, p_p0_8 = unzip(data_p0_8)
    n_p0_85, p_p0_85 = unzip(data_p0_85)

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.plot(list(n_p0_6), list(p_p0_6), label='p >= 0.6')
    ax.plot(list(n_p0_7), list(p_p0_7), label='p >= 0.7')
    ax.plot(list(n_p0_8), list(p_p0_8), label='p >= 0.8')
    ax.plot(list(n_p0_85), list(p_p0_85), label='p >= 0.85')

    ax.set(xlabel='population size', ylabel='precision', title='precision / population size')
    ax.grid()

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    path = 'tmp/chart-precision.png'
    fig.savefig(path)
    print(f'chart writen to: {path}')
    plt.show()

