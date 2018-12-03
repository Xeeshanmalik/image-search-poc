from __future__ import division, print_function
import matplotlib.pyplot as plt
import os
import collections
import numpy as np
import operator

OUTPUT_FILE = "./butterflies_rgb.txt"

def get_vector_rgb(img):

    colors = ["r", "g", "b"]
    bins = np.linspace(0, 256, 25)
    means = 0.5 * (bins[1:] + bins[:-1]).astype("uint8")
    words = []
    for i in range(len(colors)):
        px_orig = img[:,:,i].flatten()
        labels = np.searchsorted(bins, px_orig)
        px_reduced = np.choose(labels.ravel(), means,
                               mode="clip").astype("uint8").tolist()
        counter = collections.Counter(px_reduced)
        words.extend([(colors[i] + str(x[0]), x[1]) for x in counter.items()])
    words_sorted = sorted(words, key=operator.itemgetter(1), reverse=True)
    max_freq = words_sorted[0][1]
    words_sorted = [(x[0], 100.0 * x[1]/max_freq) for x in words_sorted]
    return words_sorted

fout = open(OUTPUT_FILE, 'a')
img = plt.imread('/Users/zmalik/image-similarity-fusion/trainingset_tmp/1/150085495.jpg')
words = get_vector_rgb(img)
words_str = " ".join(w[0] + "|" + ("%.3f" % (w[1])) for w in words)
fout.write("%s\t%s\n" % ('150085495', words_str))

