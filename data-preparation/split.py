import json
import sys
import argparse
import numpy as np

def split(dataset, train_set_path, dev_set, test_set):
    dev_set_path, dev_set_size = dev_set
    test_set_path, test_set_size = test_set

    # shuffle
    arr = np.array(dataset)
    shuffled_indices = np.random.permutation(np.arange(len(dataset)))
    arr_shuffled = arr[shuffled_indices]
    dataset = arr_shuffled.tolist()

    dataset_len = len(dataset)
    dev_set_len = int(dataset_len * dev_set_size)
    test_set_len = int(dataset_len * test_set_size)

    dev_set = dataset[:dev_set_len]
    test_set = dataset[dev_set_len:dev_set_len + test_set_len]
    train_set = dataset[dev_set_len + test_set_len:]

    with open(dev_set_path, mode='w') as f:
        json.dump(dev_set, f, indent=4, separators=(',', ': '))

    with open(test_set_path, mode='w') as f:
        json.dump(test_set, f, indent=4, separators=(',', ': '))

    with open(train_set_path, mode='w') as f:
        json.dump(train_set, f, indent=4, separators=(',', ': '))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split dataset into train / test / dev datasets.')
    parser.add_argument('--train_set_path', type=str, required=True,
            help='path to the file the TRAIN set will be writen to.')

    parser.add_argument('--dev_set_path', type=str, required=True,
            help='path to the file the DEV set will be writen to.')
    parser.add_argument('--dev_set_size', type=float, default=0.1,
            help='size of DEV set, represented as a percent of the total scaled to 0..1')

    parser.add_argument('--test_set_path', type=str, required=True,
            help='path to the file the TEST set will be writen to.')
    parser.add_argument('--test_set_size', type=float, default=0.1,
            help='size of TEST set, represented as a percent of the total scaled to 0..1')

    args = parser.parse_args()

    split(json.load(sys.stdin), args.train_set_path,
            (args.dev_set_path, args.dev_set_size),
            (args.test_set_path, args.test_set_size))
