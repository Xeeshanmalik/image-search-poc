import ijson
import sys
import argparse
import os
from functools import partial
import multiprocessing
import numpy as np
import cv2


def grab_cut(image_input_path, image_output_path):
    img = cv2.imread(image_input_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # step 1
    rect = (0, 0, img.shape[:2][0] - 1, img.shape[:2][1] - 1)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

    # step 2
    mask[9:107, 9:107] = cv2.GC_FGD
    mask[0:15, 0:15] = cv2.GC_BGD
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD), 0, 1).astype('uint8')
    np.putmask(img[:,:,0], mask2 == 0, 255)
    np.putmask(img[:,:,1], mask2 == 0, 255)
    np.putmask(img[:,:,2], mask2 == 0, 255)
    cv2.imwrite(image_output_path, img)

def contours(image_input_path, image_output_path):
    img = cv2.imread(image_input_path)
    blur = cv2.blur(img, (4, 4))
    edged = cv2.Canny(blur, 10, 100)

    #applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    #finding_contours
    (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    mask = np.zeros_like(img)
#    for c in cnts:
#        peri = cv2.arcLength(c, True)
#        if peri < 100:
#            continue
#        approx = cv2.approxPolyDP(c, 0.001 * peri, True)
#        cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)

    cv2.drawContours(mask, cnts, -1, (255, 255, 255), -1)

    np.putmask(img[:,:,:], mask == 0, 255)
    cv2.imwrite(image_output_path, img)


def extract_vehicle(image_base_path, algo, image_output_path, item):

    def to_image_path(base_path, image_id):
        return os.path.join(base_path, image_id) + '.jpg'

    image_id = item['image_id']

    print(f'Extracting vehicle for {image_id} ...', file=sys.stderr)

    image_input_path = to_image_path(image_base_path, image_id)
    image_output_path = to_image_path(image_output_path, image_id)

    if not os.path.isfile(image_input_path):
        print(f'Missing image {image_input_path}!', file=sys.stderr)
        return

    algo(image_input_path, image_output_path)


    return image_id

def extract_vehicles(dataset,image_base_path, image_output_path, algo, number_of_processes):
    with multiprocessing.Pool(processes=number_of_processes) as pool:
        return pool.map(partial(extract_vehicle, image_base_path, algo, image_output_path), dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='crop image to bounding box.')
    parser.add_argument('-i', '--image_base_path', type=str, default='.',
                        help='base path of the folder where images are expected to be found.')
    parser.add_argument('-o', '--image_output_path', type=str,
                        help='base path of the folder where output images will be writen to.')
    parser.add_argument('-p', '--number_of_processes', type=int, default=multiprocessing.cpu_count(),
                        help='number of processed that will be used to crop the images concurrently.')
    parser.add_argument('-a', '--algorithm', type=str, choices=['contours','grab_cut'], default='grab_cut',
                        help='the name of the algorithm used to extract the foreground object.')
    args = parser.parse_args()

    dataset = ijson.items(sys.stdin, 'item')
    extract_vehicles(
            dataset,
            args.image_base_path, args.image_output_path, globals()[args.algorithm],
            args.number_of_processes)
