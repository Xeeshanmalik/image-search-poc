# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import os
# from scipy.misc import imsave
#
# thresh = 100, 200
# root_dir = os.path.dirname(os.path.abspath(__file__))
# img = cv2.imread(root_dir + '/edge_detection_output/Motorbikes/image_0031.jpg')
#
# img = cv2.convertScaleAbs(img)
#
# ret, thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
import cv2
import numpy as np
import os


def thresh_callback(thresh):

    edges = cv2.Canny(blur, thresh, thresh*2)
    drawing = np.zeros(img.shape, np.uint8)     # Image to draw the contours
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    bx, by, bw, bh = cv2.boundingRect(cnt)
    cv2.rectangle(drawing, (bx, by), (bx+bw, by+bh), (255, 0, 0), 3) # draw rectangle in blue color)
    cv2.imshow('output', drawing)
    cv2.imshow('input', img)

root_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(root_dir + '/edge_detection_output/Motorbikes/image_0031.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.namedWindow('input')

thresh = 100
max_thresh = 255

cv2.createTrackbar('canny thresh:', 'input', thresh, max_thresh, thresh_callback)

thresh_callback(0)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
