import cv2
import numpy as np
import os
from scipy.misc import imsave
from PIL import Image


root_dir = os.path.dirname(os.path.abspath(__file__))
# img = cv2.imread('/Users/zmalik/image-similarity-fusion/trainingset_tmp/1/150086093.jpg')
# print(img.shape)

img = Image.open('/Users/zmalik/image-similarity-fusion/trainingset_tmp/1/150086093.jpg')
img.convert('RGB')

# img = img[71:464,33:759,:]
img = img.crop((33,71,759,464))
imsave('crop.png', img)
