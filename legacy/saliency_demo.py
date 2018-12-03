import pyimgsaliency as psal
import cv2

# path to the image

filename = '/Users/zmalik/image-similarity-fusion/bird.jpg'

# get the saliency maps using the 3 implemented methods

rbd = psal.get_saliency_rbd(filename).astype('uint8')

ft = psal.get_saliency_ft(filename).astype('uint8')

mbd = psal.get_saliency_mbd(filename).astype('uint8')

# often, it is desirable to have a binary saliency map

binary_sal = psal.binarise_saliency_map(mbd,method='adaptive')

img = cv2.imread(filename)

cv2.imshow('img',img)
cv2.imshow('rbd',rbd)
cv2.imshow('ft',ft)
cv2.imshow('mbd',mbd)

# openCV cannot display numpy type 0, so convert to uint8 and scale

cv2.imshow('binary',255 * binary_sal.astype('uint8'))


cv2.waitKey(0)


# ------------------generator to compile training data of stanford dataset--------------------------------------

        # image_path = os.path.join(root_dir, cfg.data)
        #
        # data_path = glob(image_path + "/*")
        #
        # # Reading the matlab file for class info
        # annos_path = str(image_path).split("/")
        # annos_path = annos_path[:-1]
        # annos_path = "/".join(annos_path)
        # path = annos_path + '/' + 'cars_test_annos.mat'
        # mat = scipy.io.loadmat(path)
        # class_info = mat['annotations']['class']
        # class_info = np.asarray(class_info[0])

        # Preprocessor.__generate_set__(root_dir, data_path, class_info)

# --------------------------------------------------------------------------------------------------------------


# ------------------generator to compile training data of stanford dataset--------------------------------------

        # image_path = os.path.join(root_dir, cfg.data)
        #
        # data_path = glob(image_path + "/*")
        #
        # # Reading the matlab file for class info
        #
        # annos_path = str(image_path).split("/")
        # annos_path = annos_path[:-1]
        # annos_path = "/".join(annos_path)
        # path = annos_path + '/' + 'cars_train_annos.mat'
        # mat = scipy.io.loadmat(path)
        # class_info = mat['annotations']['class']
        # class_info = np.asarray(class_info[0])
        #
        # Preprocessor.__generate_set__(root_dir, data_path, class_info)

# --------------------------------------------------------------------------------------------------------------

# ------------------generator to compile training data of kijiji dataset----------------------------------------
