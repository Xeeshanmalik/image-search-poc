try:
    import numpy
    import sys
    import matplotlib
    import tensorflow as tf
    import os
    from keras import backend as K
    from preprocessing.preprocessor import Preprocessor
    import matplotlib.pyplot as plt
    from model_1.model_1 import Model
    from train.train import Train
    from dimensionality_reduction.reduction import Reduction
    from similarity.similarity import Similarity
    from configuration.config import Configuration
    from keras.models import load_model
    from yad2k.models.keras_yolo import yolo_eval,yolo_head
    import os
    from glob import glob
    import time
    import sys
    import shutil
    import numpy as np
    from matplotlib.pyplot import imshow
    from random import random
    import sys
    import colorsys
    import h5py
    import scipy.io

    sys.dont_write_bytecode = True

    def __run_training__():

        cfg = Configuration()

        # These variable would be parametrized

        GPU = True

        if GPU != True:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Input Path

        root_dir = os.path.dirname(os.path.abspath(__file__))

        image_path = cfg.image_path

        json_path = os.path.join(root_dir, cfg.input_filename)

        trainingset = os.path.join(root_dir, 'trainingset')

        Preprocessor.__generate_kijiji_set__(root_dir, image_path, json_path, trainingset, 'make')

        # --------------------------------------------------------------------------------------------------------------

        image_path = os.path.join(root_dir, 'trainingset')

        data_path = glob(image_path + "/*")

        # Image Segmentation Parameters

        model_path = os.path.expanduser(cfg.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        anchors_path = os.path.expanduser(cfg.anchors_path)
        classes_path = os.path.expanduser(cfg.classes_path)
        test_path = os.path.expanduser(cfg.test_path)
        output_path = os.path.expanduser(cfg.segmented_output_path)
        json_path = os.path.expanduser(cfg.json_output)

        if not os.path.exists(output_path):
            print('Creating output path {}'.format(output_path))
            os.mkdir(output_path)

        sess = K.get_session()

        class_names = Preprocessor.__return_class_names__(classes_path)

        anchors = Preprocessor.__return_anchors__(anchors_path)

        yolo_model = load_model(model_path)

        # Verify model, anchors, and classes are compatible

        num_classes = len(class_names)

        num_anchors = len(anchors)

        info = 'Mismatch between model and given anchor and class sizes. ' \
               'Specify matching anchors and classes with --anchors_path and --classes_path flags.'

        model_output_channels = yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), info
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Check if model is fully convolutional, assuming channel last order.

        model_image_size = yolo_model.layers[0].input_shape[1:3]

        is_fixed_size = model_image_size != (None, None)

        # Generate Colors for drawing bounding boxes

        hsv_tuples, colors = Preprocessor.__generate_colors_for_bounding_boxes__(class_names)

        yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=cfg.score_threshold,
        iou_threshold=cfg.iou_threshold)

        # Load Images from the root folder

        input_images_model_1, all_images, data_path, data_path_with_image_name = Preprocessor.__load_image_data_thumbnails__(data_path,
                                                                                 cfg.compressed_image_height,
                                                                                 cfg.compressed_image_width,
                                                                                 cfg.compressed_channel,
                                                                                 cfg.number_of_categories,
                                                                                 cfg.number_of_images_per_category,
                                                                                 root_dir,
                                                                                 is_fixed_size,
                                                                                 model_image_size,
                                                                                 sess,
                                                                                 yolo_model,
                                                                                 input_image_shape,
                                                                                 boxes,
                                                                                 scores,
                                                                                 classes,
                                                                                 cfg.font_path,
                                                                                 class_names,
                                                                                 colors,
                                                                                 output_path,
                                                                                 json_path,
                                                                                 test_path,
                                                                                 True,  # Segmentation Flag
                                                                                 False, # Edge-detection Flag
                                                                                 True,  # Extract object Flag
                                                                                 False) # Gray Scale Flag

        input_images_model_2, all_images, data_path, data_path_with_image_name = Preprocessor.__load_image_data_thumbnails__(data_path,
                                                                                 cfg.compressed_image_height,
                                                                                 cfg.compressed_image_width,
                                                                                 cfg.compressed_channel,
                                                                                 cfg.number_of_categories,
                                                                                 cfg.number_of_images_per_category,
                                                                                 root_dir,
                                                                                 is_fixed_size,
                                                                                 model_image_size,
                                                                                 sess,
                                                                                 yolo_model,
                                                                                 input_image_shape,
                                                                                 boxes,
                                                                                 scores,
                                                                                 classes,
                                                                                 cfg.font_path,
                                                                                 class_names,
                                                                                 colors,
                                                                                 output_path,
                                                                                 json_path,
                                                                                 test_path,
                                                                                 False, # Segmentation Flag
                                                                                 True,  # Edge-detection Flag
                                                                                 False, # Extract object Flag
                                                                                 False) # Gray Scale Flag

        input_images_model_3, all_images, data_path, data_path_with_image_name = Preprocessor.__load_image_data_thumbnails__(data_path,
                                                                                 cfg.image_height,
                                                                                 cfg.image_width,
                                                                                 cfg.channel,
                                                                                 cfg.number_of_categories,
                                                                                 cfg.number_of_images_per_category,
                                                                                 root_dir,
                                                                                 is_fixed_size,
                                                                                 model_image_size,
                                                                                 sess,
                                                                                 yolo_model,
                                                                                 input_image_shape,
                                                                                 boxes,
                                                                                 scores,
                                                                                 classes,
                                                                                 cfg.font_path,
                                                                                 class_names,
                                                                                 colors,
                                                                                 output_path,
                                                                                 json_path,
                                                                                 test_path,
                                                                                 False, # Segmentation Flag
                                                                                 False, # Edge-detection Flag
                                                                                 False, # Extract object Flag
                                                                                 False) # Gray Scale Flag

        input_shape = [cfg.compressed_image_height, cfg.compressed_image_width, cfg.compressed_channel]

        input_shape_3 = [cfg.image_height, cfg.image_width, cfg.channel]

        # Model Save Paths

        model_1_save_path = os.path.join(root_dir + cfg.model_1_save)
        model_2_save_path = os.path.join(root_dir + cfg.model_2_save)
        model_3_save_path = os.path.join(root_dir + cfg.model_3_save)

        Preprocessor.__create_output_directories__(model_1_save_path)
        Preprocessor.__create_output_directories__(model_2_save_path)
        Preprocessor.__create_output_directories__(model_3_save_path)

        # Instantiating the training class

        train = Train(input_images_model_1,
                      input_images_model_2,
                      input_images_model_3,
                      input_shape,
                      input_shape_3,
                      cfg.batch_size,
                      cfg.epochs,
                      model_1_save_path,
                      model_2_save_path,
                      model_3_save_path)

        # Output Path

        output_path_model_1 = os.path.join(root_dir + cfg.output_model_1)
        output_path_model_2 = os.path.join(root_dir + cfg.output_model_2)
        output_path_model_3 = os.path.join(root_dir + cfg.output_model_3)

        Preprocessor.__create_output_directories__(output_path_model_1)
        Preprocessor.__create_output_directories__(output_path_model_2)
        Preprocessor.__create_output_directories__(output_path_model_3)

        # FCN Model

        model_1 = train.__train_model_1__()

        # VGG Model

        model_2 = train.__train_model_2__()

        # Inception-v3

        model_3 = train.__train_model_3__()

        features_from_model_1 = Preprocessor.__get_score_model__(model_1, input_images_model_1, output_path_model_1)
        features_from_model_2 = Preprocessor.__get_score_model__(model_2, input_images_model_2, output_path_model_2)
        features_from_model_3 = Preprocessor.__get_score_model__(model_3, input_images_model_3, output_path_model_3)

        print("Output FeatureMap For Model 1 \n")
        print(features_from_model_1.shape)
        print("\n")

        print("Output FeatureMap For Model 2 \n")
        print(features_from_model_2.shape)
        print("\n")

        print("Output FeatureMap For Model 3 \n")
        print(features_from_model_3.shape)
        print("\n")

    def main(_):

      __run_training__()

    if __name__ == '__main__':
        tf.app.run()


except ImportError as E:
    raise E