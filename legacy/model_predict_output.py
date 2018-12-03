try:

    import os
    import tensorflow as tf
    import matplotlib
    import sys
    from preprocessing.preprocessor import Preprocessor
    from glob import glob
    from keras import backend as K
    from model_1.model_1 import Model_1
    from model_2.model_2 import Model_2
    from model_3.model_3 import Model_3
    import shutil
    import numpy as np
    from similarity.similarity import Similarity
    from dimensionality_reduction.reduction import Reduction
    from random import random
    from preprocessing.preprocessor import Preprocessor
    from matplotlib.pyplot import imshow
    import time
    from yad2k.models.keras_yolo import yolo_eval,yolo_head
    from matplotlib import pyplot as plt
    from scipy.misc import imsave
    from dimensionality_reduction.reduction import Reduction
    from configuration.config import Configuration
    import sys
    from keras.models import load_model
    import scipy.io
    from random import random, randint
    import csv
    sys.dont_write_bytecode = True

    def __predict_output__():

        plt.interactive(False)
        cfg = Configuration()
        GPU = True

        if GPU != True:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Input Path

        root_dir = os.path.dirname(os.path.abspath(__file__))

        image_path = cfg.image_path

        json_path = os.path.join(root_dir, cfg.input_filename)

        testingset = os.path.join(root_dir,'testingset')

        Preprocessor.__generate_kijiji_set__(root_dir, image_path, json_path, testingset, 'model')

        # ------------------generator to compile training data of kijiji dataset----------------------------------------

        image_path = os.path.join(root_dir, 'testingset')

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
                                                                                 True, # Segmentation Flag
                                                                                 False, # Edge-detection Flag
                                                                                 True, # Extract object Flag
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
                                                                                 False,
                                                                                 True,
                                                                                 False,
                                                                                 False)

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
                                                                                 False,
                                                                                 False,
                                                                                 False,
                                                                                 False)

        input_shape = [cfg.compressed_image_height, cfg.compressed_image_width, cfg.compressed_channel]

        input_shape_3 = [cfg.image_height, cfg.image_width, cfg.channel]

        # load (pre-trained) weights for model_1

        print('-'*30)
        print('Loading model weights...\n')
        weight_folder = cfg.model_1_save # the path where the model weights are stored
        weight_file = 'model_1.h5'
        model_1 = Preprocessor.__load_model_weights__(weight_folder, weight_file, input_shape, input_shape_3, "Model_1")

        # load (pre-trained) weights for model_2

        print('-'*30)
        print('Loading model weights...\n')
        weight_folder = cfg.model_2_save # the path where the model weights are stored
        weight_file = 'model_2.h5'
        model_2 = Preprocessor.__load_model_weights__(weight_folder, weight_file, input_shape, input_shape_3, "Model_2")

        # load (pre-trained) weights for model_2

        print('-'*30)
        print('Loading model weights...\n')
        weight_folder = cfg.model_3_save  # the path where the model weights are stored
        weight_file = 'model_3.h5'
        model_3 = Preprocessor.__load_model_weights__(weight_folder, weight_file, input_shape, input_shape_3, "Model_3")
        print(root_dir)
        print(os.path.join(root_dir, cfg.output_model_1))

        output_path_model_1 = os.path.join(root_dir + cfg.output_model_1)
        output_path_model_2 = os.path.join(root_dir + cfg.output_model_2)
        output_path_model_3 = os.path.join(root_dir + cfg.output_model_3)

        Preprocessor.__create_output_directories__(output_path_model_1)
        Preprocessor.__create_output_directories__(output_path_model_2)
        Preprocessor.__create_output_directories__(output_path_model_3)

        features_from_model_1 = Preprocessor.__get_score_model__(model_1, input_images_model_1, output_path_model_1)
        features_from_model_2 = Preprocessor.__get_score_model__(model_2, input_images_model_2, output_path_model_2)
        features_from_model_3 = Preprocessor.__get_score_model__(model_3, input_images_model_3, output_path_model_3)

        features_from_model_1 = Preprocessor.__flatten_img_data__(features_from_model_1)
        features_from_model_2 = Preprocessor.__flatten_img_data__(features_from_model_2)
        features_from_model_3 = Preprocessor.__flatten_img_data__(features_from_model_3)

        fused_features = np.concatenate([features_from_model_1, features_from_model_2, features_from_model_3], axis=1)

        fused_features = [Preprocessor.__binarize__(features) for features in fused_features]

        counter_for_predictions = 0

        sub_average_precision_make, sub_average_precision_color = [], []
        sub_average_precision_body, sub_average_precision_model = [], []

        cum_average_precision_make, cum_average_precision_color = [], []
        cum_average_precision_body, cum_average_precision_model = [], []

        precision_at_3_5_10_all = ''.join(cfg.precision_counter).split(',')

        while counter_for_predictions <= 2:

            test_image_idx = int(len(input_images_model_1) * random())

            if test_image_idx < len(data_path_with_image_name):

                idx_closest = Preprocessor.__get_closest_images__(test_image_idx,
                                                                  fused_features,
                                                                  cfg.number_of_predictions)
                test_image = Preprocessor.__get_concatenated_images__(data_path_with_image_name,
                                                                      [test_image_idx],
                                                                      cfg.compressed_image_width)
                results_image = Preprocessor.__get_concatenated_images__(data_path_with_image_name,
                                                                         idx_closest,
                                                                         cfg.compressed_image_width)

                source_category = str(data_path_with_image_name[test_image_idx]).split('/')
                similar_image = []
                similar_idx_closest = []

                for counter_for_recommendations in range(0, len(idx_closest)):

                    category = str(data_path_with_image_name[idx_closest[counter_for_recommendations]]).split('/')

                    if str(source_category[-2]).strip() == str(category[-2].strip()):
                        similar_image.append(data_path_with_image_name[idx_closest[counter_for_recommendations]])
                        similar_idx_closest.append(idx_closest[counter_for_recommendations])

                print("Test Image ID:", test_image_idx)
                print("\n")
                print("Closest Images ID:", idx_closest)
                print("\n")
                print("Similar Images ID", similar_idx_closest)
                print("\n")

                precision_per_make, precision_per_color = [], []
                precision_per_body_wise, precision_per_model_wise = [], []
                results_image_recommendations = []

                for i in range(0, len(precision_at_3_5_10_all)):

                    results_image_recommendations = Preprocessor.__get_concatenated_images__(data_path_with_image_name,
                                                                                             similar_idx_closest,
                                                                                            cfg.compressed_image_width)

                    list_of_similar_image_names = Preprocessor.__return_image_names__(data_path_with_image_name,
                                                                                      similar_idx_closest)

                    name_of_test_image = Preprocessor.__return_image_names__(data_path_with_image_name,
                                                                             [test_image_idx])

                    dict_of_attributes_of_similar_images = Preprocessor.__get_attributes_list__(list_of_similar_image_names,
                                                                                                os.path.join(root_dir,
                                                                                                cfg.input_filename))

                    dict_of_attributes_of_test_image = Preprocessor.__get_attributes_list__(name_of_test_image,
                                                                                                os.path.join(root_dir,
                                                                                                cfg.input_filename))

                    similar_make_wise = Preprocessor.__get_similar__(dict_of_attributes_of_test_image,
                                                                      dict_of_attributes_of_similar_images[:int(precision_at_3_5_10_all[i])],
                                                                     'make')

                    similar_color_wise = Preprocessor.__get_similar__(dict_of_attributes_of_test_image,
                                                                      dict_of_attributes_of_similar_images[:int(precision_at_3_5_10_all[i])],
                                                                      'color')

                    similar_body_wise = Preprocessor.__get_similar__(dict_of_attributes_of_test_image,
                                                                      dict_of_attributes_of_similar_images[:int(precision_at_3_5_10_all[i])],
                                                                     'body')

                    similar_model_wise = Preprocessor.__get_similar__(dict_of_attributes_of_test_image,
                                                                      dict_of_attributes_of_similar_images[:int(precision_at_3_5_10_all[i])],
                                                                      'model')

                    precision_per_make.append(float(float(len(similar_make_wise))/int(precision_at_3_5_10_all[i])))
                    precision_per_color.append(float(float(len(similar_color_wise))/int(precision_at_3_5_10_all[i])))
                    precision_per_body_wise.append(float(float(len(similar_body_wise))/int(precision_at_3_5_10_all[i])))
                    precision_per_model_wise.append(float(float(len(similar_model_wise))/int(precision_at_3_5_10_all[i])))

                sub_average_precision_make.append(precision_per_make)
                sub_average_precision_color.append(precision_per_color)
                sub_average_precision_body.append(precision_per_body_wise)
                sub_average_precision_model.append(precision_per_model_wise)

                imsave('test.png', test_image)
                imsave('recommendations.png', results_image_recommendations)
                imsave('total_results.png', results_image)
                counter_for_predictions += 1
                time.sleep(1)

            else:

                print("Index is out of bound")

            cum_average_precision_make.append(map(Preprocessor.__mean__, zip(*sub_average_precision_make)))
            cum_average_precision_color.append(map(Preprocessor.__mean__, zip(*sub_average_precision_color)))
            cum_average_precision_body.append(map(Preprocessor.__mean__, zip(*sub_average_precision_body)))
            cum_average_precision_model.append(map(Preprocessor.__mean__, zip(*sub_average_precision_model)))

        print("\n \n \n")
        print("-----------------------------------------------------------------------------------")
        print("Average Precision Make-Wise", precision_at_3_5_10_all, map(Preprocessor.__mean__, zip(*cum_average_precision_make)))
        print("Average Precision Color-Wise", precision_at_3_5_10_all, map(Preprocessor.__mean__, zip(*cum_average_precision_color)))
        print("Average Precision Body-Wise", precision_at_3_5_10_all, map(Preprocessor.__mean__, zip(*cum_average_precision_body)))
        print("Average Precision Model-Wise", precision_at_3_5_10_all, map(Preprocessor.__mean__, zip(*cum_average_precision_model)))

        writer = csv.writer(open(os.path.join(root_dir, 'results.csv'), 'w'))

        writer.writerow(["Make-Wise: Precision at 3", "Make-Wise: Precision at 5", "Make-Wise: Precision at 10"])
        for row in zip(*cum_average_precision_make):
            writer.writerow(row)

        writer.writerow('\n')

        writer.writerow(["Color-Wise: Precision at 3", "Color-Wise: Precision at 5", "Color-Wise: Precision at 10"])

        for row in zip(*cum_average_precision_color):
            writer.writerow(row)

        writer.writerow('\n')

        writer.writerow(["Body-Wise: Precision at 3", "Body-Wise: Precision at 5", "Body-Wise: Precision at 10"])

        for row in zip(*cum_average_precision_body):
            writer.writerow(row)

        writer.writerow('\n')

        writer.writerow(["Model-Wise: Precision at 3","Model-Wise: Precision at 5", "Model-Wise: Precision at 10"])

        for row in zip(*cum_average_precision_model):
            writer.writerow(row)

        writer.writerow('\n')

    def main(_):

      __predict_output__()

    if __name__ == '__main__':
        tf.app.run()

except ImportError as E:

    raise E