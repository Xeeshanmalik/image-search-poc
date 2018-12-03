try:

    import sys
    import tensorflow as tf
    import numpy as np
    from glob import glob
    from itertools import islice
    import os, random, colorsys
    from matplotlib import pyplot as plt
    from keras.preprocessing import image
    from keras import backend as K
    import time
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial import distance
    import os
    from scipy import sum, average
    from scipy.linalg import norm
    import cv2, io
    import shutil
    from unidecode import unidecode
    import json
    from scipy.misc import imsave
    from model_1.model_1 import Model_1
    from model_2.model_2 import Model_2
    from model_3.model_3 import Model_3
    from segmentation.segmentation import Segmentation
    import pyimgsaliency as psal
    from shutil import copyfile
    from collections import Counter
    import ijson

    class Preprocessor:

        @staticmethod
        def __generate_set__(root_dir, data_path, class_info):

            training_set_path = os.path.join(root_dir, "trainingset")

            for j in range(0, 196):
                for i in range(0, len(class_info)):
                    if class_info[i][0][0] == j:
                        path = str(data_path[i]).split("/")
                        filename = path[-1:]
                        output_path_model = os.path.join(training_set_path, str(j))
                        if not os.path.exists(output_path_model):
                            os.makedirs(output_path_model)
                        copyfile(data_path[i], os.path.join(output_path_model, ''.join(filename)))


        @staticmethod
        def __generate_kijiji_set__(root_dir, image_path, json_path, destination_path, filter_feature_type):

            training_set_path = os.path.join(root_dir, destination_path)
            subdict = []

            with open(json_path, 'r') as json_file:

                dict_cars = json.load(json_file)
                filter_feature = []
                for i, attr in enumerate(d for d in dict_cars):
                    subdict.append([str(attr['image_id']).encode('ascii', 'ignore'),
                                    str(attr[filter_feature_type]).encode('ascii', 'ignore')])
                    filter_feature.append(attr[filter_feature_type])
                print(Counter(filter_feature))

                subdict.sort(key=lambda x: x[1])
                classes = set(x[1] for x in subdict)
                print(classes)
                time.sleep(2)
                for index in classes:
                    Preprocessor.__create_output_directories__(os.path.join(training_set_path, index))
                    for i, attr in enumerate(d for d in dict_cars):
                        if attr[filter_feature_type] == str(index):
                            if os.path.exists(os.path.join(image_path, attr['image_id']) +
                                     '.jpg'):
                                copyfile(os.path.join(image_path, attr['image_id']) +
                                     '.jpg',
                                os.path.join(training_set_path, str(index))+ '/'+attr['image_id'] +
                                     '.jpg')
                            else:
                                print("File",os.path.join(image_path, attr['image_id']) +
                                     '.jpg',"does not exist")

                    print(os.path.join(training_set_path, str(index)))

                print("DataSet Generated Successfully")


        @staticmethod
        def __generate_colors_for_bounding_boxes__(class_names):

            hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
            random.seed(10101)  # Fixed seed for consistent colors across runs.
            random.shuffle(colors)  # Shuffle colors to de-correlate adjacent classes.
            random.seed(None)  # Reset seed to default.
            return hsv_tuples, colors

        @staticmethod
        def __return_class_names__(classes_path):

            with open(classes_path) as f:
                class_names = f.readlines()
            class_names = [c.strip() for c in class_names]
            return class_names

        @staticmethod
        def __return_anchors__(anchors_path):

            with open(anchors_path) as f:
                anchors = f.readline()
                anchors = [float(x) for x in anchors.split(',')]
                anchors = np.array(anchors).reshape(-1, 2)
            return anchors

        @staticmethod
        def __binarize__(features):

            for i in range(0, len(features)):
                if features[i] > 0.5:
                    features[i] = 1
                else:
                    features[i] = 0
            return features

        @staticmethod
        def __get_similar__(test_dict, predict_dict, type):

            source  = test_dict[0][type]
            similar = []
            for j, attr in enumerate(d for d in predict_dict):
                if source == attr[type]:
                    similar.append({'type': attr[type], 'image': attr['image']})

            return similar

        @staticmethod
        def __mean__(a):
            return float(float(sum(a)) / float(len(a)))

        @staticmethod
        def __get_attributes_list__(indexes, json_file):

            attributes = []

            with open(json_file, 'r') as json_file:

                dict_cars = json.load(json_file)

                for i in range(0, len(indexes)):
                    for j, attr in enumerate(d for d in dict_cars):
                        if ''.join(indexes[i]) == ''.join(attr['image_id']) + '.jpg':
                            attributes.append({'image_id': attr['image_id'],
                                               'model': attr['model'],
                                               'image': attr['image'],
                                               'color': attr['color'],
                                               'body' : attr['body'],
                                               'make' : attr['make']
                                               })
                            break

                return attributes

        @staticmethod
        def __return_image_names__(data_path_with_image_name, indexes):

            image_names = []
            for idx in indexes:
                path = ''.join(data_path_with_image_name[idx]).split('/')
                image_names.append(path[-1:])

            return image_names

        @staticmethod
        def __get_concatenated_images__(data_path_with_image_name, indexes, thumb_height):

            thumbs = []

            for idx in indexes:
                img = image.load_img(data_path_with_image_name[idx])
                img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
                thumbs.append(img)
            concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
            return concat_image

        @staticmethod
        def __get_closest_images__(test_image_idx, pca_features, num_results):

             distances = [distance.hamming(pca_features[test_image_idx], feat) for feat in pca_features]

             idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]

             return idx_closest

        @staticmethod
        def __return_all_images__(datapath, image_per_category, num_of_categories):

            all_images = np.array([])
            shrinked_data_path = []

            for i in range(0, num_of_categories):

                extract_all_images = [x for x in os.listdir(datapath[i]) if x[-4:] == '.JPG' or x[-4:] == '.bmp' or x[-4:] == '.jpg']
                extract_all_images = extract_all_images[:image_per_category]
                shrinked_data_path.append(datapath[i])
                all_images = np.append(extract_all_images, all_images)

            return all_images, shrinked_data_path

        @staticmethod
        def __convert_to_thumbnails__(extract_all_images, path,root_dir):

            thumbnail_size = 128, 128
            thumbnail_output = os.path.join(root_dir + '/thumbnail_output')
            split_path = str(path).split('/')
            thumbnail_output = os.path.join(thumbnail_output, split_path[-1])

            if not os.path.exists(thumbnail_output):
                os.makedirs(thumbnail_output)
            else:
                shutil.rmtree(thumbnail_output)
                os.makedirs(thumbnail_output)

            for i, name in enumerate(extract_all_images):

                im = Image.open(os.path.join(path, extract_all_images[i]))
                im.thumbnail(thumbnail_size)
                im.save(os.path.join(thumbnail_output, name))
            return thumbnail_output

        @staticmethod
        def __convert_to_edges__(extract_all_images, path, root_dir):

            edge_detection_output = os.path.join(root_dir + '/edge_detection_output')
            split_path = str(path).split('/')
            edge_detection_output = os.path.join(edge_detection_output, split_path[-1])

            if not os.path.exists(edge_detection_output):
                os.makedirs(edge_detection_output)
            else:
                shutil.rmtree(edge_detection_output)
                os.makedirs(edge_detection_output)
            sigma = 0.33
            for i, name in enumerate(extract_all_images):
                img = cv2.imread(os.path.join(path, extract_all_images[i]))
                v = np.median(img)
                lower = int(max(0, (1.0 - sigma) * v))
                upper = int(min(255, (1.0 + sigma) * v))
                edges = cv2.Canny(img, lower, upper)
                imsave(os.path.join(edge_detection_output, name), edges)

            return edge_detection_output

        @staticmethod
        def __saliency_detection__(extract_all_images, path, root_dir):

            saliency_detection_output = os.path.join(root_dir + '/saliency_detection_output')
            split_path = str(path).split('/')
            saliency_detection_output = os.path.join(saliency_detection_output, split_path[-1])

            if not os.path.exists(saliency_detection_output):
                os.makedirs(saliency_detection_output)
            else:
                shutil.rmtree(saliency_detection_output)
                os.makedirs(saliency_detection_output)

            for i, name in enumerate(extract_all_images):
                # print(os.path.join(saliency_detection_output, name))
                mbd = psal.get_saliency_mbd(os.path.join(path, extract_all_images[i]))
                imsave(os.path.join(saliency_detection_output, name), mbd)

            return saliency_detection_output


        @staticmethod
        def __read_bounding_box_json__(root_dir, json_path):

            with open(root_dir+'/'+json_path) as json_file:
                data = json.load(json_file)
                return data

        @staticmethod
        def __extract_objects_from_images__(num_of_categories, datapath, s_datapath,
                                            object_extracted_path, dict_bounds, images_per_categories,
                                            root_dir, extract_obj_flag):

            object_extracted_path_lists = []

            for i in range(0, num_of_categories):

                extract_all_images = [x for x in os.listdir(s_datapath[i]) if x[-4:] == '.JPG' or x[-4:] == '.bmp' or x[-4:] == '.jpg']
                extract_all_images = extract_all_images[:images_per_categories]
                x = str(s_datapath[i]).split('/')
                category = ''.join(x[-1:])

                if extract_obj_flag == True:

                    Preprocessor.__create_output_directories__(os.path.join(root_dir, object_extracted_path + '/' + category))

                object_extracted_path_lists.append(os.path.join(root_dir, object_extracted_path + '/' + category))

                print("Extracting objects From", category)

                if extract_obj_flag == True:

                    for j, name in enumerate(extract_all_images):

                        im = Image.open(os.path.join(s_datapath[i], extract_all_images[j]))
                        im = im.convert('RGB')

                        for p in dict_bounds['bound']:

                            if ''.join(os.path.join(p['path'], p['name'])) == ''.join(os.path.join(datapath[i], extract_all_images[j]))\
                                   and p['label'] == 'car' and p['name'] == name:

                                im = im.crop((p['left'], p['top'], p['right'], p['bottom']))

                                if np.max(im) - np.min(im) != 0:

                                    imsave(os.path.join(root_dir, object_extracted_path + '/' + category) + '/' + name, im)
                                    break

                            else:

                                if np.max(im) - np.min(im) != 0 and p['label'] == 'car' and p['name'] == name:

                                    imsave(os.path.join(root_dir, object_extracted_path + '/' + category) + '/' + name, im)

            return object_extracted_path_lists


        @staticmethod
        def __load_image_data_thumbnails__(datapath, img_height, img_width, n_channel, num_of_categories, images_per_categories, root_dir,
                                           is_fixed_size, model_image_size, sess, yolo_model, input_image_shape, boxes,
                                           scores, classes, font_path, class_names, colors, segmented_output_path, json_path,
                                           object_extracted_path, segmentation_flag, edge_detection_flag, extract_obj_flag,grayscale_flag):

            print('-' * 30)
            print("DataSet Name:", datapath)
            print("DataSet Loading in Progress...\n")

            all_images, datapath = Preprocessor.__return_all_images__(datapath, images_per_categories, num_of_categories)

            # object segmentation

            if segmentation_flag == True:

                Segmentation.object_extraction(root_dir, json_path, num_of_categories, datapath, is_fixed_size, segmented_output_path,
                                  model_image_size, boxes, scores, classes, yolo_model, input_image_shape, font_path,
                                  class_names, sess, colors, all_images, Preprocessor, images_per_categories)

            dict_bounds = Preprocessor.__read_bounding_box_json__(root_dir, json_path)

            image_path = os.path.join(root_dir, segmented_output_path)

            segmented_data_path = glob(image_path + "/*")

            object_extracted_path = Preprocessor.__extract_objects_from_images__(num_of_categories,
                                                             datapath,
                                                             segmented_data_path,
                                                             object_extracted_path,
                                                             dict_bounds,
                                                             images_per_categories,
                                                             root_dir, extract_obj_flag)

            all_images, datapath = Preprocessor.__return_all_images__(object_extracted_path, images_per_categories, num_of_categories)

            if grayscale_flag == True:

                reading_as_rgb = np.empty((len(all_images), img_height, img_width, 1), dtype='float32')

            else:

                reading_as_rgb = np.empty((len(all_images), img_height, img_width, n_channel), dtype='float32')

            counter = 0

            shrinked_data_path_with_image_name = []

            for i in range(0, num_of_categories):

                extract_all_images = [x for x in os.listdir(object_extracted_path[i]) if x[-4:] == '.JPG' or x[-4:] == '.bmp' or x[-4:] == '.jpg']

                # Down-sizing the image to improve training time.

                thumbnail_datapath = Preprocessor.__convert_to_thumbnails__(extract_all_images, object_extracted_path[i], root_dir)

                if edge_detection_flag == True:

                    thumbnail_datapath = Preprocessor.__saliency_detection__(extract_all_images, object_extracted_path[i], root_dir)

                for j, name in enumerate(extract_all_images):

                    im = image.load_img(os.path.join(thumbnail_datapath, name), grayscale=grayscale_flag,
                                        target_size=(img_height, img_width, n_channel))

                    if grayscale_flag == True:

                        im = np.asarray(im, dtype='float32')
                        im = im.reshape([img_height, img_width, 1])

                    shrinked_data_path_with_image_name.append(object_extracted_path[i] + '/' + name)

                    im = np.asarray(im, dtype='float32')

                    if np.max(im) - np.min(im) != 0:
                        im = (im - np.min(im)) / (np.max(im) - np.min(im))

                    reading_as_rgb[counter] = im
                    counter += 1

                print(datapath[i] + "category converted")

            if grayscale_flag == True:

                reading_as_rgb = reading_as_rgb.reshape([np.size(reading_as_rgb, 0), img_height, img_width,1])

            else:

                reading_as_rgb = reading_as_rgb.reshape([np.size(reading_as_rgb, 0), img_height, img_width, n_channel])

            print('Loaded input images dim: ', len(reading_as_rgb), '\n...')

            return reading_as_rgb, all_images, datapath, shrinked_data_path_with_image_name


        @staticmethod
        def __save_image_score_maps__(test_input,score,output_path):

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            for id in range(0, len(test_input)):

                ax[0].imshow(test_input[id])  # if rgb
                ax[0].title.set_text('Input')
                ax[0].axis('off')
                ax[1].imshow(score[id, :, :, 0])  # , cmap='gray'
                ax[1].title.set_text('Score_map')
                ax[1].axis('off')
                plt.suptitle('sample' + str(id))
                plt.savefig(output_path + 'sample' + str(id) + '.png')

        @staticmethod
        def __get_score_model__(model, test_input, output_path):

            print("-" * 30)
            print("Generating scores in progress... \n")
            score = model.predict(test_input)
            print('Scores are being stored in' + output_path + '...\n')

            # Enable This If you want to check output score maps
            Preprocessor.__save_image_score_maps__(test_input, score, output_path)

            return score

        @staticmethod
        def __create_output_directories__(output_path_model):

            if not os.path.exists(output_path_model):
                os.makedirs(output_path_model)
            else:
                shutil.rmtree(output_path_model)
                os.makedirs(output_path_model)

        @staticmethod
        def __find_similarity__(feature1, feature2):

            for i in range(0, 14):
                img1 = Preprocessor.to_grayscale(feature1)
                img2 = Preprocessor.to_grayscale(feature2)

            n_m, n_0 = Preprocessor.compare_images(img1, img2)
            print("Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size)
            print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)

        @staticmethod
        def __flatten_img_data__(x_data):
            n_data = x_data.shape[0]
            flatten_dim = np.prod(x_data.shape[1:])
            x_data_flatten = x_data.reshape((n_data, flatten_dim))
            return x_data_flatten

        @staticmethod
        def __load_model_weights__(weight_folder, weight_file, input_shape, input_shape_3, type):

            weight_file_path = weight_folder + weight_file
            print('Weight File Name: ' + weight_file_path + '....\n')
            model = ""
            if type == "Model_1":
                model = Model_1.Model_1(input_shape)
                model.load_weights(weight_file_path)
            elif type == "Model_2":
                model = Model_2.Model_2(input_shape)
                model.load_weights(weight_file_path)
            elif type == "Model_3":
                model = Model_3.Model_3(input_shape_3)
                model.load_weights(weight_file_path)

            return model

        @staticmethod
        def to_grayscale(arr):

            if len(arr.shape) == 3:
                return average(arr, -1)  # average over the last axis (color channels)
            else:
                return arr

        @staticmethod
        def compare_images(img1, img2):

            img1 = Preprocessor.normalize(img1)
            img2 = Preprocessor.normalize(img2)
            diff = img1 - img2
            m_norm = sum(abs(diff))
            z_norm = norm(diff.ravel(), 0)
            return (m_norm, z_norm)

        @staticmethod
        def normalize(arr):
            rng = arr.max()-arr.min()
            amin = arr.min()
            return (arr-amin)*255/rng

except ImportError as E:
    raise E