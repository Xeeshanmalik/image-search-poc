try:

    import os
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from keras import backend as K
    import json, io
    import time

    class Segmentation:

        @staticmethod
        def object_extraction(root_dir, json_path, num_of_categories, datapath, is_fixed_size, segmented_output_path,
                              model_image_size, boxes, scores, classes, yolo_model, input_image_shape, font_path,
                              class_names, sess, colors, all_images, preprocessor,images_per_categories):

            # object segmentation

            with open(root_dir+'/'+json_path, 'w') as outfile:

                data = {}

                data['bound'] = []

                for i in range(0, num_of_categories):

                    extract_all_images = [x for x in os.listdir(datapath[i]) if x[-4:] == '.JPG' or x[-4:] == '.bmp' or x[-4:] == '.jpg']

                    extract_all_images = extract_all_images[:images_per_categories]

                    category_name = str(datapath[i]).split('/')

                    preprocessor.__create_output_directories__(os.path.join(root_dir, segmented_output_path + '/' + ''.join(category_name[-1:])))

                    for j, name in enumerate(extract_all_images):
                        if os.path.exists(os.path.join(datapath[i], name)):

                            try:

                                imge = Image.open(os.path.join(datapath[i], name))
                                imge = imge.convert('RGB')

                                if is_fixed_size:

                                    resized_image = imge.resize(
                                    tuple(reversed(model_image_size)), Image.BICUBIC)
                                    image_data = np.array(resized_image, dtype='float32')
                                else:

                                    new_image_size = (imge.width - (imge.width % 32),
                                          imge.height - (imge.height % 32)), 3
                                    resized_image = imge.resize(new_image_size, Image.BICUBIC)
                                    image_data = np.array(resized_image, dtype='float32')

                                image_data /= 255.
                                image_data = np.expand_dims(image_data, 0)  # Add batch dimension
                                out_boxes, out_scores, out_classes = sess.run(
                                [boxes, scores, classes],

                                feed_dict={
                                            yolo_model.input: image_data,
                                            input_image_shape: [imge.size[1], imge.size[0]],
                                            K.learning_phase(): 0
                                          })

                                print('Found {} boxes for {}'.format(len(out_boxes), os.path.join(datapath[i], name)))

                                if len(out_boxes) > 0:
                                        result = np.arange(len(out_boxes))
                                        for t in range(0,len(out_boxes)):
                                            first_difference = abs(out_boxes[t][0] - out_boxes[t][2])
                                            second_difference= abs(out_boxes[t][1] - out_boxes[t][3])
                                            result[t] = first_difference + second_difference
                                        index = np.argmax(result)

                                        out_boxes = out_boxes[index][0:4]

                                font = ImageFont.truetype(
                                font=font_path,
                                size=np.floor(3e-2 * imge.size[1] + 0.5).astype('int32'))
                                thickness = (imge.size[0] + imge.size[1]) // 300

                                for k, c in reversed(list(enumerate(out_classes))):

                                    predicted_class = class_names[c]
                                    box = out_boxes
                                    score = out_scores[k]
                                    label = '{}'.format(predicted_class)

                                    draw = ImageDraw.Draw(imge)
                                    label_size = draw.textsize(label, font)
                                    top, left, bottom, right = box
                                    top = max(0, np.floor(top + 0.5).astype('int32'))
                                    left = max(0, np.floor(left + 0.5).astype('int32'))
                                    bottom = min(imge.size[1], np.floor(bottom + 0.5).astype('int32'))
                                    right = min(imge.size[0], np.floor(right + 0.5).astype('int32'))

                                    data['bound'].append({
                                         'name': name,
                                         'path': datapath[i],
                                         'left': int(left),
                                         'top' : int(top),
                                         'right': int(right),
                                         'bottom': int(bottom),
                                         'label': label,
                                         'probability': float(score)
                                          })

                                    if top - label_size[1] >= 0:
                                        text_origin = np.array([left, top - label_size[1]])
                                    else:
                                        text_origin = np.array([left, top + 1])

                                    for l in range(thickness):
                                        draw.rectangle(
                                            [left + l, top + l, right - l, bottom - l],
                                            outline=colors[c])
                                    draw.rectangle(
                                        [tuple(text_origin), tuple(text_origin + label_size)],
                                        fill=colors[c])
                                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                                    del draw
                                imge.save(os.path.join(root_dir, segmented_output_path + '/' + ''.join(category_name[-1:]) + '/'
                                                   + name), quality=90)
                            except IOError as e:

                                print("Corrupt File", e)

                        else:
                            print("File Name", name, "does not exist")

                json.dump(data, outfile, indent=2)
                print("Total Number of Samples Found:" + str(len(all_images)) + '\n')

except ImportError as E:
    raise E