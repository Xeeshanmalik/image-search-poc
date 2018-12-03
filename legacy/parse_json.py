try:
    import ijson
    import tensorflow as tf
    import os
    import json
    from itertools import islice
    import time
    from collections import Counter
    import random
    from sys import argv

    def __return_unique_filter_feature__(objects, filter_feature, i, type):

        for line in islice(objects, 10000000):
            filter_feature.add(line[type])

            if len(list(filter_feature)) == i:
                break
        return filter_feature

    def __return_populated_dict__(model, data_dictionary, objects, count_per_makes, type):

        for m in model:
               counter = 0
               for line in islice(objects, 1000000):
                   if line[type] == m:
                       data_dictionary.append({'make': str(line['make']),
                                               'model': str(line['model']),
                                               'image_id': str(line['image_id']),
                                               'id': int(line['id']),
                                               'image_index': int(line['image_index']),
                                               'seller': line['seller'],
                                               'color': line['color'],
                                               'body': line['body'],
                                               'image': line['image']})

                       print("...................")
                       print("\n")
                       print("Parsing Big Json File of All Images....")
                       print("\n")
                       print("...................")
                       print("Record Processed for ", ''.join(m).encode('utf-8'), counter)

                       counter += 1

                   if counter == count_per_makes:
                            break

        return data_dictionary

    def __return_unique_filter_feature_per_make__(objects, filter_feature, i,type, make):

         for line in islice(objects, 10000000):
            if line['make'] == make:
                filter_feature.add(line[type])

                if len(list(filter_feature)) == i:
                    break

         return filter_feature


    def __parse_json_file__():

       root_dir = os.path.dirname(os.path.abspath(__file__))
       super_set_filter= int(argv[1])
       number_of_class_per_filter = int(argv[2])
       count_per_filter = int(argv[3])
       starttime = time.clock()

       with open(os.path.join(root_dir, argv[5])) as json_file:

           objects = ijson.items(json_file, 'item')
           filter_feature = set()
           data_dictionary = []

           print("\n")
           print("Parsing Big Json File of All Images....")
           print("\n")
           print("This may take few minutes depend on how much data you are extracting")
           print("\n")
           print("If you want to run it quick decrease the number of makes/model and count per each make/model")
           print("\n")

           if argv[7] == 'train':

               for i in range(1, super_set_filter+1):

                   filter_feature = __return_unique_filter_feature__(objects, filter_feature, i, str(argv[4]))

               print(filter_feature)
               print(number_of_class_per_filter)

               filter_feature = random.sample(filter_feature, number_of_class_per_filter)

           elif argv[7] == 'test':

               d = os.path.join(root_dir, 'trainingset')

               list_of_make = filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d))

               for i in range(1, number_of_class_per_filter+1):

                    filter_feature = __return_unique_filter_feature_per_make__(objects,
                                                                               filter_feature,
                                                                               i,
                                                                               str(argv[4]),
                                                                               list_of_make[i-1])
           print(time.clock() - starttime)

           resulting_dict = __return_populated_dict__(filter_feature, data_dictionary, objects, count_per_filter, str(argv[4]))

           filtered_by_index = []

           for k, index in enumerate(d['image_index'] for d in resulting_dict):
                if index == 0:

                    filtered_by_index.append(resulting_dict[k])

       with open(os.path.join(root_dir, str(argv[6])), 'w') as f_json_file:

           f_json_file.write(json.dumps(filtered_by_index, indent=1))

       f_json_file.close()

       print("Total Time Used for Model-Wise Filtration", time.clock()-starttime)

       with open(os.path.join(root_dir, argv[6]), 'r') as w_json_file:
           dict_cars = json.load(w_json_file)

       filter_feature = []

       for l, attr in enumerate(d[str(argv[4])] for d in dict_cars):
           filter_feature.append(attr)

       print(Counter(filter_feature))

    def main(_):

      __parse_json_file__()

    if __name__ == '__main__':
        tf.app.run()

except ImportError as E:
    raise E