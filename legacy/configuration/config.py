import argparse


class Configuration:

    def __init__(self):

        parser = argparse.ArgumentParser(description="Process Job arguments.")
        parser.add_argument("image_height", type=int,help="height of an input image")
        parser.add_argument("image_width", type=int, help="width of an input image")
        parser.add_argument("channel", type=int,help="number of channels")
        parser.add_argument("compressed_image_height", type=int, help="thumbnail height of an input image")
        parser.add_argument("compressed_image_width", type=int, help="thumbnail width of an input image")
        parser.add_argument("compressed_channel", type=int,help="number of channels in thumbnail")
        parser.add_argument("number_of_categories", type=int, help="total number of categories")
        parser.add_argument("number_of_images_per_category", type=int,help="total number of images per category")
        parser.add_argument("batch_size", type=int,help="batch size per epoch")
        parser.add_argument("epochs", type=int, help="batch size per epoch")
        parser.add_argument("gpu_state", type=str, help="batch size per epoch")
        parser.add_argument("output_model_1", type=str, help="model_1 output")
        parser.add_argument("output_model_2", type=str, help="model_2 output")
        parser.add_argument("output_model_3", type=str, help="model_3 output")
        parser.add_argument("model_1_save", type=str, help="model_1 output")
        parser.add_argument("model_2_save", type=str, help="model_2 output")
        parser.add_argument("model_3_save", type=str, help="model_3 output")
        parser.add_argument("data", type=str, help="model_3 output")

        # object detection and segmentation arguments

        parser.add_argument('model_path', help='path to h5 model file containing body of a YOLO model')
        parser.add_argument('anchors_path', help='path to anchors file, defaults to yolo_anchors.txt', default='model_data/yolo_anchors.txt')
        parser.add_argument('classes_path', help='path to classes file, defaults to coco_classes.txt', default='model_data/coco_classes.txt')
        parser.add_argument('test_path', help='path to directory of test images, defaults to images/', default='trainingset')
        parser.add_argument('segmented_output_path', help='path to output test images, defaults to images/out',  default='trainingset_segmented/')
        parser.add_argument('score_threshold', type=float, help='threshold for bounding box scores, default .3', default=.3)
        parser.add_argument('iou_threshold', type=float, help='threshold for non max suppression IOU, default .5', default=.5)
        parser.add_argument('font_path', help='path to output test images, defaults to images/out',  default='./font/FiraMono-Medium.otf')
        parser.add_argument('json_output', type=str, help='file to output bounding box dimensions', default='bounding_boxes.txt')
        parser.add_argument('image_path', type=str, help='name of the json file used as input')
        parser.add_argument('input_filename', type=str, help='name of the json file used as input')
        parser.add_argument('number_of_predictions', type=int, help='number of predictions to calculate precision')
        parser.add_argument('precision_counter', type=str,help='list of all images and their assoc. attributed ')

        self.__args = parser.parse_args()

    @property
    def image_height(self):
        return self.__args.image_height

    @property
    def image_width(self):
        return self.__args.image_width

    @property
    def channel(self):
        return self.__args.channel

    @property
    def compressed_image_height(self):
        return self.__args.compressed_image_height

    @property
    def compressed_image_width(self):
        return self.__args.compressed_image_width

    @property
    def compressed_channel(self):
        return self.__args.compressed_channel

    @property
    def number_of_categories(self):
        return self.__args.number_of_categories

    @property
    def number_of_images_per_category(self):
        return self.__args.number_of_images_per_category

    @property
    def batch_size(self):
        return self.__args.batch_size

    @property
    def epochs(self):
        return self.__args.epochs

    @property
    def gpu_state(self):
        return self.__args.gpu_state

    @property
    def output_model_1(self):
        return self.__args.output_model_1

    @property
    def output_model_2(self):
        return self.__args.output_model_2

    @property
    def output_model_3(self):
        return self.__args.output_model_3

    @property
    def model_1_save(self):
        return self.__args.model_1_save

    @property
    def model_2_save(self):
        return self.__args.model_2_save

    @property
    def model_3_save(self):
        return self.__args.model_3_save

    @property
    def data(self):
        return self.__args.data

    @property
    def model_path(self):
        return self.__args.model_path

    @property
    def anchors_path(self):
        return self.__args.anchors_path

    @property
    def classes_path(self):
        return self.__args.classes_path

    @property
    def test_path(self):
        return self.__args.test_path

    @property
    def segmented_output_path(self):
        return self.__args.segmented_output_path

    @property
    def score_threshold(self):
        return self.__args.score_threshold

    @property
    def iou_threshold(self):
        return self.__args.score_threshold

    @property
    def font_path(self):
        return self.__args.font_path

    @property
    def json_output(self):
        return self.__args.json_output

    @property
    def image_path(self):
        return self.__args.image_path

    @property
    def input_filename(self):
        return self.__args.input_filename

    @property
    def number_of_predictions(self):
        return self.__args.number_of_predictions

    @property
    def precision_counter(self):
        return self.__args.precision_counter