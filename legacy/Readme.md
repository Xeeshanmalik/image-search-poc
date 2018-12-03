# Fusion of Unsupervised Feature Maps For Image Similarity Detection

![alt text](https://github.corp.ebay.com/zmalik/image-similarity-fusion/blob/master/architecture/architecture/Slide1.png)

## Currently in Rigor/Prototype Phase (in Progress)

This prototype consist of fusion of output maps in a totally unsupervised 
manner for more than one deep networks to perform image to image similarity 
detection. The rationale of this idea is to output image related recommendations 
or rank similar images based on similarity within category to power products such as

1.  Image search (based on similarity). For example 'find this car'
2.  Similar items on VIP or Watchlists

The benefit of this approach is that results can also have a layer of business rules or further pipelines added on the output. For instance, ranking certain categories higher, filtering, etc...

There are lots of improvements due for this prototype to improve its
performance. Each of its phase is right now in its prototype state explicitly
intended to first complete and evaluate the output of the full cycle of 
image similarity detection. 

## Potential Data sets containing worst-case scenarios for this algorithm

1) https://www.cs.toronto.edu/~kriz/cifar.html
2) http://www.vision.caltech.edu/Image_Datasets/Caltech101/
3) https://groups.csail.mit.edu/vision/SUN/
4) http://ai.stanford.edu/~jkrause/cars/car_dataset.html

<b>Note 1: Download the data set and add it to your root folder. Change 
the name of ${DATA} variable in both root shell scripts (run-prediction.sh)
and (run-training.sh) with your dataset root folder name.</b>
 

## Package Dependency

1)  backports.csv==1.0.5
2)  backports.functools-lru-cache==1.4
3)  cli-helpers==1.0.1
4)  configparser==3.5.0
5)  cycler==0.10.0
6)  decorator==4.1.2
7)  funcsigs==1.0.2
8)  h5py==2.7.1
9)  ijson==2.3
10) Keras==2.1.2
11) matplotlib==2.1.0
12) mock==2.0.0
13) networkx==2.0
14) numpy==1.13.3
15) olefile==0.44
16) opencv-python==3.3.0.10
17) pbr==3.1.1
18) Pillow==4.3.0
19) protobuf==3.5.0.post1
20) pyparsing==2.2.0
21) python-dateutil==2.6.1
22) pytz==2017.3
23) PyWavelets==0.5.2
24) PyYAML==3.12
25) scikit-image==0.13.1
26) scikit-learn==0.19.1
27) scipy==1.0.0
29) six==1.11.0
30) subprocess32==3.2.7
31) tabulate==0.8.2
32) tensorflow-gpu==1.1.0
33) terminaltables==3.1.0
34) Theano==1.0.1
35) Unidecode==0.4.21
36) Werkzeug==0.12.2

## Configuration by installing conda virtual environment

1) First Install Conda
2) Second run 'conda env create' inside the project folder
3) Third run 'source activate image-similarity-fusion	

## Note: Potential Idea for Within Category Image to Image Recommendation

## For Training (You need to run once before testing the output)

````
    ./run-training.sh
````

## For Prediction

````
    ./run-prediction.sh
````

## Test Image / Source Input Image

![alt text](https://github.corp.ebay.com/zmalik/image-similarity-fusion/blob/master/architecture/test.png)

## Recommendations

![alt text](https://github.corp.ebay.com/zmalik/image-similarity-fusion/blob/master/architecture/recommendations.png)


## Code Independence

1) Object Detection

   To run only object detection pre-trained model and predict bounding box on a small test json set:-
   
   ````
    python object_detection.py --base_image_path local_test_data/ < local_test_data/test_object_detection.json
    
   ````

2) Thumb-Nail Generation
   
   To run only thumbnail generation 
 
   ````
    python thumbnail.py --base_image_path local_test_dev/ < local_test_data/test_object_detection.json --output_image_path project_root_dir/thumbnail_output
   
   ````
3) Salient Image Generation
   
   To run only salient generation 
 
   ````
    python saliency.py --base_image_path thumbnail_output/ < local_test_data/test_object_detection.json --output_image_path project_root_dir/saliency_output
   
   ````
4) Running Training
   
   To run training for each model
   
   ````
    python train.py --base_image_path_for_model local_test_data/ --model_save_path /save_model --im_height 128 --im_width 128 --im_channel 3 < local_test_data/test_object_detection.json
   
   ````