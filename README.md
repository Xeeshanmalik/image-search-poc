# Visual Search 


1. Download Cars Data From 
<a>https://ai.stanford.edu/~jkrause/cars/car_dataset.html</a>
    <br> Download the following Files
    1. cars_meta.mat
    2. cars_train_annos.mat
    
    Save the files inside ${ProjectFolder}/devkit/

2. #####Data Preparation
    1. cd data-preparation
    2. source activate image-search-poc-data-preparation
    3. python prepare-stanford-car-dataset.py 
    --cars_meta ../devkit/cars_meta.mat 
    --cars_annos ../devkit/cars_train_annos.mat 
    '> ../devkit/by-id/01/dataset.json
    
    4. python split.py 
    --train_set_path ../devkit/by-id/01/train.json 
    --dev_set_path ../devkit/by-id/01/dev.json 
    --test_set_path ../devkit/by-id/01/test.json 
    <../devkit/by-id/01/dataset.json 
    
    5. python stats.py 
    <../devkit/by-id/01/test.json 
    '>' ../devkit/by-id/01/test-stats.json
    
3.  