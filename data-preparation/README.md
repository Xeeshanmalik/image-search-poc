# data preparation

Tips and Tricks

- how to download the images corresponding to a sample ?

  - produce sample: ```cat dev.json | python sample-ad.py --by_make --by_model -n 5 > tmp/sample-5-per-make-model.json```
  - download images: ```cat tmp/sample-5-per-make-model.json | ./download-images-from-bucket.sh gs://dev_visual_search/v0/images-thumbnail/v1  ./tmp/images/```
