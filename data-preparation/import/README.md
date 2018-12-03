# import

- ssh into ngdbbatch001.prod2
- copy the files export.sh and query.sql to it
- run the ./export.sh script (u can get the required password from ecg-puppet)
- copy the generated file to your machine
- run ./csv-to-json.sh cars.csv > cars.json to generate  the json file
- flatten.py
- download_images.sh
- cleanup.py
