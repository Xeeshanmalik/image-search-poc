#!/bin/bash

jq  'map(.image , "-o", "images/"+.image_id+".jpg") | .[] | tostring' | xargs -P 128 -n 3 curl -s -L
