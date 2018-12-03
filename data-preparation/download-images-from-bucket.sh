#!/bin/bash

SOURCE=$1 jq  --raw-output 'map(env.SOURCE+"/"+.image_id+".jpg") | .[] | tostring' | gsutil -m cp -I ${2:-.}

#jq  --raw-output --arg SOURCE $1 'map($SOURCE.image_id+".jpg") | .[] | tostring' | gsutil -m cp -I ./


