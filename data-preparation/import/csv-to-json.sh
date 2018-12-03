#!/bin/bash

set -e 

fgrep -v NULL $1 | jq --slurp --raw-input --raw-output \
  'split("\n") | .[0:-1] | map(split("\",\"")) |
      map({"id": .[0]|ltrimstr("\""),
           "make": .[1],
           "model": .[2],
           "year": .[3],
           "body": .[4],
           "color": .[5],
           "seller": .[6],
           "images": .[7]|rtrimstr("\"")|split(",")})'
