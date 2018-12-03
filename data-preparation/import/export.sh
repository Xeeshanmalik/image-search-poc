#!/bin/bash

#
# get the password from hiera/ca/defaults/yaml  @  ecg-puppet

set -e 

mysql belen_ad -u fisci_ro -p < query.sql | tail -n +2 > cars.csv
