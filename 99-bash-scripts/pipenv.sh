#!/bin/bash

# install a new pipenv
pipenv install scikit-learn==1.0.2 flask --python=/Users/da.weber/opt/anaconda3/envs/mlops/bin/python

# activate it
pipenv shell

# exit it
exit

# remove it
pipenv --rm