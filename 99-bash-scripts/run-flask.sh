#!/bin/bash

# run flask app
python predict.py

# run flask in development mode
gunicorn --bind=0.0.0.0:9696 predict:app