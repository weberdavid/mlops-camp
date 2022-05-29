#!/bin/bash

mlflow ui --backend-store-uri sqlite:///mlflow.db

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root artifacts