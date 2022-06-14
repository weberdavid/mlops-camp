#!/bin/bash

#Notes: https://gist.github.com/Qfl3x/8dd69b8173f027b9468016c118f3b6a5

#start the UI
prefect orion start

# check prefect storage
prefect storage ls

# setup prefect storage
prefect storage create

# create a deployment
prefect create deployment create file-name.py

#preview work queue with queue id
prefect work-queue preview a60c8457-3699-4cb6-9450-ef1b3f0156b1

#list available work queues
prefect work-queue ls

#inspect a work-queue with queue id
prefect work-queue inspect a60c8457-3699-4cb6-9450-ef1b3f0156b1