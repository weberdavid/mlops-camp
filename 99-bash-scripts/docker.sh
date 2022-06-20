#!/bin/bash

# building the docker container
docker build -t ride-duration-pred-service:v1 .

# run the container
# it = interactive mode
# rm = remove it after ending it
# --entrypoint = specifying an entrypoint, such as bash
docker run -it --rm -p 9696:9696 --entrypoint /bin/bash ride-duration-pred-service:v1