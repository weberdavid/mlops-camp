# should be same python as used for development
FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

# copied both files, as I was not using pip
COPY [ "Pipfile", "Pipfile.lock", "./" ]

# --system = no new virtual env, but installing in system
# --deploy = enforce that Pipfile.lock is up to date, if not it fails
RUN pipenv install --system --deploy

COPY [ "predict.py", "lin_reg.bin", "./" ]

# needed, as this opens the port
EXPOSE 9696

# start gunicorn to run flask app
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]