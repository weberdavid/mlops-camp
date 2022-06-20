FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app
RUN mkdir output
RUN mkdir output/fhv output/green output/yellow


RUN pip install -U pip
RUN pip install pipenv 
COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "starter.py", "./" ]

EXPOSE 9696
