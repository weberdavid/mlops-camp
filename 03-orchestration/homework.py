import pandas as pd

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule, CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

from datetime import datetime, timedelta
from dateutil import relativedelta

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):

    #logging
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    #logging
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):

    #logging
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


# path_date = '2021-03-15'
# renamed date to path_date, as it could interfer with the module "date" from datetime
@task
def get_paths(path_date):

    #logging
    logger = get_run_logger()
    
    if path_date is None:
        path_date = datetime.today()
    else:
        path_date = datetime.strptime(path_date, '%Y-%m-%d')
    
    train_date = path_date - relativedelta.relativedelta(months = 2)
    val_date = path_date - relativedelta.relativedelta(months = 1)

    train_year, train_month = train_date.year, train_date.strftime('%m')
    val_year, val_month = val_date.year, val_date.strftime('%m')

    train_path = f'../data/fhv_tripdata_{train_year}-{train_month}.parquet'
    val_path = f'../data/fhv_tripdata_{val_year}-{val_month}.parquet'

    logger.info(f"The path for training data is: {train_path}")
    logger.info(f"The path for validation data is: {val_path}")

    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(path_date=None):

    #logging
    logger = get_run_logger()

    train_path, val_path = get_paths(path_date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # when do you need to call .result() on a tasks return value?
    # - when multiple values are returned (prefect does not support that yet)
    # - when you want to interact with a return value directly in plain python (like return_val > 3 or so)
    # - in prefect tasks, .result() is automatically called, that's why you don't need it there

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # save dv and save model
    with open(f'../models/dv-{path_date}.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
    logger.info(f'Saved DictVectorizer as: dv-{path_date}.b')

    with open(f'../models/model-{path_date}.bin', 'wb') as f_out:
            pickle.dump(lr, f_out)
    logger.info(f'Saved Model as: model-{path_date}.bin')



# main(path_date = '2021-08-15')


DeploymentSpec(
    flow=main,
    name='model_training-hw',
    schedule=CronSchedule(
        cron='0 9 15 * *',
        timezone='Europe/Vienna'
    ),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml', 'cron']
) 