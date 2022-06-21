import sys
import pickle
import pandas as pd

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    return df


def prepare_dicts(df):
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts


def apply_model(input_file, year, month, output_file):

    #load model
    dv, lr = load_model()

    #load data
    print(f'reading the data from {input_file}...')
    df = read_data(input_file)
    print(df.head())

    #get dicts
    dicts = prepare_dicts(df)

    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('Mean of preds: ', y_pred.mean())
    #2021-03: 16,29

    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    taxi_type = sys.argv[1] # 'fhv'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3
    
    input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    apply_model(
        input_file=input_file,
        year=year,
        month=month,
        output_file=output_file
    )


if __name__ == '__main__':
    run()
