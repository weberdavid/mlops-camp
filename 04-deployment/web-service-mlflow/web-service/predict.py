from multiprocessing.pool import RUN
import pickle
from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient

app = Flask('duration-prediction')


RUN_ID = 'some-run-id from ml flow'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('experiment name where we pull the model from')
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)
path = client.download_artifacts(run_id=RUN_ID, path='dict_vect.bin')

with open(path, 'rb') as f_in:
    dv = pickle.load(f_in)


def prep_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    X = dv.transform(features)
    pred = model.predict(X)
    return pred[0]


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prep_features(ride)
    pred = predict(features)

    result = {
        'duration' : pred
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 9696)
