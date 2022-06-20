import pickle
from flask import Flask, request, jsonify

app = Flask('duration-prediction')


with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


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
