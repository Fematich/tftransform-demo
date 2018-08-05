"""
Preprocess module that defines the transform function and applies it to the data
"""

from googleapiclient import discovery

from trainer.config import PROJECT_ID


def get_predictions(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): GCP project where the ML Engine Model is deployed.
        model (str): model name
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version (str) version of the model to target

    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.

    """
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


if __name__ == "__main__":
    predictions = get_predictions(
        project=PROJECT_ID,
        model="digitaltwin",
        instances=[
            {
                'ButterMass':120,
                'ButterTemperature': 20,
                'SugarMass': 200,
                'SugarHumidity': 0.22,
                'FlourMass': 50,
                'FlourHumidity': 0.23,
                'HeatingTime': 50,
                'MixingSpeed': 'Max Speed',
                'MixingTime': 200,
            }]
    )
    print(predictions)
