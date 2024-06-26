if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow


mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc_yellow_taxis")

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    model, dv = data

    with mlflow.start_run():
        mlflow.set_tag("developer", "pedro")
        mlflow.sklearn.log_model(model, "models_mlflow")
        # mlflow.log_artifact(dv, "dict_vectorizer")

    return data