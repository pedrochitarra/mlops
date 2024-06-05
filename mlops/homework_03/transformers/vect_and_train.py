from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
# import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs) -> tuple:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    # Fit the DictVectorizer and preprocess data
    print("Here!")
    y_train = data["duration"].values
    categorical = ['PULocationID', 'DOLocationID']
    dv = DictVectorizer()
    dicts = data[categorical].to_dict(orient='records')
    print("Fitting DV")
    X_train = dv.fit_transform(dicts)
    # data, dv = preprocess(data, True)
    print("Fitting Linear Regression")
    reg = LinearRegression().fit(X_train, y_train)
    print(reg.intercept_)

    return reg, dv


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'