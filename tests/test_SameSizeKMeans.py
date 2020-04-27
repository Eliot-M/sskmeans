from sskmeans import SameSizeKmeans
import pytest
import pandas as pd
import numpy as np


def test_default_ssk():
    """
    Check for logical default values. Randomness and fixed sizes.
    """
    x = SameSizeKmeans()

    assert x.random_state is None
    assert x.flexibility_size == 0


def test_error_ssk():
    """
    Check for basic non-sense inputs.
    """
    with pytest.raises(ValueError):
        x = SameSizeKmeans(max_iter=-1)
        x = SameSizeKmeans(flexibility_size=-1)
        x = SameSizeKmeans(n_clusters=-1)

    with pytest.raises(TypeError):
        x = SameSizeKmeans(max_iter="1")
        x = SameSizeKmeans(flexibility_size="1")
        x = SameSizeKmeans(n_clusters="5")


def test_x_format():
    """
    Check if format "Pandas" or "Numpy" are still valids and not others.
    """
    myclass_fitted = SameSizeKmeans(n_clusters=2)

    myclass_fitted.fit(X=pd.DataFrame({'x': [0, 1, 2, 3, 4, 5, 6]}))
    assert myclass_fitted._n_samples == 7

    myclass_fitted.fit(X=np.array([[1, 2], [2, 2], [3, 2]]))
    assert myclass_fitted._n_samples == 3

    with pytest.raises(TypeError):
        myclass_fitted.fit(X="Random string")
        myclass_fitted.fit(X=[1, 2, 3])
        myclass_fitted.fit(X=10)


def test_transform_add_columns():
    """
    Check if 2 new columns are created.
    """
    myclass_fitted = SameSizeKmeans(n_clusters=3)
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5, 6], 'y': [1, 2, 3, 4, 5, 6, 7]})

    myclass_fitted.fit(X=df)
    df_out = myclass_fitted.transform(X=df)
    assert df_out.shape == (myclass_fitted._n_samples, myclass_fitted._n_features + 2)


def test_clusters():
    """
    Check if it returns the correct number of clusters with contraint sizes
    """
    myclass_fitted = SameSizeKmeans(n_clusters=3)
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5, 6], 'y': [1, 2, 3, 4, 5, 6, 7]})

    myclass_fitted.fit(X=df)
    df_out = myclass_fitted.transform(X=df)
    assert set(df_out.cluster_id) == {0, 1, 2}

    assert len(df_out[df_out['cluster_id'] == 0]) >= myclass_fitted._k_min
    assert len(df_out[df_out['cluster_id'] == 0]) <= myclass_fitted._k_max
    assert len(df_out[df_out['cluster_id'] == 1]) >= myclass_fitted._k_min
    assert len(df_out[df_out['cluster_id'] == 1]) <= myclass_fitted._k_max
    assert len(df_out[df_out['cluster_id'] == 2]) >= myclass_fitted._k_min
    assert len(df_out[df_out['cluster_id'] == 2]) <= myclass_fitted._k_max

#
