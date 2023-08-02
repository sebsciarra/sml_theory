import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from smltheory.generate_data import provide_weights, compute_rescaling_factor


def generate_bayes_features(data):
    """Generates features for Bayes decision function.

     Bayes decision function for wine quality was set to be the outcome a second-order
     polynomial model of weather and winemaking quality and the interaction between the
     first-order version of the predictors.

     Parameters
     -----------
     data : pandas.core.frame.DataFrame
        A dataframe created using `generate_mult_trunc_normal()` from the `generate_data`
        module.

     Returns
     -------
     bayes_features : pandas.core.frame.DataFrame
         A dataframe that contains features needed to compute Bayes decision function predictions

     See Also
     --------
     generate_data.generate_mult_trunc_normal()

     Examples
     ---------
     >>> mu=[5, 7]
     >>> sd=[1.2, 1.7]
     >>> rho_weather_winemaking =  0.35
     >>> cov_matrix=generate_data.create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
     >>>sample_size_gen_error = 150
     >>> data_gen_error = generate_data.generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                          mu=mu, sample_size=sample_size_gen_error)
     >>> compute_bayes_features(data=data_gen_error)

     """

    # Feature columns
    predictors = data[["weather", "winemaking_quality"]]
    bayes_features = pd.concat(objs=[predictors, predictors**2, predictors.prod(axis=1)], axis=1)

    return bayes_features


def compute_bayes_intercept(data):
    """Computes intercept for Bayes decision function.

    Bayes decision function for wine quality was set to be the outcome a second-order
    polynomial model of weather and winemaking quality and the interaction between the
    first-order version of the predictors.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
       A pd.DataFrame created using `generate_mult_trunc_normal()` from the `generate_data`
       module

    Returns
    ---------
    intercept: np.array
        Intercept value of Bayes decision function.

    See Also
    ---------
    generate_data.generate_mult_trunc_normal()

    Examples
    ---------
    >>> mu = [5, 7]
    >>> sd = [1.2, 1.7]
    >>> rho_weather_winemaking =  0.35
    >>> cov_matrix = generate_data.create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    >>> sample_size_gen_error = 150
    >>> data_gen_error = generate_data.generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                         mu=mu, sample_size=sample_size_gen_error)
    >>> compute_bayes_intercept(data=data_gen_error)
    """

    bayes_features = generate_bayes_features(data=data)

    # Intercept
    intercept = np.mean(data["wine_quality"]) - np.sum(provide_weights() * compute_rescaling_factor() *
                                                       np.mean(bayes_features, axis=0))

    return intercept


def compute_bayes_risk(data):
    """Computes generalization error/risk of Bayes decision function .

    Bayes decision function for wine quality was set to be the outcome a second-order
    polynomial model of weather and winemaking quality and the interaction between the
    first-order version of the predictors.

    Parameters
    --------
    data: pandas.core.frame.DataFrame
       A pd.DataFrame created using `generate_mult_trunc_normal()` from the `generate_data`
       module

    Returns
    --------
    bayes_risk: numpy.float64
        Generalization error/risk of Bayes decision function.

    See Also
    --------
    generate_data.generate_mult_trunc_normal()

    Examples
    --------
    >>> mu = [5, 7]
    >>> sd = [1.2, 1.7]
    >>> rho_weather_winemaking =  0.35
    >>> cov_matrix = generate_data.create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    >>> sample_size_gen_error = 150
    >>> data_gen_error = generate_data.generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                         mu=mu, sample_size=sample_size_gen_error)
    >>> compute_bayes_risk(data=data_gen_error)
    """

    bayes_features = generate_bayes_features(data=data)

    # Compute intercept
    intercept = compute_bayes_intercept(data=data)

    # Compute predictions for wine quality
    test_pred = intercept + np.dot(a=bayes_features, b=provide_weights() * compute_rescaling_factor())

    # Compute mean squared error
    bayes_risk = mean_squared_error(y_true=data['wine_quality'], y_pred=test_pred)

    return bayes_risk
