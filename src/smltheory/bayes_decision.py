"""Computes generalization error of Bayes decision function.

To provide context, Bayes decision function predicts wine quality using weather and
winemaking quality such that

:math:`\\begin{align}
\\mathbf{y} &= \\mathbf{Xw}, \\text { where} \\\\
\\mathbf{X} &= [\\mathbf{x}_1, \\mathbf{x}_2, \\mathbf{x}_1^2, \\mathbf{x}_2^2, \\mathbf{x}_1\\mathbf{x}_2], \\\\
\\mathbf{w} &= [0.3 ,  0.1 ,  0.07, -0.1 ,  0.1].
\\end{align}`

For more details, see whitepaper at
`sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.
"""

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
     generate_data.create_covariance_matrix()
     generate_data.generate_mult_trunc_normal()

     Examples
     ---------
     >>> mu=[5, 7]
     >>> sd=[1.2, 1.7]
     >>> rho_weather_winemaking =  0.35
     >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
     #generate data set
     >>> sample_size_gen_error = 150
     >>> data_gen_error = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
     ... sample_size=sample_size_gen_error)
     #compute features for Bayes decision function
     >>> compute_bayes_features(data=data_gen_error)
          weather  winemaking_quality    weather  winemaking_quality          0
     0    5.214258            6.655854  27.188488           44.300390  34.705340
     1    5.459996            4.313195  29.811557           18.603650  23.550027
     2    4.813125            8.830114  23.166172           77.970922  42.500444
     3    5.499881            6.345226  30.248688           40.261895  34.897987
     4    5.235549            8.265799  27.410970           68.323429  43.275991
     ..        ...                 ...        ...                 ...        ...
     145  6.020267            7.603160  36.243618           57.808035  45.773053
     146  3.982080            7.635946  15.856965           58.307675  30.406952
     147  5.654110            9.818515  31.968960           96.403227  55.514961
     148  8.112048            6.618283  65.805322           43.801670  53.687829
     149  3.384190            6.122903  11.452739           37.489943  20.721065
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
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #generate data set
    >>> sample_size_gen_error = 150
    >>> data_gen_error = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
    ... sample_size=sample_size_gen_error)
    #compute intercept of Bayes decision function
    >>> compute_bayes_intercept(data=data_gen_error)
    4.977118063990485
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
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #generate data
    >>> sample_size_gen_error = 150
    >>> data_gen_error = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
    ... sample_size=sample_size_gen_error)
    >>> compute_bayes_risk(data=data_gen_error)
    0.25105984029122547
    """

    bayes_features = generate_bayes_features(data=data)

    # Compute intercept
    intercept = compute_bayes_intercept(data=data)

    # Compute predictions for wine quality
    test_pred = intercept + np.dot(a=bayes_features, b=provide_weights() * compute_rescaling_factor())

    # Compute mean squared error
    bayes_risk = mean_squared_error(y_true=data['wine_quality'], y_pred=test_pred)

    return bayes_risk
