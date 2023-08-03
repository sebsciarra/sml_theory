"""Generates data according to second-order polynomial function (and an interaction term).

Data are generated in the context of a wine tasting example. In the example,
weather (:math:`\\mathbf{x}_2`) and winemaking quality (:math:`\\mathbf{x}_2`) are treated as
predictors of wine quality (:math:`\\mathbf{y}`). Note that, together, weather and winemaking
quality make up the feature space (:math:`\\mathbf{X}`). Specifically, wine quality is the outcome
of the following second-order polynomial model:

:math:`\\begin{align}
\\mathbf{y} &= \\mathbf{Xw} + \\mathbf{e}, \\text { where} \\\\
\\mathbf{X} &= [\\mathbf{x}_1, \\mathbf{x}_2, \\mathbf{x}_1^2, \\mathbf{x}_2^2, \\mathbf{x}_1\\mathbf{x}_2], \\\\
\\mathbf{w} &= [0.3 ,  0.1 ,  0.07, -0.1 ,  0.1], \\text{ and} \\\\
\\mathbf{e} &\\sim \\mathcal{N}(\\mu = 0, \\sigma = 1.5).
\\end{align}`

For more details, see whitepaper at
`sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.
"""

import numpy as np
import pandas as pd
from . import trun_mvnt as tvm


def provide_weights():
    """Returns weights of Bayes decision function.

    Returns
    --------
    weights: numpy.ndarray
        Weights of Bayes decision function.

    Examples
    --------
    >>> provide_weights()
    """

    b_weight_weather = 0.3
    b_weight_winemaking = 0.1
    b_weight_weather_squared = 0.07
    b_weight_winemaking_squared = -0.1
    b_weight_weather_winemaking = 0.1

    weights = np.array([b_weight_weather, b_weight_winemaking,
                        b_weight_weather_squared, b_weight_winemaking_squared, b_weight_weather_winemaking])

    return weights


def compute_old_scale_range():
    """Computes original scale of outcome variable.

    Data for features and outcome were generate to be on a 1--10 scale. To convert the outcome variable of
    wine quality to a 1--10 scale, the original or old range must be obtained. To do so, I calculated
    the lower and upper limits and computed the difference. For more details,see
    `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.


    Returns
    --------
    old_upper_limit - old_lower_limit: numpy.float.64
        Old range of outcome variable.

    Examples
    --------
    >>> compute_old_scale_range()
    """

    # assume highest/lowest possible error scores are +-3 SDs
    old_lower_limit = np.dot(a=provide_weights(), b=np.array([1, 10, 1, 100, 10])) - 3 * 1.5
    old_upper_limit = np.dot(a=provide_weights(), b=np.array([10, 10, 100, 100, 100])) + 3 * 1.5

    return old_upper_limit - old_lower_limit


def compute_rescaling_factor(new_upper_limit=10, new_lower_limit=1):
    """Computes rescaling factor for converting data from old to new scale.

    Rescaling factor is computed from scale of original predictors of weather and winemaking quality
    to new scale with desired lower and upper limits. Rescaling factor is computed according to following

    :math:`\\begin{align}
    \\text{Scaling factor} &= \\frac{\\text{New scale range}}{\\text{Old scale range}}
    \\end{align}`

    For more information on how the old scale range is computed, see whitepaper at
    `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

    Parameters
    --------
    new_upper_limit: int
        Upper limit of new scale.
    new_lower_limit: int
        New lower limit

    Returns
    --------
    rescaling_factor: numpy.float64
        Factor for rescaling from old scale to new scale.

    Examples
    --------
    >>> compute_rescaling_factor(new_upper_limit=5, new_lower_limit=10)
    """
    old_scale_range = compute_old_scale_range()
    new_scale_range = new_upper_limit - new_lower_limit

    rescaling_factor = new_scale_range / old_scale_range

    return rescaling_factor


def create_covariance_matrix(sd, rho):
    """Creates 2x2 covariance matrix.

    Given a vector of two standard deviation values (one for weather and one for winemaking quality),
    :math:`\\mathbf{\\sigma} = [\\mathbf{\\sigma}_1, \\mathbf{\\sigma}_2]`, and a population
    correlation value between the two variables, :math:`\\rho_{\\mathbf{x}_1, \\mathbf{x}_2}`,
    a covariance matrix is computed such that

    :math:`\\begin{align}
    \\boldsymbol{\\Sigma} &= \\begin{bmatrix}
    \\boldsymbol{\\sigma}^2_1  & \\rho_{\\mathbf{x}_1, \\mathbf{x}_2} \\boldsymbol{\\sigma}_1\\boldsymbol{\\sigma}_2 \\\\
    \\rho_{\\mathbf{x}_1, \\mathbf{x}_2} \\boldsymbol{\\sigma}_1\\boldsymbol{\\sigma}_2 & \\boldsymbol{\\sigma}^2_2
    \\end{bmatrix}
    \\end{align}`

    For more information on how the old scale range is computed, see whitepaper at
    `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.


    Parameters
    --------
    sd: list
        List containing two standard deviation values.
    rho: float
        Specifies population-level correlation between two predictor variables of weather and
        winemaking quality.

    Returns
    --------
    cov_matrix: numpy.ndarray
        Covariance matrix for predictors of weather and winemaking quality.

    Examples
    --------
    >>> mu = [5, 7]
    >>> sd = [1.2, 1.7]
    #population correlation between weather and winemaking quality
    >>> rho_weather_winemaking =  0.35
    >>> cov_matrix = create_covariance_matrix(sd = sd, rho =  rho_weather_winemaking)
    """

    # Create a lower triangular matrix with zeros
    n = len(sd)
    cov_matrix = np.zeros((n, n))

    # Fill lower and upper triangles of covariance matrix
    cov_matrix[np.tril_indices(n, -1)] = rho * np.prod(sd)
    cov_matrix = cov_matrix + cov_matrix.T

    # Fill diagonal of covariance matrix
    np.fill_diagonal(a=cov_matrix, val=sd)

    return cov_matrix


def generate_trunc_predictors(mu, cov_matrix, sample_size,
                              lower_limits=1, upper_limits=10, seed=27):
    """Generates predictor variables for weather and winemaking quality.

    Outcome variable of wine quality is rescaled by dividing the desired (or new) scale range by the
    original (or old) scale range. For more information, see whitepaper
    at `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

    Parameters
    ---------
    mu: pandas.core.frame.DataFrame
        Data set containing predictors (weather, winemaking quality) and outcome variable (wine quality).
    cov_matrix: int
        Upper limit of new scale.
    sample_size: int
        New lower limit.
    lower_limits: int
    upper_limits: int
    seed: int

    Returns
    --------
    truncated_data:  pandas.core.frame.DataFrame
        Factor for rescaling from old scale to new scale.

    Examples
    --------
    >>> compute_rescaling_factor(new_upper_limit=10, new_lower_limit=2)
    """
    # Upper and lower limits for variables
    lower_limits = np.repeat(lower_limits, len(mu))
    upper_limits = np.repeat(upper_limits, len(mu))

    # Generate samples from multivariate distribution
    sample_size = int(sample_size)
    D = np.diag(np.ones(len(mu)))
    np.random.seed(seed)

    truncated_data = pd.DataFrame(tvm.rtmvn(n=sample_size, Mean=mu, Sigma=cov_matrix,
                                            lower=lower_limits, upper=upper_limits, D=D),
                                  columns=["weather", "winemaking_quality"])

    return truncated_data


def compute_outcome_variable(data):
    """Computes outcome variable (i.e., wine quality).

    Data for outcome variable of wine quality is generated such that

    :math:`\\begin{align}
    \\mathbf{y} &= \\mathbf{Xw} + \\mathbf{e}, \\text { where} \\\\
    \\mathbf{X} &= [\\mathbf{x}_1, \\mathbf{x}_2, \\mathbf{x}_1^2, \\mathbf{x}_2^2, \\mathbf{x}_1\\mathbf{x}_2], \\\\
    \\mathbf{w} &= [0.3 ,  0.1 ,  0.07, -0.1 ,  0.1], \\text{ and} \\\\
    \\mathbf{e} &\\sim \\mathcal{N}(\\mu = 0, \\sigma = 1.5).
    \\end{align}`

    For more details,see `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Data set containing predictors (weather, winemaking quality).

    Returns
    --------
    data: pandas.core.frame.DataFrame
        Data set containing predictors (weather, winemaking quality) and outcome variable (wine quality).

    See Also
    --------
    generate_data.create_covariance_matrix()
    generate_data.generate_trunc_predictors()

    Examples
    --------
    >>> mu = [5, 7]
    >>> sd = [1.2, 1.7]
    >>> rho_weather_winemaking =  0.35
    >>> cov_matrix = generate_data.create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    >>> sample_size_gen_error = 150
    >>> sample_size_data_best_in_class = 1e4
    >>> data = generate_trunc_predictors(mu=mu, cov_matrix=cov_matrix,
                                         sample_size=sample_size, seed=27)
    >>> compute_outcome_variable(data=data)
    """

    feature_cols = pd.concat(objs=[data, data ** 2, data.prod(axis=1)], axis=1)

    # Error
    error = np.random.normal(loc=0, scale=1.5, size=data.shape[0])

    # Compute outcome variable of wine quality
    data["wine_quality"] = np.dot(a=feature_cols, b=provide_weights()) + error

    return data


def rescale_outcome_variable(data, new_lower_limit=1, new_upper_limit=10):
    """Rescales outcome variable to have scale defined by new lower and upper limits.

    Outcome variable of wine quality is rescaled by dividing the desired (or new) scale range by the
    original (or old) scale range. For more information, see whitepaper at `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

    Parameters
    --------
    data: pandas.core.frame.DataFrame
        Data set containing predictors (weather, winemaking quality) and outcome variable (wine quality).
    new_upper_limit: int
        Upper limit of new scale.
    new_lower_limit: int
        New lower limit.

    Returns
    --------
    data: pandas.core.frame.DataFrame
        Data set containing predictors (weather, winemaking quality) and rescaled outcome variable (wine quality).

    Examples
    --------
    >>> compute_rescaling_factor(new_upper_limit=10, new_lower_limit=2)
    """
    rescaling_factor = compute_rescaling_factor(new_upper_limit=new_upper_limit,
                                                new_lower_limit=new_lower_limit)
    old_lower_limit = np.dot(a=provide_weights(), b=np.array([1, 10, 1, 100, 10])) - 3 * 1.5
    lower_limit_center = (data["wine_quality"] - old_lower_limit)

    data["wine_quality"] = pd.DataFrame(data=new_lower_limit + lower_limit_center * rescaling_factor,
                                        columns=["wine_quality"])

    return data


def generate_mult_trunc_normal(cov_matrix, mu, sample_size, seed=27):
    """Generates multivariate normal truncated data.

    Data for features (weather, winemaking quality) and outcome (wine quality) are generated according
    to truncated normal distrib utions. For more details,see `
    sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

    Parameters
    ----------
    cov_matrix: pandas.core.frame.DataFrame
        Data set containing predictors (weather, winemaking quality).
    mu: list
        Specifies population-level mean for features.
    sample_size: int
        Sample sizeof generated data set.
    seed:
        Seed value of random number generator.

    Returns
    --------
    data_mult_trunc_normal: pandas.core.frame.DataFrame
        Data set containing predictors (weather, winemaking quality) and outcome variable (wine quality),
        with data for the predictors and outcome rescaled on the desired range.

    See Also
    --------
    generate_data.generate_trunc_predictors()
    generate_data.compute_outcome_variable()
    generate_data.rescale_outcome_variable
    Examples
    --------
    >>> mu = [5, 7]
    ... sd = [1.2, 1.7]
    ... rho_weather_winemaking =  0.35
    #generate covariance matrix
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #specify sample sizes for data sets
    >>> generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
    ... sample_size=150, seed=27)
    """
    # generate predictors
    data_mult_trunc_normal = generate_trunc_predictors(mu=mu, cov_matrix=cov_matrix,
                                                       sample_size=sample_size, seed=seed)

    # generate outcome variable
    data_mult_trunc_normal = compute_outcome_variable(data=data_mult_trunc_normal)

    # scale outcome variable
    data_mult_trunc_normal = rescale_outcome_variable(data=data_mult_trunc_normal)

    return data_mult_trunc_normal
