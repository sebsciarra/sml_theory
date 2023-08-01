import numpy as np
import pandas as pd
from . import trun_mvnt as tvm


def provide_weights():

    b_weight_weather = 0.3
    b_weight_winemaking = 0.1
    b_weight_weather_squared = 0.07
    b_weight_winemaking_squared = -0.1
    b_weight_weather_winemaking = 0.1

    weights = np.array([b_weight_weather, b_weight_winemaking,
                        b_weight_weather_squared, b_weight_winemaking_squared, b_weight_weather_winemaking])

    return weights


def compute_rescaling_factor(new_upper_limit=10, new_lower_limit=1):
    old_lower_limit = np.dot(a=provide_weights(), b=np.array([1, 10, 1, 100, 10])) - 3 * 1.5
    old_upper_limit = np.dot(a=provide_weights(), b=np.array([10, 10, 100, 100, 100])) + 3 * 1.5

    # formula components
    new_scale_range = new_upper_limit - new_lower_limit
    old_scale_range = old_upper_limit - old_lower_limit

    rescaling_factor = new_scale_range / old_scale_range

    return rescaling_factor


def rescale_outcome_variable(data, new_lower_limit=1, new_upper_limit=10):
    #scale outcome variable to a 1--10 scale
    '''old_lower_limit: assume smallest scores for features with positive weights and largest scores for
    features with negative weights'''
    '''old_upper_limit: assume largest scores for features with positive weights and smallest scores for 
    features with negative weights'''
    ##assume highest/lowest possible error scores are +-4 SDs
    ##reminder: population function = ax + by + cx^2 + dy^2 + e(xy) + error
    old_lower_limit = np.dot(a=provide_weights(), b=np.array([1, 10, 1, 100, 10])) - 3 * 1.5
    old_upper_limit = np.dot(a=provide_weights(), b=np.array([10, 10, 100, 100, 100])) + 3 * 1.5

    # formula components
    lower_limit_center = (data["wine_quality"] - old_lower_limit)
    new_scale_range = new_upper_limit - new_lower_limit
    old_scale_range = old_upper_limit - old_lower_limit

    data["wine_quality"] = pd.DataFrame(data=1 + (lower_limit_center * new_scale_range) / old_scale_range,
                                        columns=["wine_quality"])

    return data


def compute_outcome_variable(data):
    # Feature columns
    # feature_cols = pd.concat(objs=[data, np.sin(data["weather"]), data["winemaking_quality"]**2,
    # data.prod(axis=1)], axis=1)
    feature_cols = pd.concat(objs=[data, data ** 2, data.prod(axis=1)], axis=1)

    # Error
    error = np.random.normal(loc=0, scale=1.5, size=data.shape[0])

    # Compute outcome variable of wine quality
    data["wine_quality"] = np.dot(a=feature_cols, b=provide_weights()) + error

    return data


def generate_trunc_predictors(mu, cov_matrix, sample_size,
                              lower_limits=1, upper_limits=10, seed=27):
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


def create_covariance_matrix(sd, rho):

    # Create a lower triangular matrix with zeros
    n = len(sd)
    cov_matrix = np.zeros((n, n))

    # Fill lower and upper triangles of covariance matrix
    cov_matrix[np.tril_indices(n, -1)] = rho * np.prod(sd)
    cov_matrix = cov_matrix + cov_matrix.T

    # Fill diagonal of covariance matrix
    np.fill_diagonal(a=cov_matrix, val=sd)

    return cov_matrix


def generate_mult_trunc_normal(cov_matrix, mu, sample_size, seed=27):
    # generate predictors
    data_mult_trunc_normal = generate_trunc_predictors(mu=mu, cov_matrix=cov_matrix,
                                                       sample_size=sample_size, seed=seed)

    # generate outcome variable
    data_mult_trunc_normal = compute_outcome_variable(data=data_mult_trunc_normal)

    # scale outcome variable
    data_mult_trunc_normal = rescale_outcome_variable(data=data_mult_trunc_normal)

    return data_mult_trunc_normal
