"""
Computes generalization error of optimization risk minimizer.

In the excess risk decomposition setup, a constrained set of functions must be
considered to reduce the incidence of overfitting. Because sample sizes are limited
in practice, the best possible function in the constrained set of functions (i.e.,
constrained empirical risk minimizer, :math:`f_\\mathcal{F}`) is unlikely to be obtained. Instead, an estimate
of the constrained empirical risk minimizer will result and will have a larger
generalization error than the constrained empirical risk minimizer. I call this estimate
the sample risk minimizer, :math:`\\hat{f}_s`. If no closed-form solution exists for
computing parameter estimates, then only an estimate of the sample risk minimizer will be
obtained, which I call the optimization risk minimizer, :math:`\\tilde{f}_s`.
To the extent that optimization is imperfect, the generalization error of the optimization
risk minimizer will be different from the sample risk minimizer (note that
the optimization risk minimizer can have a generalization error that is larger or smaller
than the sample risk minimizer. This module computes the generalization error of the optimization risk
minimizer. For more details, see whitepaper at `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

References
----------
[1] Bottou, L. & Bousquet, O. (2007). The tradeoffs of large scale learning. InKoller, S.
(Eds.), Advances in neural information processing systems.
(pp. 161–168). Curran Associates, Inc. `bit.ly/3qo1xpI <bit.ly/3qo1xpI>`_.
"""

import numpy as np
from smltheory.overfitting import compute_all_emp_gen_errors
from sklearn.metrics import mean_squared_error
from smltheory.gradient_descent import gradient_descent
from smltheory.least_squares import extract_feature_outcome_data


def determine_best_polynomial_model(data_sample, data_gen_error,
                                    poly_order_range=range(1, 6)):
    """Determines polynomial model with lowest generalization error.

    Parameters
    --------
    data_sample: pandas.core.frame.DataFrame
       Sample size to use for empirical data (can be conceptualized as either the training or
       validation set).

    data_gen_error: pandas.core.frame.DataFrame
        Data set used to estimate generalization of sample risk minimizer. Similar to the data
        set specified for the data_best_in_class argument, the data set here should likewise be
        large so that accurate estimate is obtained for a function's generalization error.

    poly_order_range: range
        Range of polynomial functions over which to conduct empirical risk minimization.

    Returns
    --------
    best_poly_order: numpy.int64
        Polynomial order of best model.

    See Also
    --------
    generate_data.generate_mult_trunc_normal()

    Examples
    --------
    >>> from smltheory.generate_data import create_covariance_matrix, generate_mult_trunc_normal
    >>> mu = [5, 7]
    ... sd = [1.2, 1.7]
    ... rho_weather_winemaking =  0.35
    #generate covariance matrix
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #specify sample sizes for data sets
    >>> sample_size_gen_error = 150
    ... sample_size_data_best_in_class = 1e4
    #generate data sets
    >>> data_best_in_class = generate_mult_trunc_normal(cov_matrix = cov_matrix, mu = mu,
    ... sample_size=sample_size_data_best_in_class, seed=7)
    >>> data_gen_error = generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
    ... mu=mu, sample_size=sample_size_gen_error, seed = 22
    >>> data_sample = data_best_in_class.sample(n=50, random_state=27)
    #determine best polynomial model
    >>> best_model_poly_order = determine_best_polynomial_model(data_sample=data_sample,
    ... data_gen_error=data_gen_error)
    """
    # Use random_state to ensure reproducibility and prevent resampling from adding noise to estimates
    df_all_emp_gen_errors = compute_all_emp_gen_errors(data_emp_loss=data_sample,
                                                       data_gen_error=data_gen_error,
                                                       include_interactions=False,
                                                       poly_order_range=poly_order_range)

    # Identify best polynomial model
    best_poly_order = df_all_emp_gen_errors["poly_order"][df_all_emp_gen_errors["gen_error"].idxmin()]

    return best_poly_order


def compute_opt_gen_error(opt_weights, poly_order, data_gen_error):
    """Computes generalization error of best polynomial model.

    Parameters
    --------
    opt_weights: numpy.ndarray
        Regression weights of polynomial model.

    poly_order: int
        Polynomial order of model.

    data_gen_error: pandas.core.frame.DataFrame
        Data set used to estimate generalization of sample risk minimizer. Similar to the data
        set specified for the data_best_in_class argument, the data set here should likewise be
        large so that accurate estimate is obtained for a function's generalization error.

    Returns
    --------
    gen_error: numpy.int64
        Generalization error of regression model with weights defined by `opt_weights`.

    See Also
    --------
    generate_data.generate_mult_trunc_normal()

    Examples
    --------
    >>> from smltheory.generate_data import *
    >>> mu = [5, 7]
    ... sd = [1.2, 1.7]
    ... rho_weather_winemaking =  0.35
    #generate covariance matrix
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #specify sample sizes for data sets
    >>> sample_size_gen_error = 150
    ... sample_size_data_best_in_class = 1e4
    #generate data sets
    >>> data_best_in_class = generate_mult_trunc_normal(cov_matrix = cov_matrix, mu = mu,
    ... sample_size=best_in_class_sample_size, seed=7)
    >>> data_gen_error = generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
    ... mu=mu, sample_size=sample_size_gen_error, seed = 22
    >>> data_sample = data_best_in_class.sample(n=sample_size, random_state=27)
    #determine best polynomial model
    >>> best_model_poly_order = determine_best_polynomial_model(data_sample=data_sample,
    ... data_gen_error=data_gen_error)
    #set initial guess for regression weights
    >>> opt_weights = np.random.uniform(low=0, high=1, size=2 * best_model_poly_order)
    #compute generalization error of optimization risk minimizer
    >>> compute_opt_gen_error(opt_weights=opt_weights, poly_order=poly_order,
    ... data_gen_error=data_gen_error)
    """
    # gather necessary components for matrix-matrix multiplications
    dict_data = extract_feature_outcome_data(data=data_gen_error, poly_order=poly_order)
    features = dict_data["features"]
    outcome = dict_data["outcome"].ravel()

    # compute predictions and generalization error
    predictions = np.dot(features, opt_weights)
    gen_error = mean_squared_error(y_true=outcome, y_pred=predictions)

    return gen_error


def get_opt_risk_min(sample_size, data_best_in_class, data_gen_error, num_iterations=500):
    """Computes generalization error of optimization risk minimizer.

    To obtain generalization error of optimization risk minimizer, two steps are followed:

    1. Determine which polynomial model has the best generalization error.
    2. Compute regression weights for the best polynomial model using a numerical optimization
       method of using gradient descent (instead of a closed-form solution).


    Because this function was constructed to simulate optimization error, gradient descent was
    handicapped so that a set of subtoptimal weights could be obtained with a generalization error
    that differed from the sample risk minimizer, :math:`\\hat{f}_s`.

    Parameters
    --------
    sample_size: numpy.ndarray
        Regression weights of polynomial model.

    data_best_in_class: pandas.core.frame.DataFrame
        Data set used to obtain constrained empirical risk minimizer (i.e., best possible function
        in constrained set of functions). Data set should be large (i.e., >1e4) so that an
        accurate estimate of the contrained empirical risk minimizer is obtained.

    data_gen_error: pandas.core.frame.DataFrame
        Data set used to estimate generalization of sample risk minimizer. Similar to the data
        set specified for the data_best_in_class argument, the data set here should likewise be
        large so that accurate estimate is obtained for a function's generalization error.

    num_iterations: int
        Number of iterations to use for gradient descent.

    Returns
    --------
    gen_error: numpy.int64
        Generalization error of regression model with weights defined by `opt_weights`.

    See Also
    --------
    generate_data.generate_mult_trunc_normal()

    Examples
    --------
    >>> mu = [5, 7]
    ... sd = [1.2, 1.7]
    ... rho_weather_winemaking =  0.35
    #generate covariance matrix
    >>> cov_matrix = generate_data.create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #specify sample sizes for data sets
    >>> sample_size_gen_error = 150
    ... sample_size_data_best_in_class = 1e4
    #generate data sets
    >>> data_best_in_class = generate_data.generate_mult_trunc_normal(cov_matrix = cov_matrix, mu = mu,
    ... sample_size=best_in_class_sample_size, seed=7)
    >>> data_gen_error = generate_data.generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
    ... mu=mu, sample_size=sample_size_gen_error, seed = 22
    #compute generalization error of optimization risk minimizer
    >>> get_opt_risk_min(sample_size = 50, data_best_in_class=data_best_in_class, data_gen_error=data_gen_error,
    ... num_iterations=500)
    """
    # Step 1: Using empirical risk minimization to determine the polynomial model order that results in the lowest
    # empirical loss
    data_sample = data_best_in_class.sample(n=sample_size, random_state=27)
    best_model_poly_order = determine_best_polynomial_model(data_sample=data_sample,
                                                            data_gen_error=data_gen_error)

    # Step 2: Obtain regression weights for the polynomial order model by running gradient descent.
    # I fix learning rate to 1e-5 and number of iterations to 500
    np.random.seed(27)
    initial_weights = np.random.uniform(low=0, high=1, size=2 * best_model_poly_order)

    opt_weights = gradient_descent(data=data_sample, initial_weights=initial_weights,
                                   num_iterations=num_iterations, return_log=False, learning_rate=5e-3,
                                   poly_order=best_model_poly_order, expedite_algorithm=True)

    # Step 3: Use the number of columns in features data set to generate random guesses for regression
    # weight starting values
    opt_gen_error = compute_opt_gen_error(opt_weights=opt_weights, poly_order=best_model_poly_order,
                                          data_gen_error=data_gen_error)

    return opt_gen_error
