"""Gets data set(s) necessary for demonstrating supervised machine learning proposition.

Six data sets are used in this package to demonstrate supervised machine learning propositions:

1) `data_emp_loss`: data set for computing empirical loss.
2) `data_gen_error`: data set for computing generalization error.
3) `data_best_in_class`: data set for determining best function in constrained set (or class) of functions.
4) `data_est_error`: data set for depicting estimation error.
5) `data_opt_error`: data set for depicting optimization error.
6) `data_bv_tradeoff`: data set for depicting bias-variance tradeoff.

For more details, see
whitepaper at `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.
"""

from importlib import resources
import pandas as pd


def get_data_emp_loss():
    """Returns data set for computing empirical loss  (i.e., "data_emp_loss.csv").

    Returns
    -------
    data_emp_loss:  pandas.core.frame.DataFrame
        Data set for computing empirical loss.

    Notes
    -----
    Data set for computing empirical loss was generated using the following code:

    .. code-block:: python

        sd = [1.5, 1.2]
        mu = [5, 7]
        sd = [1.2, 1.7]
        rho_weather_winemaking = 0.35
        cov_matrix = generate_data.create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
        # generate data
        sample_size_emp_loss = 150
        data_emp_loss = generate_data.generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                        mu=mu, sample_size=sample_size_emp_loss, seed=17)

    """
    with resources.path("smltheory.data", "data_emp_loss.csv") as path:
        data_emp_loss = pd.read_csv(path)

    return data_emp_loss


def get_data_gen_error():
    """Returns data set for computing generalization error (i.e., "data_gen_error.csv").

    Returns
    -------
    data_gen_error:  pandas.core.frame.DataFrame
        Data set for computing generalization error.

    Notes
    -----
    Data set for computing generalization error was generated using the following code:

    .. code-block:: python

        sd = [1.5, 1.2]
        mu = [5, 7]
        sd = [1.2, 1.7]
        rho_weather_winemaking = 0.35
        cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
        # generate data
        sample_size_gen_error= 1e4
        data_gen_error = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
                         sample_size=sample_size_gen_error, seed = 21)

    """
    with resources.path("smltheory.data", "data_gen_error.csv") as path:
        data_gen_error = pd.read_csv(path)

    return data_gen_error


def get_data_best_in_class():
    """Returns data set for obtaining the constrained empirical risk minimizer (i.e., best function in 
    constrained set of functions). 

    Returns
    -------
    data_gen_error:  pandas.core.frame.DataFrame
        Data set for computing generalization error.

    Notes
    -----
    Data set for obtaining the constrained empirical risk minimizer (i.e., best function in
    constrained set/class of functions) was generated using the following code:

    .. code-block:: python

        sd = [1.5, 1.2]
        mu = [5, 7]
        sd = [1.2, 1.7]
        rho_weather_winemaking = 0.35
        cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
        # generate data
        sample_size_best_in_class = 1e4
        data_best_in_class = generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                             mu=mu, sample_size=sample_size_best_in_class, seed = 7)

    """
    with resources.path("smltheory.data", "data_best_in_class.csv") as path:
        data_best_in_class = pd.read_csv(path)

    return data_best_in_class


def get_data_est_error():
    """Returns data set for depicting estimation error.

    Returns
    -------
    data_est_error:  pandas.core.frame.DataFrame
        Data set for depicting estimation error

    Notes
    -----
    Data set for depicting estimation error was generated using the following code:

    .. code-block:: python

        sd = [1.5, 1.2]
        mu = [5, 7]
        sd = [1.2, 1.7]
        rho_weather_winemaking = 0.35
        cov_matrix = generate_data.create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)

        #sample sizes
        sample_size_best_in_class, sample_size_gen_error = 1e4

        #generate data sets
        data_gen_error = generate_data.generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                         mu=mu, sample_size=sample_size_best_in_class, seed = 21)
        data_best_in_class = generate_data.generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                             mu=mu, sample_size=sample_size_best_in_class, seed = 7)

        #fix the arguments except for sample_size and then use map to vectorize over sample size values
        compute_sample_risk_gen_error_partial = functools.partial(compute_sample_risk_gen_error,
                                                data_best_in_class = data_best_in_class,
                                                data_gen_error = data_gen_error,
                                                poly_order_range = range(1, 5))

        #compute estimation error for each sample size between 5 and 1000 (inclusive)
        # Call the partial function with est_sample_sizes
        est_sample_sizes = range(5, 1000)
        est_gen_errors = list(map(compute_sample_risk_gen_error_partial, est_sample_sizes))

        #Create data frame for estimation error
        pd_est_error = pd.DataFrame({"sample_size": np.array(est_sample_sizes),
                                     "sample_risk_gen_error": est_gen_errors})

    """
    with resources.path("smltheory.data", "data_est_error.csv") as path:
        data_est_error = pd.read_csv(path)

    return data_est_error


def get_data_opt_error():
    """Returns data set for depicting optimization error.

    Returns
    -------
    data_opt_error:  pandas.core.frame.DataFrame
        Data set for depicting optimization error

    Notes
    -----
   Data set for depicting optimization error was generated using the following code:

    .. code-block:: python

        sd = [1.5, 1.2]
        mu = [5, 7]
        sd = [1.2, 1.7]
        rho_weather_winemaking = 0.35
        cov_matrix = generate_data.create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)

        #sample sizes
        sample_size_best_in_class, sample_size_gen_error = 1e4

        #generate data sets
        data_gen_error = generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                         mu=mu, sample_size=sample_size_best_in_class, seed = 21)
        data_best_in_class = generate_mult_trunc_normal(cov_matrix=cov_matrix, sd=sd,
                             mu=mu, sample_size=sample_size_best_in_class, seed = 7)

        #fix the arguments except for sample_size and then use map to vectorize over sample size values
        get_opt_risk_min_partial = functools.partial(get_opt_risk_min,
                                                     data_best_in_class = data_best_in_class,
                                                     data_gen_error = data_gen_error)

        # Call the partial function with each sample size specified in opt_sample_sizes
        opt_sample_sizes = range(5, 1000)
        with concurrent.futures.ThreadPoolExecutor(max_workers = 3) as executor:
            opt_gen_errors = list(executor.map(get_opt_risk_min_partial, opt_sample_sizes))

        #Create data frame
        df_opt_error = pd.DataFrame({"sample_size": np.array(opt_sample_sizes),
                                     "opt_gen_error": opt_gen_errors})

    """
    with resources.path("smltheory.data", "data_opt_error_fast.csv") as path:
        data_opt_error = pd.read_csv(path)

    return data_opt_error


def get_data_bv_tradeoff():
    """Returns data set for depicting bias-variance tradeoff.

    Returns
    -------
    data_bv_tradeoff:  pandas.core.frame.DataFrame
        Data set for depicting bias-variance tradeoff.

    Notes
    -----
    Given that code for computing this data set was more complicated, see see
    whitepaper at `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.
    """
    with resources.path("smltheory.data", "data_bv_tradeoff.csv") as path:
        data_bv_tradeoff = pd.read_csv(path)

    return data_bv_tradeoff
