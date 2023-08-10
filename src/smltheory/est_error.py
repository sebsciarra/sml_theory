"""
Computes generalization error of sample risk minimizer.

In the excess risk decomposition setup, a constrained set of functions must be
considered to reduce the incidence of overfitting. Because sample sizes are limited
in practice, the best possible function in the constrained set of functions (i.e.,
constrained empirical risk minimizer, :math:`f_\\mathcal{F}`) is unlikely to be obtained. Instead, an estimate
of the constrained empirical risk minimizer will result and will have a larger
generalization error than the constrained empirical risk minimizer. I call this estimate
the sample risk minimizer, :math:`\\hat{f}_s`. This module computes the generalization error of the optimization risk
minimizer. For more details, see whitepaper at
 `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

References
----------
[1] Bottou, L. & Bousquet, O. (2007). The tradeoffs of large scale learning. InKoller, S.
(Eds.), Advances in neural information processing systems.
(pp. 161–168). Curran Associates, Inc. `bit.ly/3qo1xpI <bit.ly/3qo1xpI>`_.
"""
from smltheory.overfitting import compute_all_emp_gen_errors


def compute_sample_risk_gen_error(sample_size, data_best_in_class, data_gen_error,
                                  poly_order_range=range(1, 6)):
    """Computes generalization error of sample risk minimizer.

    Parameters
    --------
    sample_size: int
       Sample size to use for empirical data (can be conceptualized as either the training or
       validation set). Note that data of specified sample size is obtained from data set specified
       in data_best_in_class argument.

    data_best_in_class: pandas.core.frame.DataFrame
        Data set used to obtain constrained empirical risk minimizer (i.e., best possible function
        in constrained set of functions). Data set should be large (i.e., >1e4) so that an
        accurate estimate of the contrained empirical risk minimizer is obtained.

    data_gen_error: pandas.core.frame.DataFrame
        Data set used to estimate generalization of sample risk minimizer. Similar to the data
        set specified for the data_best_in_class argument, the data set here should likewise be
        large so that accurate estimate is obtained for a function's generalization error.

    poly_order_range: range
        Range of polynomial functions over which to conduct empirical risk minimization.

    Returns
    --------
    gen_errors.min(): numpy.float64
        Generalization error/risk of sample risk minimizer.

    See Also
    --------
    generate_data.generate_mult_trunc_normal()

    Examples
    --------
    >>> mu = [5, 7]
    >>> sd = [1.2, 1.7]
    >>> rho_weather_winemaking =  0.35
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    >>> sample_size_gen_error = 150
    >>> sample_size_data_best_in_class = 500
    #generate data sets
    >>> data_best_in_class = generate_mult_trunc_normal(cov_matrix = cov_matrix, mu = mu,
    ... sample_size=sample_size_data_best_in_class, seed=7)
    >>> data_gen_error = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
    ... sample_size=sample_size_gen_error, seed = 21)
    #compute generalization error of sample risk minimizer
    >>> compute_sample_risk_gen_error(sample_size=50, data_best_in_class=data_best_in_class,
    ... data_gen_error=data_gen_error)
    0.22130558411285997
"""
    # Use random_state to ensure reproducibility and prevent resampling from adding noise to estimates
    gen_errors = compute_all_emp_gen_errors(data_emp_loss=data_best_in_class.sample(n=sample_size, random_state=27),
                                            data_gen_error=data_gen_error,
                                            include_interactions=False,
                                            poly_order_range=poly_order_range)['gen_error']

    # Return generalization error of sample risk minimizer
    return gen_errors.min()
