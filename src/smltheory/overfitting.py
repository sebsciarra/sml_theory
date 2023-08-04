"""Depicts how overfitting results from using complex polynomial models.

Empirical loss and generalization errors are computed for set of polynomial models in a desired range. As
the polynomial order increases, empirical loss remains relatively unchanged, but generalization increases
exponentially. Thus, although more complex models perform well on training data, their actual performance is
considerably worse.

For more details, see
whitepaper at `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.
"""

import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from functools import partial
import pandas as pd


def gen_poly_reg_eq(poly_order, include_interactions=False):
    """Generates equation for polynomial model of specified order.

    Parameters
    ----------
    poly_order: int
        Order of polynomial model to use when creating the set of features.
    include_interactions: bool
        If True, then all possible interactions are included in model.

    Returns
    -------
    equation: str
        String specifying polynomial model.

    Examples
    --------
    >>> gen_poly_reg_eq(poly_order=2)
    'wine_quality ~ np.power(weather, 1) + np.power(weather, 2) + np.power(winemaking_quality, 1) + np.power(winemaking_quality, 2)'
    """
    # compute polynomial terms for predictors
    weather_pred = ' + '.join(['np.power(weather, {})'.format(ord_value) for ord_value in range(1, poly_order + 1)])
    winemaking_quality_pred = ' + '.join(['np.power(winemaking_quality, {})'.format(ord_value)
                                          for ord_value in range(1, poly_order + 1)])

    # Compute all two-way interactions between weather and winemaking_quality
    if include_interactions and poly_order > 1:

        interaction_terms = ' + '.join(
            ['np.multiply(np.power(weather, {}), np.power(winemaking_quality, {}))'.format(p1, p2)
             for p1 in range(1, poly_order)
             for p2 in range(1, poly_order)])

        predictors = ' + '.join([weather_pred, winemaking_quality_pred, interaction_terms])

    else:
        predictors = ' + '.join([weather_pred, winemaking_quality_pred])

    # create regression equation
    equation = "wine_quality ~ {}".format(predictors)

    return equation

def compute_emp_gen_error(equation, data_emp_loss, data_gen_error):
    """Computes empirical loss and generalization error for a polynomial model.

    Parameters
    ----------
    equation: str
        String specifying polynomial model.

    data_emp_loss: pandas.core.frame.DataFrame
        Data set for computing empirical loss.

    data_gen_error: pandas.core.frame.DataFrame
        Data set for computing generalization error.

    Returns
    -------
    emp_loss_gen_error_dict: dict
        Dictionary containing empirical loss and generalization error.

    Examples
    --------
    >>> mu = [5, 7]
    ... sd = [1.2, 1.7]
    ... rho_weather_winemaking =  0.35
    #generate covariance matrix
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #generate data
    >>> data_emp_loss = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
    ... sample_size=150, seed=27)
    #sample sizes
    >>> sample_size_emp_loss = 150
    ... sample_size_gen_error = 1e4
    #generate data set for empirical loss
    >>> data_emp_loss = generate_mult_trunc_normal(cov_matrix = cov_matrix, mu = mu,
    ... sample_size = sample_size_emp_loss)
    >>> data_gen_error = generate_mult_trunc_normal(cov_matrix = cov_matrix, mu = mu,
    ... sample_size = sample_size_gen_error)
    #equation
    >>> equation = gen_poly_reg_eq(poly_order=2)
    >>> compute_emp_gen_error(equation=equation, data_emp_loss=data_emp_loss,
    ... data_gen_error=data_gen_error)
    {'emp_loss': 0.2420572087052563, 'gen_error': 0.25353431587715874}
    """
    model = smf.ols(data=data_emp_loss, formula=equation).fit()
    y_test = model.predict(data_gen_error)

    emp_loss = mean_squared_error(y_true=data_emp_loss['wine_quality'], y_pred=model.fittedvalues)
    gen_error = mean_squared_error(y_true=data_gen_error['wine_quality'], y_pred=y_test)

    emp_loss_gen_error_dict = {'emp_loss': emp_loss, 'gen_error': gen_error}

    return emp_loss_gen_error_dict


def compute_all_emp_gen_errors(data_emp_loss, data_gen_error, poly_order_range, include_interactions=False):
    """Computes empirical loss and generalization error for a polynomial model.

    Parameters
    ----------
    data_emp_loss: pandas.core.frame.DataFrame
        Data set for computing empirical loss.

    data_gen_error: pandas.core.frame.DataFrame
        Data set for computing generalization error.

    poly_order_range: range
        Specifies range of polynomial models to use.

    include_interactions: bool
        If True, polynomial models contain all the corresponding interactions.

    Returns
    -------
    df_emp_gen_errors: pandas.core.frame.DataFrame
        DataFrame containing the empirical loss and generalization error of each polynomial model.

    Examples
    --------
    >>> mu = [5, 7]
    ... sd = [1.2, 1.7]
    ... rho_weather_winemaking =  0.35
    #generate covariance matrix
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #generate data
    >>> data_emp_loss = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
    ... sample_size=150, seed=27)
    #sample sizes
    >>> sample_size_emp_loss = 150
    ... sample_size_gen_error = 1e4
    #generate data set for empirical loss
    >>> data_emp_loss = generate_mult_trunc_normal(cov_matrix = cov_matrix, mu = mu,
    ... sample_size = sample_size_emp_loss)
    >>> data_gen_error = generate_mult_trunc_normal(cov_matrix = cov_matrix, mu = mu,
    ... sample_size = sample_size_gen_error)
    #compute empirical loss and generalization error for each polynomial model
    >>>  compute_all_emp_gen_errors(data_emp_loss = data_emp_loss,
    ... data_gen_error = data_gen_error, include_interactions = True,
    ... poly_order_range=range(1, 10))
        poly_order  emp_loss     gen_error
    0           1  0.244933      0.254885
    1           2  0.242056      0.253487
    2           3  0.235921      0.263245
    3           4  0.219007      0.339122
    4           5  0.202056      2.038340
    5           6  0.184981    537.737954
    6           7  0.177600   6801.505982
    7           8  0.176475  11862.620947
    8           9  0.176777  14885.917939
    """
    # create all polynomial equations within the desired range
    gen_poly_reg_eq_partial = partial(gen_poly_reg_eq, include_interactions=include_interactions)

    poly_equations = list(map(gen_poly_reg_eq_partial, poly_order_range))

    # create partial version of function with data_emp_loss and data_gen_error fixed
    emp_gen_error_partial = partial(compute_emp_gen_error, data_emp_loss=data_emp_loss,
                                    data_gen_error=data_gen_error)

    # for each polynomial equation, compute empirical loss and generalization error
    all_emp_gen_errors = list(map(emp_gen_error_partial, poly_equations))

    # convert dictionary to dataframe and then compute polynomial orders by using row indexes
    df_emp_gen_errors = pd.DataFrame(all_emp_gen_errors)
    poly_orders = pd.Series([poly_order for poly_order in poly_order_range], name="poly_order")

    # concatenate poly_orders and dataframe to create complete dataframe
    df_emp_gen_errors = pd.concat([poly_orders, df_emp_gen_errors], axis=1)

    return df_emp_gen_errors
