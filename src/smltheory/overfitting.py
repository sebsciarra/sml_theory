import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from functools import partial
import pandas as pd


def compute_emp_gen_error(equation, data_emp_loss, data_gen_error):
    model = smf.ols(data=data_emp_loss, formula=equation).fit()
    y_test = model.predict(data_gen_error)

    emp_loss = mean_squared_error(y_true=data_emp_loss['wine_quality'], y_pred=model.fittedvalues)
    gen_error = mean_squared_error(y_true=data_gen_error['wine_quality'], y_pred=y_test)

    return {'emp_loss': emp_loss,
            'gen_error': gen_error}


def gen_poly_reg_eq(poly_order, include_interactions=False):
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


def compute_all_emp_gen_errors(data_emp_loss, data_gen_error, poly_order_range, include_interactions=False):
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
