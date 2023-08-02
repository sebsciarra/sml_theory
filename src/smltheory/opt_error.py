import numpy as np
from smltheory.overfitting import compute_all_emp_gen_errors
from sklearn.metrics import mean_squared_error
from smltheory.gradient_descent import gradient_descent
from smltheory.generate_data import generate_mult_trunc_normal
from smltheory.least_squares import compute_weight_min_mse, extract_feature_outcome_data


def determine_best_polynomial_model(data_sample, data_gen_error,
                                    poly_order_range=range(1, 6)):
    # Use random_state to ensure reproducibility and prevent resampling from adding noise to estimates
    emp_gen_errors = compute_all_emp_gen_errors(data_emp_loss=data_sample,
                                                data_gen_error=data_gen_error,
                                                include_interactions=False,
                                                poly_order_range=poly_order_range)

    return emp_gen_errors


def compute_opt_gen_error(opt_weights, poly_order, data_gen_error):
    # gather necessary components for matrix-matrix multiplications
    dict_data = extract_feature_outcome_data(data=data_gen_error, poly_order=poly_order)
    features = dict_data["features"]
    outcome = dict_data["outcome"].ravel()

    # compute predictions and generalization error
    predictions = np.dot(features, opt_weights)
    gen_error = mean_squared_error(y_true=outcome, y_pred=predictions)

    return gen_error


def get_opt_risk_min(sample_size, data_best_in_class, data_gen_error, num_iterations=500):
    # Step 1: Using empirical risk minimization to determine the polynomial model order that results in the lowest
    # empirical loss
    data_sample = data_best_in_class.sample(n=sample_size, random_state=27)
    df_all_emp_gen_errors = determine_best_polynomial_model(data_sample=data_sample,
                                                            data_gen_error=data_gen_error)

    best_model_poly_order = df_all_emp_gen_errors["poly_order"][df_all_emp_gen_errors["gen_error"].idxmin()]

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


def gen_sample_compute_weights(cov_matrix, mu, sample_size, seed, poly_order):
    # Generate sample (note that seed value is set to iteration number to ensure reproducibility)
    data_sample = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
                                             sample_size=sample_size,
                                             seed=seed)

    # compute weights using closed-form solution (w = (XX)^-1Xy)
    weights = compute_weight_min_mse(data=data_sample, poly_order=poly_order)

    return weights
