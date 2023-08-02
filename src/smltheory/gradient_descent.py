import numpy as np
import pandas as pd
from smltheory.least_squares import extract_feature_outcome_data, compute_weight_min_mse


def compute_gradient(features, outcome, weights, num_samples):
    # Compute predictions (Xw using current weights
    predictions = np.dot(features, weights)

    # Compute the error (difference between predictions and target, Xw - y)
    residuals = outcome - predictions

    # Compute the gradient of the cost function with respect to weights (2X(Xw - ))
    gradient = -2 * np.dot(features.T, residuals) / num_samples

    return gradient


def gradient_descent(data, initial_weights, learning_rate=0.01, num_iterations=1000, poly_order=2,
                     epsilon=1e-20, expedite_algorithm=False, return_log=False):

    # gather necessary components for matrix-matrix multiplications
    dict_data = extract_feature_outcome_data(data=data, poly_order=poly_order)
    features = dict_data["features"]
    outcome = dict_data["outcome"].ravel()

    # Initialize weights and variable that stores all weights on each iteration
    all_weights = initial_weights  # variable necessary for logging weight values after each iteration
    weights = np.array(initial_weights)
    num_samples = len(features)

    iteration = 0
    diff = float('inf')  # Set an initial large value to ensure the loop runs at least once

    """loop proceeds until either maximum number of iterations is reaches or the difference 
    becomes less than the desired error"""
    while iteration < num_iterations and diff > epsilon:

        gradient = compute_gradient(features=features, outcome=outcome,
                                    weights=weights, num_samples=num_samples)

        # Update weights using the gradient and learning rate
        weights -= learning_rate * gradient

        # if desired, I expedite the algorithm by setting all the weights except the first one to the
        # values obtained by the closed-form solution in each iteration
        if expedite_algorithm:
            weights[1:] = compute_weight_min_mse(data=data, poly_order=poly_order)[1:]

        # update iteration number and check if difference in weights is less than epsilon
        iteration += 1
        diff = np.sum(np.square(learning_rate * gradient))

        if return_log:
            all_weights = np.vstack(tup=(all_weights, weights))

    if return_log:
        # create dataframe
        col_names = ["{}{}".format('w', str(weight_num)) for weight_num in range(1, len(all_weights) + 1)]
        df_all_weights = pd.DataFrame(data=np.array(all_weights),
                                      columns=col_names)

        # insert iteration number
        df_all_weights.insert(0, "iteration_num", df_all_weights.index)

        return df_all_weights
    else:
        return weights
