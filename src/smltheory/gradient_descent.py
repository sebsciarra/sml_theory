"""Runs gradient descent algorithm.

The gradient descent algorithm in this module works by updating an initial set of guesses, :math:`\\mathbf{w}^{(0)}`,
of regression weights and then returning a final set of weights after a specified number of iterations is
reached, :math:`k = K`, or the sum of squared differences between weight values of subsequent iterations is
below some threshold, :math:`\\lVert \\mathbf{w}^{(k)} - \\mathbf{w}^{(k+1)}  \\rVert^2_2 < \\epsilon`.

Note that the mean squared error loss function shown below is used

:math:`\\begin{align}
\\ell_{MSE} &= \\frac{1}{n} \\lVert \\mathbf{y} - \\mathbf{Xw} \\rVert^2_2,
\\end{align}`

and the gradient of the mean squared error loss function, :math:`\\nabla_w \\ell_{MSE}` is computed with respect
to the weights such that

:math:`\\begin{align}
\\nabla_w \\ell_{MSE} &= \\frac{1}{n} \\bigg(2\\mathbf{X}^\\top(\\underbrace{\\mathbf{Xw} -
\\mathbf{y}}_{\\text{Residual}})\\bigg).
\\end{align}`.

Gradient descent works by updating the initial set of weights by stepping in the negative gradient direction
according a set step size (or learning rate, :math:`\\tau`) such that

:math:`\\begin{align}
\\mathbf{w}^{(k+1)} = \\mathbf{w}^{(k)} - \\tau \\nabla_{\\mathbf{w}}\\ell_{MSE}(\\mathbf{w}^{(k)})
\\end{align}`.

Also note that the gradient is a function of the residual, :math:`(\\mathbf{Xw} - \\mathbf{y})`. For more details, see
whitepaper at `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.
"""

import numpy as np
import pandas as pd
from smltheory.least_squares import extract_feature_outcome_data, compute_weight_min_mse


def compute_gradient(features, outcome, weights, num_samples):
    """Computes gradient of mean squared error loss function.

    For more details,see `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

    Parameters
    ----------
    features: pandas.core.frame.DataFrame
        Feature used to predict wine quality (weather and winemaking quality).
    outcome:  pandas.core.frame.DataFrame
        Data containing wine quality values.
    weights: numpy.ndarray
        Set of weight values used to compute predictions
    num_samples: int
        Number of data points in either the feature/outcome data sets.

    Returns
    --------
    gradient: pandas.core.frame.DataFrame
        Gradient of mean squared error loss function with respect to regression weights.

    See Also
    --------
    generate_data.create_covariance_matrix()
    generate_data.generate_trunc_predictors()
    least_squares.extract_feature_outcome_data()

    Examples
    --------
    >>> mu = [5, 7]
    ... sd = [1.2, 1.7]
    ... rho_weather_winemaking =  0.35
    #generate covariance matrix
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #generate data
    >>> data = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
    ... sample_size=150, seed=27)
    #extract features and outcome
    >>> features = smltheory.least_squares.extract_feature_outcome_data(data=data)["features"]
    ... outcome = smltheory.least_squares.extract_feature_outcome_data(data=data)["outcome"]
    #compute gradient
    >>> weights = np.random.uniform(low = 0, high = 1, size = 4)
    >>> compute_gradient(features=features, outcome=outcome, weights=weights,
    ... num_samples=len(features))
    array([ 691.91945491,  959.77824399, 3812.79874487, 7289.85051885])
    """
    # Compute predictions (Xw using current weights
    predictions = np.dot(features, weights)

    # Compute the error (difference between predictions and target, Xw - y)
    residuals = outcome.ravel() - predictions

    # Compute the gradient of the cost function with respect to weights (2X(Xw - ))
    gradient = -2 * np.dot(features.T, residuals) / num_samples

    return gradient

def gradient_descent(data, initial_weights, learning_rate=0.01, num_iterations=1000, poly_order=2,
                     epsilon=1e-20, expedite_algorithm=False, return_log=False):
    """Runs gradient descent algorithm.

    For a description, see module description at top of page. For complete details see
    `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Data set containing features (weather, winemaking quality) and outcome (wine quality).

    initial_weights:  numpy.ndarray
        Array specifiying initial weights.

    learning_rate: float
        Rate at which parameters are updated in each step (i.e., step size parameter)

    num_iterations: int
        Maximum number of iterations to run in gradient descent algorithm.

    poly_order: int
        Order of polynomial order to use in computing gradient descent.

    epsilon: float
        Threshold value set for the difference in the sum of squared differences between weights  between
        subsequent iterations.

    expedite_algorithm: bool
        If True, the gradient descent algorithm will be expedited by replacing all the weight values except the
        first one with the closed-form solution values.

    return_log: bool
        If True, a pandas.DataFrame will be returned that contains the set of weight obtained after each iteration.

    Returns
    --------
    weights: numpy.ndarray
        Set of weights obtained after running gradient descent algorithm.

    See Also
    --------
    generate_data.create_covariance_matrix()
    generate_data.generate_trunc_predictors()
    least_squares.extract_feature_outcome_data()

    Examples
    --------
    >>> mu = [5, 7]
    ... sd = [1.2, 1.7]
    ... rho_weather_winemaking =  0.35
    #generate covariance matrix
    >>> cov_matrix = create_covariance_matrix(sd=sd, rho=rho_weather_winemaking)
    #generate data
    >>> data = generate_mult_trunc_normal(cov_matrix=cov_matrix, mu=mu,
    ... sample_size=150, seed=27)
    #set weights
    >>> np.random.seed(27) #ensure reproducibility
    ... initial_weights = np.random.uniform(low = 0, high = 1, size = 3)
    #run gradient descent algorithm (with no log returned)
    >>> gradient_descent(data=data, initial_weights=initial_weights,  num_iterations=1100,
    ... learning_rate=0.0001, epsilon=1e-20, expedite_algorithm=True, return_log=False)
    array([ 0.85159995,  1.03699274, -0.03039457, -0.09719031])
    #run gradient descent algorithm (with log returned)
    >>> gradient_descent(data=data, initial_weights=initial_weights,  num_iterations=1100,
    ... learning_rate=0.0001, epsilon=1e-20, expedite_algorithm=True, return_log=True)
    iteration_num        w1        w2        w3        w4
    0                 0  0.000000  0.425721  0.814584  0.735397
    1                 1 -0.059489  1.036993 -0.030395 -0.097190
    2                 2 -0.054620  1.036993 -0.030395 -0.097190
    3                 3 -0.049778  1.036993 -0.030395 -0.097190
    4                 4 -0.044961  1.036993 -0.030395 -0.097190
    ...             ...       ...       ...       ...       ...
    1096           1096  0.850382  1.036993 -0.030395 -0.097190
    1097           1097  0.850396  1.036993 -0.030395 -0.097190
    1098           1098  0.850409  1.036993 -0.030395 -0.097190
    1099           1099  0.850423  1.036993 -0.030395 -0.097190
    1100           1100  0.850437  1.036993 -0.030395 -0.097190
    """
    # gather features and outcome data in different sets for later computations
    dict_data = extract_feature_outcome_data(data=data, poly_order=poly_order)
    features = dict_data["features"]
    outcome = dict_data["outcome"]

    # Initialize weights and variable that stores all weights on each iteration
    all_weights = initial_weights  # variable necessary for logging weight values after each iteration
    weights = np.array(initial_weights)
    num_samples = len(features)

    iteration = 0
    diff = float('inf')  # Set an initial large value to ensure the loop runs at least once

    # loop proceeds until either maximum number of iterations is reaches or the difference
    # becomes less than the desired error"""
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
