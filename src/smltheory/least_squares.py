"""Computes cross-section of mean squared error loss function.

To compute the cross-section of the mean squared error loss function, two steps are followed. First, the
closed-form solution  for the regression weights is computed (i.e., the set of regression weights that
minimizes mean squared error). Second, all the weight value from the closed form solution are used except
the first value, and the first value is varied to compute a set of mean squared error values.

For more details, see
whitepaper at `sebastiansciarra.com <https://sebastiansciarra.com/technical_content/understanding_ML>`_.
"""
import numpy as np
import pandas as pd


def extract_feature_outcome_data(data, sample_size=10, sample_data=False, poly_order=2):
    """Extracts features and outcome variables from data set.

    The features of weather and winemaking quality are extracted and the outcome variable of wine quality
    is extracted.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Data set containing features (weather, winemaking quality) and outcome (wine quality).

    sample_size: int
        If sample_data=True, then a sample of the specified sample size will be drawn randomly from the data
        set.

    sample_data: bool
        If true, only a sample of the specified sample size will be drawn from the data.

    poly_order: int
        Order of polynomial model to use when creating the set of features.

    Returns
    -------
    dict_features_outcome: dict
        Dictionary containing the features and outcome DataFrames.

    See Also
    --------
    generate_data.create_covariance_matrix()
    generate_data.generate_trunc_predictors()

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
    >>> extract_feature_outcome_data(data=data)["features"][1:5]
    array([[ 5.45999607,  4.31319487, 29.81155704, 18.60365003],
           [ 4.81312495,  8.83011447, 23.16617174, 77.97092155],
           [ 5.49988075,  6.34522617, 30.24868824, 40.26189513],
           [ 5.23554863,  8.26579874, 27.41096951, 68.32342882]])
    >>> extract_feature_outcome_data(data=data)["outcome"][1:5]
    array([6.85591702, 4.85722482, 5.17003467, 4.65399294])
    """
    if sample_data:
        data = data.sample(n=sample_size, random_state=27)

    # extract set of features and then expand according to polynomial order that is specified
    original_features = data[["weather", "winemaking_quality"]]
    features = pd.concat(objs=[original_features ** order for order in range(1, poly_order + 1)], axis=1).to_numpy()

    outcome = data[["wine_quality"]].to_numpy().ravel()

    dict_features_outcome = {"features": features, "outcome": outcome}

    return dict_features_outcome


def compute_weight_min_mse(data, poly_order=2):
    """Computes closed-form solution for regression weights (i.e., set of weights that minimize mean squared error).

    The following closed-form solution is used to compute regression weights:

    :math:`\\begin{align}
    \\mathbf{w}_{MSE} &= (\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top\\mathbf{y}.
    \\end{align}`

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Data set containing features (weather, winemaking quality) and outcome (wine quality).
    poly_order: int
        Order of polynomial model to use when creating the set of features.

    Returns
    --------
    w_mse.ravel(): numpy.ndarray
        Array that contains regression weights that correspond to closed-form solution.

    See Also
    --------
    generate_data.create_covariance_matrix()
    generate_data.generate_trunc_predictors()

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
    #compute closed-form solution regression weights
    >>> compute_weight_min_mse(data=data)
    array([ 0.85298795,  1.03699274, -0.03039457, -0.09719031])
    """
    # gather necessary components for matrix-matrix multiplications
    dict_data = extract_feature_outcome_data(data=data, poly_order=poly_order)
    features = dict_data["features"]
    outcome = dict_data["outcome"]

    # compute weight values that result from using closed-form solution
    w_mse = np.linalg.inv(np.transpose(features).dot(features)).dot(np.transpose(features)).dot(outcome)

    return w_mse.ravel()


def compute_ind_mse(data, w_guess, poly_order=2):
    """Computes individual mean squared error value.

    To compute individual mean squared error value, a set of regression weights is needed that
    corresponds to the specified polynomial order. For the later purpose of computing a cross-section of
    the mean squared error function, I fix all the weighht value except the first one to the corresponding
    closed-form solution values.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Data set containing features (weather, winemaking quality) and outcome (wine quality).
    w_guess: numpy.float64
        Weight value to use for first regression weight in computing mean squared error.
    poly_order: int
        Order of polynomial model to use when creating the set of features.

    Returns
    --------
    mse: numpy.float64
        Mean squared error value.

    See Also
    --------
    generate_data.create_covariance_matrix()
    generate_data.generate_trunc_predictors()

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
    #compute closed-form solution regression weights
    >>> compute_ind_mse(data=data, w_guess=0.34)
    1095.2514584107207
    """
    # gather necessary components for matrix-matrix multiplications
    dict_data = extract_feature_outcome_data(data=data, poly_order=poly_order)
    features = dict_data["features"]
    outcome = dict_data["outcome"]

    # compute weights that result from using closed-form solution
    w_mse = compute_weight_min_mse(data=data, poly_order=poly_order)

    # extract all the values of the closed-form solution except the first value
    w_initial = w_mse[1:]
    w_initial = np.insert(arr=w_initial, obj=0, values=w_guess)

    # compute mean squared error value (i.e., mse = yy - 2wXy + wXXw)
    yy = np.transpose(outcome).dot(outcome)
    wXy = np.transpose(w_initial).dot(np.transpose(features)).dot(outcome)
    wXXw = np.transpose(w_initial).dot(np.transpose(features)).dot(features).dot(w_initial)

    mse = yy - 2 * wXy + wXXw

    return mse


def compute_all_mse_loss_values(data, w_guess_list):
    """Computes cross-section of mean squared error function.

    To compute a cross-section, all but one weight value is fixed. In this case, the first weight value is allowed
    to vary and is set to the value of `w_guess`. All the other weight values are fixed to the values of the
    closed-form solution.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Data set containing features (weather, winemaking quality) and outcome (wine quality).
    w_guess_list: numpy.ndarray
        Set of weight values over which to compute cross-section of mean squared error function.

    Returns
    --------
    df_mse: numpy.float64
        Mean squared error value.

    See Also
    --------
    generate_data.create_covariance_matrix()
    generate_data.generate_trunc_predictors()

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
    #compute closed-form solution regression weights
    >>> w_guess_list = np.arange(start = 0, stop = 3, step= 0.01)
    ... compute_all_mse_loss_values(data=data, w_guess_list=w_guess_list)
          w_guess     mse_value
    0       0.00   2953.803317
    1       0.01   2885.934168
    2       0.02   2818.865374
    3       0.03   2752.596934
    4       0.04   2687.128850
    ..       ...           ...
    295     2.95  17639.803853
    296     2.96  17808.039463
    297     2.97  17977.075428
    298     2.98  18146.911748
    299     2.99  18317.548423

    [300 rows x 2 columns]
    """
    # return flattened array
    all_mse_loss_values = np.array([compute_ind_mse(data, guess) for guess in w_guess_list])

    # create dataframe with guesses and corresponding MSE values
    df_mse = pd.DataFrame({"w_guess": w_guess_list,
                           "mse_value": all_mse_loss_values})

    return df_mse
