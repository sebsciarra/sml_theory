import numpy as np
import pandas as pd


def extract_feature_outcome_data(data, sample_size=10, sample_data=False, poly_order=2):
    if sample_data:
        data = data.sample(n=sample_size, random_state=27)

    # gather necessary components for matrix-matrix multiplications
    original_features = data[["weather", "winemaking_quality"]]
    features = pd.concat(objs=[original_features ** order for order in range(1, poly_order + 1)], axis=1).to_numpy()

    outcome = data[["wine_quality"]].to_numpy()

    return {"features": features,
            "outcome": outcome}


def compute_weight_min_mse(data, poly_order=2):
    # gather necessary components for matrix-matrix multiplications
    dict_data = extract_feature_outcome_data(data=data, poly_order=poly_order)
    features = dict_data["features"]
    outcome = dict_data["outcome"]

    # compute w_MSE =
    w_mse = np.linalg.inv(np.transpose(features).dot(features)).dot(np.transpose(features)).dot(outcome)

    return w_mse.ravel()


def compute_ind_mse(data, w_guess, poly_order=2):
    # gather necessary components for matrix-matrix multiplications
    dict_data = extract_feature_outcome_data(data=data, poly_order=poly_order)
    features = dict_data["features"]
    outcome = dict_data["outcome"]

    # construct vector of weights
    # in w_initial, the three weight values come from the least mean squares solution.
    w_initial = np.array([1.05928459, -0.01280733, -0.09713313])
    w_initial = np.insert(arr=w_initial, obj=0, values=w_guess)

    # MSE = (XX)^-1Xy
    yy = np.transpose(outcome).dot(outcome)
    wXy = np.transpose(w_initial).dot(np.transpose(features)).dot(outcome)
    wXXw = np.transpose(w_initial).dot(np.transpose(features)).dot(features).dot(w_initial)

    mse = yy - 2 * wXy + wXXw

    return mse


def compute_all_mse_loss_values(data, w_guess_list):
    # return flattened array
    all_mse_loss_values = np.concatenate([compute_ind_mse(data, guess) for guess in w_guess_list]).ravel()

    # create dataframe with guesses and corresponding MSE values
    df_mse = pd.DataFrame({"w_guess": w_guess_list,
                           "mse_value": all_mse_loss_values})

    return df_mse
