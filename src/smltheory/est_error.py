from smltheory.overfitting import compute_all_emp_gen_errors


def compute_sample_risk_gen_error(sample_size, data_best_in_class, data_gen_error,
                                  poly_order_range=range(1, 6)):
    # Use random_state to ensure reproducibility and prevent resampling from adding noise to estimates
    gen_errors = compute_all_emp_gen_errors(data_emp_loss=data_best_in_class.sample(n=sample_size, random_state=27),
                                            data_gen_error=data_gen_error,
                                            include_interactions=False,
                                            poly_order_range=poly_order_range)['gen_error']

    # Return generalization error of sample risk minimizer
    return gen_errors.min()
