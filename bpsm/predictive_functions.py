from typing import List, Optional

import numpy as np
import pandas as pd
from lifelines.utils import qth_survival_times


def reconstruct_hazard_time_points(
    interval_hazard: np.array, interval_length: int, n_intervals: int
):
    """
    Reconstruct hazard time points if more than one time points are considered within
    piecewise intervals.

    Parameters
    ----------
    interval_hazard : np.array
        array with posterior values of hazard
    interval_length: int
        length per piecewise interval
    n_intervals: int
        number of intervals that were constructed.

    Returns
    -------
    numpy.array
        recontructed hazard function.

    """
    return np.hstack(
        [
            np.repeat(interval_hazard[:, n], interval_length).reshape(
                (interval_hazard.shape[0], interval_length)
            )
            for n in range(n_intervals)
        ]
    )


def get_cumulative_hazard(
    hazard: np.array,
    interval_length: Optional[int] = None,
    n_intervals: Optional[int] = None,
):
    """
    This function calculates the
    cumulative hazard function by each piecewise
    interval.

    Parameters
    ----------
    hazard: np.array
        hazard function at each interval
    interval_length: length of the interval

    Returns
    -------
        pd.Series
            cumulative hazard function
    """
    # for the time points within each interval we have the same hazard
    # cumulatively this is just a multiplication within the interval.
    if interval_length == 1:
        return (interval_length * hazard).cumsum(axis=-1)
    else:
        return reconstruct_hazard_time_points(
            hazard, interval_length, n_intervals
        ).cumsum(axis=-1)


def get_baseline_cumulative_survival(
    hazard: np.array, interval_length: int, n_intervals: Optional[int] = None
):
    """
    This function calculates the baseline
    cumulative survival function, i.e.,
    S0(t) = exp(-H0(t)).

    Parameters
    ----------
    hazard: np.array
        hazard function for each interval
    interval_length: int
        interval

    Returns
    -------
        pd.Series
            baseline cumulative survival function
    """
    return np.exp(-get_cumulative_hazard(hazard, interval_length, n_intervals))


def get_beta_coefficients(
    trace: np.array, feature_names: List[str], hpd_alpha: float = 0.05
):
    """
    This function accepts the beta coefficients posterior draws
    and returns a pandas dataframe, their median and HPD
    interval and their exponents.

    Parameters
    ----------
    trace: np.array
        trace of posterior draws from pymc
    feature_names: list
        feature names corresponding to
        each coefficient
    hpd_alpha: float
        float representing the quantile level
        of getting the HPD interval, i.e.,
        extracting the [hpd_alpha/2, 1-hpd_alpha/2]
        quantiles of the posterior draws.

    Returns
    -------
        betas_draws, summary_vals, exp_vals
    """
    # pymc draws are a 3-dimensional array, reshaping to two dimensions
    # for the coefficients
    betas_draws = pd.DataFrame(trace)
    betas_draws.columns = feature_names

    # Calculate the percentiles
    percentiles = np.array([hpd_alpha / 2.0, 0.5, 1.0 - hpd_alpha / 2.0])

    # Calculate percentiles of the posterior draws
    summary_vals = betas_draws.quantile(percentiles, axis=0).T
    summary_vals.columns = [
        f"{100 * percentiles[0]}%",
        f"{100 * percentiles[1]}%",
        f"{100 * percentiles[2]}%",
    ]

    # Calculate the exponents
    exp_vals = np.exp(summary_vals)

    return betas_draws, summary_vals, exp_vals


def predict_time_to_event(
    survival_probas: np.array, id_col: str, survival_probability_threshold: float
):
    """
    Predicts time to event with hpd intervals.

    Parameters
    ----------
    survival_probas: pd.DataFrame
        DataFrame of all survival probabilities per user per time point.
    id_col: key indicating user index
    survival_probability_threshold: float
    float between 0 and 1 indicating the probability threshold
    to mark the event. Default is set to 0.5.

    Returns
    -------
    pd.DataFrame
    """
    return (
        survival_probas.groupby(id_col)
        .survival_probability.apply(
            lambda t: qth_survival_times(survival_probability_threshold, list(t)) + 1
        )
        .rename("median_survival_time")
    )


def predict_partial_hazard_by_time(betas: pd.DataFrame, x: pd.DataFrame):
    """
    This function calculates the partial hazard
    at one point by using the estimated
    coefficients and the feature matrix X, i.e.,
    exp(X*betas^T).
    Parameters
    ----------
    betas: pd.DataFrame
        DataFrame with vector of dimension 1xp of estimated coefficients

    x: pd.DataFrame
        DataFrame of feature matrix of dimension Nxp

    Returns
    -------
        pd.DataFrame
        of partial hazard with Nxp dimension.
    """
    return np.exp(np.matmul(x, betas.T))


def predict_hazard_by_time(
    cumulative_hazard: np.array, betas: pd.DataFrame, x: pd.DataFrame
):
    """
    This function calculates the hazard function, survival probabilities
    and hazard probabilities for all time
    points by accepting the estimated baseline hazard,
    the estimated beta coefficients and the feature matrix X.

    Parameters
    ----------
    cumulative_hazard: np.array
        estimated baseline hazard for interval j, i.e., h0(j)
    betas: pd.DataFrame
        DataFrame with vector of dimension 1xp of estimated coefficients
    x: pd.DataFrame
        DataFrame of feature matrix of dimension Nxp
    interval_length: int
        length of the piecewise intervals

    Returns
    -------
        hazard_func, survival_proba, hazard_proba
    """

    # Calculate partial hazard for each MCMC draw of betas
    partial_hazards = predict_partial_hazard_by_time(betas, x)

    # Initialise the matrices
    survival_proba = np.zeros(
        (x.shape[0], betas.shape[0], cumulative_hazard.shape[1] - 1)
    )

    # Calculate hazard/survival function for all time points and subjects by draw
    for i in range(cumulative_hazard.shape[1] - 1):
        survival_proba[:, :, i] = np.exp(-cumulative_hazard[:, i]) ** partial_hazards

    # Hazard probability is complementary of survival
    hazard_proba = 1 - survival_proba

    return survival_proba, hazard_proba


def batch_hpd_quantile(x: np.array, bs: int = 1000, alpha: float = 0.05):
    """
    This function accepts the survival probability draws by time per subject
    and extracts the median and the HPD area in batches. This is a faster version
    for large datasets.
    Parameters
    ----------
    x: np.array
        numpy array of survival probabilities with dimension NxKxT,
        with N being the number of subjects, K the number
        of MCMC draws and T the number of total
        time points.

    bs: int
        integer representing the size per batch,

    alpha: float
        float representing the quantile level
        of getting the HPD interval, i.e.,
        extracting the [hpd_alpha/2, 1-hpd_alpha/2]
        quantiles of the posterior draws.
    Returns
    -------
        hpd_median : pd.DataFrame,
        hpd_low : pd.DataFrame,
        hpd_high : pd.DataFrame
    """

    hpd_array = np.zeros(shape=(x.shape[0], x.shape[2]))
    hpd_array_low = np.zeros(shape=(x.shape[0], x.shape[2]))
    hpd_array_high = np.zeros(shape=(x.shape[0], x.shape[2]))

    for i in range(0, len(x), bs):
        hpd_array[i : i + bs] = np.quantile(x[i : i + bs], q=0.5, axis=(1,))
        hpd_array_low[i : i + bs] = np.quantile(x[i : i + bs], q=alpha / 2, axis=(1,))
        hpd_array_high[i : i + bs] = np.quantile(
            x[i : i + bs], q=1 - alpha / 2, axis=(1,)
        )

    hpd_median = (
        pd.DataFrame(hpd_array)
        .reset_index()
        .melt(id_vars=["index"], value_name="survival_probability", var_name="time")
    )
    hpd_low = (
        pd.DataFrame(hpd_array_low)
        .reset_index()
        .melt(id_vars=["index"], value_name="survival_probability", var_name="time")
    )

    hpd_high = (
        pd.DataFrame(hpd_array_high)
        .reset_index()
        .melt(id_vars=["index"], value_name="survival_probability", var_name="time")
    )
    return hpd_median, hpd_low, hpd_high


def get_hpd_individual_survival_probas_by_time(
    survival_proba: np.array, hpd_alpha: float = 0.05
):
    """
    This function accepts the survival probability draws by time per subject
    and extracts the median and the HPD area.

    Parameters
    ----------
    survival_proba: np.array
        numpy array of survival probabilities with dimension NxKxT,
        with N being the number of subjects, K the number
        of MCMC draws and T the number of total
        time points.

    hpd_alpha: float
        float representing the quantile level
        of getting the HPD interval, i.e.,
        extracting the [hpd_alpha/2, 1-hpd_alpha/2]
        quantiles of the posterior draws.

    Returns
    -------
        median_survival_user_id_by_time,
        hpd_low_survival_user_id_by_time,
        hpd_high_survival_user_id_by_time
    """
    median_survival_user_id_by_time = {}
    hpd_low_survival_user_id_by_time = {}
    hpd_high_survival_user_id_by_time = {}

    # For each time point calculate survival quantiles per subject
    for time in range(survival_proba.shape[2]):
        median_survival_user_id_by_time.update(
            {f"t_{time}": np.quantile(survival_proba[:, :, time], q=0.5, axis=1)}
        )

        hpd_low_survival_user_id_by_time.update(
            {
                f"t_{time}": np.quantile(
                    survival_proba[:, :, time], q=hpd_alpha / 2.0, axis=1
                )
            }
        )

        hpd_high_survival_user_id_by_time.update(
            {
                f"t_{time}": np.quantile(
                    survival_proba[:, :, time], q=1.0 - hpd_alpha / 2.0, axis=1
                )
            }
        )

    median_survival_user_id_by_time = (
        pd.DataFrame(median_survival_user_id_by_time)
        .reset_index()
        .melt(id_vars=["index"], value_name="survival_probability", var_name="time")
    )
    hpd_low_survival_user_id_by_time = (
        pd.DataFrame(hpd_low_survival_user_id_by_time)
        .reset_index()
        .melt(id_vars=["index"], value_name="survival_probability", var_name="time")
    )
    hpd_high_survival_user_id_by_time = (
        pd.DataFrame(hpd_high_survival_user_id_by_time)
        .reset_index()
        .melt(id_vars=["index"], value_name="survival_probability", var_name="time")
    )

    return (
        median_survival_user_id_by_time,
        hpd_low_survival_user_id_by_time,
        hpd_high_survival_user_id_by_time,
    )


def get_individual_time_to_event(
    survival_probas: np.array, survival_probability_threshold: float
):
    """
    This function accepts the individual survival
    probabilities for all time points and
    outputs the estimated time to event by subject, i.e., the
    first time point for which the survival probability will drop
    below 0.5.

    Parameters
    ----------
    survival_probas: np.array
        numpy array of survival probabilities wuth dimension NxKxT,
        with N being the number of subjects, K the number
        of MCMC draws and T the number of total
        time points.
    survival_probability_threshold: float
        float between 0 and 1 indicating the probability threshold
        to mark the event. Default is set to 0.5.

    Returns
    -------
        time_to_event_all_its
        Estimated time to event by subject for each MCMC draw,
    """

    # Initialise the matrix
    time_to_event_all_its = np.zeros(
        (survival_probas.shape[0], survival_probas.shape[1])
    )

    # For each MCMC draw calculate the estimated time to event by subject
    for iteration in range(survival_probas.shape[1]):
        for user in range(survival_probas.shape[0]):
            # we are adding +1 to bring time point from 0 to 1
            time_to_event_all_its[user, iteration] = (
                qth_survival_times(
                    survival_probability_threshold, survival_probas[user, iteration, :]
                )
                + 1
            )

    return time_to_event_all_its


def transform_feature_val_to_scale(
    feature: str, value: float, scaler, num_feats: List[str]
):
    """
    This function accepts a pre-scaled value for a feature
    and the min max scaler that was used in order to create
    its scaled version.

    Parameters
    ----------
    feature: str
        feature of interest
    value: float
        value that we want to scale
    scaler: MinMaxScaler()
        min max scaler that was used for modelling
    num_feats: int
        feature names that where used for modelling

    Returns
    -------
        value_scaled
    """
    # Fetch min, max from MinMaxScaler() for each feature
    min_max = pd.DataFrame([scaler.data_min_, scaler.data_max_])
    min_max.columns = num_feats
    # Select the feature of interest
    min_feat = min_max[feature][0]
    max_feat = min_max[feature][1]

    # transform the value with min max
    val_std = (value - min_feat) / (max_feat - min_feat)

    value_scaled = val_std * (max_feat - min_feat) + min_feat
    return value_scaled


def get_covariate_effect_continuous(
    betas: pd.DataFrame,
    baseline: np.array,
    feature: str,
    value: float,
    scaler,
    num_feats: List[str],
):
    """
    This function calculates the covariate effect
    of a value of a feature, conditionally on the rest.

    Parameters
    ----------
    betas: pd.DataFrame
        pd.DataFrame of NxP with the estimated coefficients
    baseline: np.array
        np.array of baseline hazard function
    feature: str
        feature name to calculate the effect on
    value: float
        value of the feature to calculate the effect on
    scaler: MinMaxScaler()
        min max scaler that was used for modelling
    num_feats: Iterable
        feature names that where used for modelling

    Returns
    -------
            h(t|feature=scaled_val)
    """
    # Create a fake DataFrame for the prediction to happen
    # the rest of the features should be zero
    fake_df = pd.DataFrame(np.zeros((1, len(num_feats))))
    fake_df.columns = num_feats
    # Replace the value we want to see the covariate effect for
    fake_df[feature] = value
    # Transform based on the scaler we used to model
    fake_scaled = scaler.transform(fake_df).reshape(-1)
    position_val = num_feats.index(feature)
    # fetch the normalised value
    normalised_val = fake_scaled[position_val]
    # calculate hazard function conditionally on the value
    return baseline * np.exp(np.atleast_2d(betas[feature]).T * normalised_val)


def get_covariate_effect_dummy(betas: pd.DataFrame, baseline: np.array, feature: str):
    """
    This function calculates the hazard ratio
    for a dummy feature, conditionally on the rest.

    Parameters
    ----------
    betas: pd.DataFrame
        pd.DataFrame of NxP with the estimated coefficients
    baseline: np.array
        np.array of baseline hazard function
    feature: str
        feature name to calculate the hazard ratio

    Returns
    -------
        h(t|feature=1) / h0(t)
    """
    return baseline * np.exp(np.atleast_2d(betas[feature]).T)
