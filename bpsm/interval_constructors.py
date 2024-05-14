import numpy as np
import pandas as pd


def get_n_piecewise_intervals(
    df: pd.DataFrame, time_to_event_col: str, interval_length: int
):
    """
    This function creates the time piecewise intervals
    based on an interval length. The partitioned interval
    is of form 0≤ s1<s2<⋯<sT, where T is the final time point.
    The intervals that are created have length of interval_length.

    ------------------
    Parameters
    df: pd.DataFrame
        pd.DataFrame of n rows and p columns. Necessary
        to have is the time to event column within
        the dataframe.
    time_to_event_col: str
        string indicating time to event.
    interval_length: int
        integer specifying the length of the intervals

    Return
    ------
        interval_bounds, n_intervals, intervals
    """
    # Create the interval bounds based on the
    # maximum time to event and the length per interval
    interval_bounds = np.arange(
        0, df[time_to_event_col].max() + interval_length + 1, interval_length
    )
    # Number of total intervals is determined by
    # the maximum values and the interval length
    n_intervals = interval_bounds.size - 1
    # Extract the intervals
    intervals = np.arange(n_intervals)
    return interval_bounds, n_intervals, intervals


def define_indicator_and_exposure(
    df: pd.DataFrame,
    time_to_event_col: str,
    event_col: str,
    interval_length: int,
    interval_bounds: np.array,
    n_intervals: int,
):
    """
    This function defines churn indicator variables
    based on whether the i-th subject died in the j-th interval, i.e.,
    di,j = 1 if the subject i dies in interval j, or di,j = 0 otherwise
    and exposure, i.e., the amount of time the
    si-th subject was at risk in the j-th interval.

    Parameters
    ----------
    df: pd.DataFrame
        pd.DataFrame of n rows and p columns. Necessary
        to have is the time to event column within
        the dataframe.
    time_to_event_col: str
        string specifying the time to event column
    event_col: str
        string specifying the event column
    interval_length: int
        integer specifying the length of the piecewise intervals
    interval_bounds: np.array
        numpy array with the piecewise interval bounds
    n_intervals: int
        integer indicating the number of piecewise intervals
    Returns
    -------
        churn: numpy.array
            array of shape n for the number of users
            and n_intervals columns. It creates
            a churn indicator for each time point.
        exposure: numpy.array
            array of shape n for the number of users
            and n_intervals columns. It creates
            an exposure indicator for each time point.
            exposure will be set to 0 at t+1
            if user churned at time t.

        churn, exposure
    """

    n_users = df.shape[0]

    users = np.arange(n_users)

    last_period = np.floor((df[time_to_event_col] - 0.01) / interval_length)

    # Create the event matrix, based on the number of intervals
    event = np.zeros((n_users, n_intervals))
    # Mark the interval for which we observed "death"
    event[users, last_period.astype(int)] = df[event_col]

    # Specify the time indicator that the user was subject to risk at each interval
    exposure = (
        np.greater_equal.outer(df[time_to_event_col].values, interval_bounds[:-1])
        * interval_length
    )
    exposure[users, last_period.astype(int)] = (
        df[time_to_event_col].values - interval_bounds[last_period.astype(int)]
    )
    # combination of matrices for a user that churn at t=3 for a total of 6 intervals:
    # event = [0, 0, 1, 0, 0, 0], exposure = [1, 1, 1, 0, 0, 0]
    return event, exposure
