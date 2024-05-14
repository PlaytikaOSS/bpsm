import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from arviz import InferenceData
from matplotlib import pyplot as plt
from pydantic import BaseModel, confloat, conint

from .interval_constructors import (
    define_indicator_and_exposure,
    get_n_piecewise_intervals,
)
from .plot_utils import plot_style
from .predictive_functions import (
    get_baseline_cumulative_survival,
    get_beta_coefficients,
    get_cumulative_hazard,
    get_hpd_individual_survival_probas_by_time,
    predict_hazard_by_time,
    predict_time_to_event,
)


class BayesianSurvivalModel(BaseModel, ABC):
    """
    Trains a Bayesian piecewise exponential survival model
    or its regularised extension via horseshoe priors
    on the feature coefficients. The number intervals or time points
    for the hazard function to be assumed different is specified by the
    user. The inference used for extracting the posteriors of the parameters
    under this model is variational inference (ADVI).
    """

    time_to_event_col: Optional[str]
    event_col: Optional[str]
    interval_length: Optional[conint(gt=0)]
    hyper_param_lambda0: Optional[List[confloat(gt=0)]]
    n_its: Optional[conint(gt=0)]
    n_samples: Optional[conint(gt=0)]
    print_elbo: Optional[bool]
    interval_bounds: Optional[List]
    n_intervals: Optional[conint(gt=0)]
    intervals: Optional[List]
    feature_names: Optional[List]
    idata: Optional[InferenceData]
    posterior_parameters: Optional[Dict]

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def _specify_model(
        self,
        n_covars: int,
        hyper_param_lambda0: list[float],
        X_train: pd.DataFrame,
        event: np.array,
        exposure: np.array,
    ):
        assert n_covars > 0
        pass

    @abstractmethod
    def get_model_specific_params(self, n_samples: int, parameters_dict: dict):
        assert n_samples > 0
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        time_to_event_col: str,
        event_col: str,
        interval_length: int,
        hyper_param_lambda0: list[float],
        n_its: int,
        n_samples: int,
        print_elbo: bool = False,
    ):
        """
        Fits the Bayesian piecewise survival
        models defined in
        https://wiki.playtika.com/display/AILAB/Bayesian+Implementation
        and plots ELBO if desired.

        Parameters
        ----------
        X : pandas.DataFrame
            pandas.DataFrame of features to be used for training.
            It is suggested for the features to be scaled.
        y : list
            list of numpy arrays in the form of [exposure, churn]
            precalculated from bayesian_funcs.define_indicator_and_exposure
        time_to_event_col : str
            time to event column name
        event_col: str
            desired event column name to be created
        interval_length: int
            length of each interval.
            The relationship is max_time_to_event + 1 = n_intervals / interval_length.
        hyper_param_lambda0: list of float
            list of floats specifying the hyperparameters of prior distribution
            for the hazard functions.
            The first element indicates the shape parameter and the second
            element indicates rate parameterer of gamma distribution.
        n_its : int
            Number of iterations to run variational inference.
        n_samples: int
            Number of parameter samples to extract from the approximate posterior
            distribution of the parameters.
        print_elbo: bool
            boolean indicating whether ELBO should be
            printed or not.
        """

        self.time_to_event_col = time_to_event_col
        self.event_col = event_col
        self.interval_length = interval_length
        self.n_its = n_its

        # Construct piecewise exponential regression intervals
        (
            self.interval_bounds,
            self.n_intervals,
            self.intervals,
        ) = get_n_piecewise_intervals(y, time_to_event_col, interval_length)
        # Define event and exposure in intervals per user
        event, exposure = define_indicator_and_exposure(
            df=y,
            time_to_event_col=self.time_to_event_col,
            event_col=self.event_col,
            interval_length=self.interval_length,
            interval_bounds=self.interval_bounds,
            n_intervals=self.n_intervals,
        )

        self.feature_names = list(X.columns)

        # Model specifications
        model = self._specify_model(
            len(self.feature_names), hyper_param_lambda0, X, event, exposure
        )

        # Use Mean Field variational inference
        with model:
            inference = pm.ADVI()
            approx = pm.fit(n=self.n_its, method=inference)

        if print_elbo:
            with plot_style():
                # ELBO diagnostics convergence
                plt.plot(approx.hist, label="ADVI", alpha=0.3)
                plt.legend()
                plt.ylabel("ELBO")
                plt.xlabel("iteration")

        # Save posterior parameters
        self.get_samples(approx, n_samples)

    def _get_traces(self, approx, n_samples: int):
        """
        Samples from the approximate posterior distributions
        estimated from ADVI and reshapes them.

        Parameters
        ----------
        approx : pm.InferenceData
            pymc Mean Field variational inference after fitting
        n_samples:  int
            Number of parameter samples to extract from the approximate posterior
            distribution of the parameters.

        Returns
        ----------
        posterior_parameters_common: dict
            dictionary of arrays incorporating the posterior parameters
        """
        # Fetch samples from the approximated posterior distributions
        self.idata = approx.sample(n_samples)
        # Reshaping due to an extra dimension indicating the one "chain"
        betas = self.idata.posterior["beta"].values.reshape(
            n_samples, len(self.feature_names)
        )
        lambda0 = self.idata.posterior["lambda0"].values.reshape(
            n_samples, self.n_intervals
        )

        posterior_parameters_common = {"betas": betas, "baseline_hazard": lambda0}

        return posterior_parameters_common

    def get_median_summaries(self, variables: list[str], is_coef: bool = True):
        """
        Accepts a list of strings for the parameter names
        and prints summaries (median and 95% credible interval)
        of their posterior estimates.

        Parameters
        ----------
        variables : list of str
            list of strings indicating the variables to describe.
        is_coef : boolean
            Indicates whether median summaries to be extracted are
            for model coefficients or not. Default is set to True.

        Returns
        ----------
        summary : az.summary
            summary statistics for posterior parameters provided by arviz library
        """

        for var in variables:
            # Get the summaries (based on median). r_hat will be NaN because we don't
            # have multiple chains as in MCMC
            summary = az.summary(
                self.idata,
                var_names=var,
                round_to=2,
                stat_focus="median",
                hdi_prob=0.95,
            )
            if is_coef:
                summary.index = self.feature_names
        return summary

    def get_samples(self, approx, n_samples: conint(ge=1)):
        """
        Returns all the posterior distributions. Specifically,
        the posteriors of regression coefficients \beta,
        their summary values and the summary of their exponents,
        their coefficient specific variances omega_{1p} and global variance omega_2
        or sigma if using the simple Bayesian Survival model,
        the piecewize hazard function lambda_i,
        the cumulative hazard function H_i and
        the baseline cumulative survival function S_i.

        Parameters
        ----------
        approx: pm.InferenceData
            approximation derived from pymc4 with variational inference
        n_samples: conint
            positive integer indicating number of posterior draws to extract

        """
        raw_parameters = self._get_traces(approx, n_samples)
        # Bring betas into a pandas DataFrame
        # and calculate median and 95% HPD.
        # Same for the exponents
        betas_draws, summary_beta_vals, summary_exp_vals = get_beta_coefficients(
            raw_parameters["betas"], feature_names=self.feature_names
        )
        # Predict hazard function per user based on features
        # and baseline hazard values
        # and fetch survival probabilities
        cumulative_hazard = get_cumulative_hazard(
            raw_parameters["baseline_hazard"], self.interval_length, self.n_intervals
        )

        baseline_surv_proba = get_baseline_cumulative_survival(
            raw_parameters["baseline_hazard"], self.interval_length, self.n_intervals
        )
        raw_parameters["betas"] = betas_draws
        raw_parameters["summary_betas"] = summary_beta_vals
        raw_parameters["summary_exp_betas"] = summary_exp_vals
        raw_parameters["cumulative_hazard"] = cumulative_hazard
        raw_parameters["baseline_survival_probability"] = baseline_surv_proba

        # Get rest of parameters if they come from horseshoe or simple survival model
        self.posterior_parameters = self.get_model_specific_params(
            n_samples, raw_parameters
        )

    def predict_survival_distributions(self, X: pd.DataFrame):
        """
        Accepts a feature set X and predicts personalised hazard lambda_i
        by time.

        Parameters
        ----------
        X : pandas.DataFrame
         pandas dataframe with features for each user.

        Returns
        ----------
            numpy.array
            numpy array of personalised predictions for each user and each interval/time
            point
        """
        survival_proba, hazard_proba = predict_hazard_by_time(
            self.posterior_parameters["cumulative_hazard"],
            betas=self.posterior_parameters["betas"],
            x=X,
        )
        return survival_proba, hazard_proba

    def predict_survival_hpd(
        self,
        X: pd.DataFrame = None,
        hpd_alpha: confloat(ge=0, le=1) = 0.05,
        from_survival_distributions: bool = False,
        survival_distributions: np.array = None,
    ):
        """
        Calculates the HPD intervals of the personalised survival
        probabilities. It calculates either from a feature matrix
        or straight from the array of survival distributions.

        Parameters
        ----------
        X: pd.DataFrame
            feature set
        hpd_alpha: confloat
        float between 0 and 1 representing the quantile level
        of getting the HPD interval, i.e., extracting the [hpd_alpha/2, 1-hpd_alpha/2]
        quantiles of the posterior draws.
        from_survival_distributions: bool
            Whether survival distributions should be used instead
            of the feature matrix X. default is set to False and survival
            distributions are predicted as well.
        survival_distributions: np.array
            Array of survival distributions to be used instead
            of the feature matrix X. default is set to None. Used only
            when from_survival_distributions is set to True.

        Returns
        -------
        pd.DataFrame, pd.DataFrame, pd.DataFrame
        """
        # Calculate for each user and time point
        # the HPD area and median for
        # survival probability
        if not from_survival_distributions:
            survival_distributions = self.predict_survival_distributions(X)[0]

        (
            median_survival_by_user,
            hpd_low_survival_by_user,
            hpd_high_survival_by_user,
        ) = get_hpd_individual_survival_probas_by_time(
            survival_distributions, hpd_alpha
        )
        return (
            median_survival_by_user,
            hpd_low_survival_by_user,
            hpd_high_survival_by_user,
        )

    def predict_time_to_event_hpd(
        self,
        X: pd.DataFrame = None,
        survival_hpd: np.array = None,
        from_survival_hpd: bool = False,
        survival_probability_threshold: confloat(ge=0, le=1) = 0.5,
        hpd_alpha: confloat(ge=0, le=1) = 0.05,
    ):
        """
        Predicts time to event with hpd intervals.

        Parameters
        ----------
        X: pd.DataFrame
            feature set to be used for predictions. If not specified,
            then survival_hpd probabilities should be used to extract
            predictions. Default is set to None.
        survival_hpd : list
            list of numpy arrays to be used for predictions. If not specified,
            then X should be used to extract
            predictions. Default is set to None.
        from_survival_hpd : bool
            indicate whether predictions are conducted from
            a feature matrix X or from survival probabilities.
        survival_probability_threshold: confloat
        float between 0 and 1 indicating the probability threshold
        to mark the event. Default is set to 0.5.
        hpd_alpha: confloat
        float between 0 and 1 representing the quantile level
        of getting the HPD interval, i.e., calculating the [hpd_alpha/2, 1-hpd_alpha/2]
        quantiles.

        Returns
        -------
        pd.DataFrame
        """

        if not from_survival_hpd:
            median_proba, low_proba, high_proba = self.predict_survival_hpd(
                X=X, hpd_alpha=hpd_alpha
            )
        else:
            median_proba, low_proba, high_proba = survival_hpd
        median_time_to_event = (
            predict_time_to_event(median_proba, "index", survival_probability_threshold)
            .rename("50%")
            .reset_index()
        )

        low_time_to_event = (
            predict_time_to_event(low_proba, "index", survival_probability_threshold)
            .rename(f"{100 * (hpd_alpha / 2)}%")
            .reset_index()
        )

        high_time_to_event = (
            predict_time_to_event(high_proba, "index", survival_probability_threshold)
            .rename(f"{100 * (1 - hpd_alpha / 2)}%")
            .reset_index()
        )
        return median_time_to_event.merge(low_time_to_event, on="index").merge(
            high_time_to_event, on="index"
        )

    def predict(
        self,
        X: pd.DataFrame,
        survival_probability_threshold: confloat(ge=0, le=1) = 0.5,
        hpd_alpha: confloat(ge=0, le=1) = 0.05,
    ):
        """
        Predicts user specific survival posterior
        distributions and time to event with hpd intervals.

        Parameters
        ----------
        X : pd.DataFrame
            Feature set
        survival_probability_threshold : confloat
        float between 0 and 1 indicating the probability threshold
        to mark the event. Default is set to 0.5.
        hpd_alpha: confloat
        float between 0 and 1 representing the quantile level
        of getting the HPD interval, i.e., extracting the [hpd_alpha/2, 1-hpd_alpha/2]
        quantiles of the posterior draws.

        Returns
        -------
        np.array, pd.DataFrame, pd.DataFrame
        """
        survival_distributions = self.predict_survival_distributions(X)[0]
        survival_hpd = self.predict_survival_hpd(
            hpd_alpha=hpd_alpha,
            from_survival_distributions=True,
            survival_distributions=survival_distributions,
        )
        time_to_event_hpd = self.predict_time_to_event_hpd(
            survival_hpd=survival_hpd,
            from_survival_hpd=True,
            survival_probability_threshold=survival_probability_threshold,
            hpd_alpha=hpd_alpha,
        )

        return survival_distributions, survival_hpd, time_to_event_hpd

    def save_model(self, file_name: str, save_idata: bool = False):
        """
        Saves model with parameters as pickle

        Parameters
        ----------
        file_name: str
            File name or path to file name
        save_idata:bool
            Whether to say or not the interactive
            data from pymc. Default is set to False

        """
        if not save_idata:
            del self.idata
        with open(f"{file_name}.pkl", "wb") as file_pi:
            pickle.dump(self, file_pi)
