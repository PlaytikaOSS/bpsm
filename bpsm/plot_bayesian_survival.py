from typing import Any, Dict, Iterable, Optional

import arviz as az
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from lifelines import KaplanMeierFitter
from pydantic import BaseModel, confloat, conint, root_validator

from .plot_utils import plot_style
from .predictive_functions import (
    get_baseline_cumulative_survival,
    get_covariate_effect_continuous,
    get_covariate_effect_dummy,
    get_cumulative_hazard,
)


class PlotBayesianSurvival(BaseModel):
    """
    Plots survival model's posterior distributions,
    personalised curves and forest plots for
    feature coefficients.

    Parameters
    ----------
    model: BayesianSurvival
    model_name: str
        model name or identifier for plot labeling
    hpd_alpha: float representing the quantile level
        of getting the HPD interval, i.e.,
        extracting the [alpha/2, 1-alpha/2]
        quantiles of the posterior draws.
    """

    model: Any
    model_name: str
    hpd_alpha: confloat(ge=0, le=1)
    up_label: Optional[confloat(ge=0, le=1)]
    low_label: Optional[confloat(ge=0, le=1)]

    @root_validator(pre=False)
    def calculate_hpd_bounds(cls, values: Dict) -> Dict:
        """
        Calculates the lower and upper bounds
        based on the hpd_alpha.

        Parameters
        ----------
        values: Dict
            Dictionary including all the constructor variables

        Returns
        -------
            Dict
        Updated dictionary for constructor variables
        """
        values["up_label"] = str((1 - values["hpd_alpha"] / 2) * 100) + "%"
        values["low_label"] = str((values["hpd_alpha"] / 2) * 100) + "%"
        return values

    class Config:
        arbitrary_types_allowed = True

    def plot_posteriors_az(self, variables: list[str]):
        """
        Plots posterior distributions with arviz
        library.

        Parameters
        ----------
        variables: list[str]
            list of strings indicating the parameter
            names.
        """
        for var in variables:
            with plot_style((15, 5)):
                # Plot posteriors with HDI
                _ = az.plot_posterior(
                    self.model.idata, var_names=var, hdi_prob=1 - self.hpd_alpha
                )
                plt.show()

    def plot_forest_vars(self, variables: list[str]):
        """
        Plots posterior distributions with arviz
        library as a forest plot.

        Parameters
        ----------
        variables: list[str]
            list of strings indicating the parameter
            names.
        """
        for var in variables:
            _ = az.plot_forest(
                self.model.idata,
                var_names=[var],
                hdi_prob=1 - self.hpd_alpha,
                figsize=(10, 10),
                backend="bokeh",
            )
            plt.show()

    def plot_forest_betas(self):
        """
        Plots posterior distributions for
         feature coefficients with arviz
        library as a forest plot.
        """
        (ax,) = az.plot_forest(
            [self.model.idata],
            var_names=["beta"],
            model_names=[self.model_name],
            kind="ridgeplot",
            ridgeplot_truncate=False,
            ridgeplot_alpha=0.5,
            ridgeplot_overlap=10,
            hdi_prob=1 - self.hpd_alpha,
            combined=True,
            figsize=(15, 10),
        )

        ax.axvspan(
            self.model.idata.posterior["beta"].min() - 0.5, 0, alpha=0.1, color="green"
        )
        ax.axvspan(
            0, self.model.idata.posterior["beta"].max() + 0.5, alpha=0.1, color="red"
        )

        ax.set_xlabel(r"$\beta_i$")

        ax.set_yticklabels(self.model.feature_names[::-1])

        ax.set_title("Posterior distribution coefficients")
        fig = plt.gcf()
        return fig

    def fancy_density_plot_by_feature(self, df: pd.DataFrame, extra_title: str = ""):
        """
        Plot the conditional posterior distribution
        for along with the HPD area.

        Parameters
        ----------
        df: pd.DataFrame
            parameter dataframe
        extra_title: str
            extra labeling for the main plot title
        """

        with plot_style((10, 5)):
            for feature in self.model.feature_names:
                # KDE on the posterior distribution
                sns.kdeplot(
                    df[feature],
                    cumulative=False,
                    label="Observed",
                    color="blue",
                    alpha=0.5,
                )
                # plot vertical lines for HPD and median
                plt.axvline(
                    x=df.T[feature].loc[self.low_label],
                    color="blue",
                    label=f"{100 * (1 - self.hpd_alpha)}% HPD",
                    alpha=0.2,
                    linestyle="--",
                )
                plt.axvline(
                    x=df.T[feature].loc[self.up_label],
                    color="blue",
                    alpha=0.2,
                    linestyle="--",
                )
                plt.axvline(
                    x=df.T[feature].loc["50.0%"],
                    color="red",
                    label="Median",
                    alpha=0.2,
                    linestyle="--",
                )
                plt.suptitle(
                    f"Conditional posterior distribution for {extra_title}{feature}"
                    "coefficient"
                )
                plt.show()

    def plot_baseline_time_points(self, label: str, n_subrows: conint(ge=1) = 4):
        """
        Plot the conditional posterior distribution
        for baseline hazard or survival at time points
        along with HPD area.

        Parameters
        ----------
        label: str
            extra labeling for the main plot title
        n_subrows: conint
            positive number of rows to consider in the subplot

        Returns
        ----------
            matplotlib.pyplot
        """
        if label == "survival":
            df = self.model.posterior_parameters["baseline_survival_probability"]
        else:
            df = self.model.posterior_parameters["baseline_hazard"]

        n_time_points = df.shape[1]
        sub_cols = n_time_points // n_subrows

        fig, axes = plt.subplots(n_subrows, sub_cols, figsize=(30, 15))
        axes = axes.flatten()

        for time_point, ax in zip(range(n_time_points), range(len(axes))):
            # KDE on the posterior distribution
            sns.kdeplot(
                df[:, time_point],
                ax=axes[ax],
                cumulative=False,
                label="Observed",
                color="blue",
                alpha=0.5,
            )
            # plot vertical lines for HPD and median
            axes[ax].axvline(
                x=np.quantile(df[:, time_point], self.hpd_alpha / 2),
                color="blue",
                label=f"{100 * (1 - self.hpd_alpha)}% HPD",
                alpha=0.2,
                linestyle="--",
            )
            axes[ax].axvline(
                x=np.quantile(df[:, time_point], 1 - self.hpd_alpha / 2),
                color="blue",
                label=f"{100 * (1 - self.hpd_alpha)}% HPD",
                alpha=0.2,
                linestyle="--",
            )
            axes[ax].axvline(
                x=np.quantile(df[:, time_point], 0.5),
                color="red",
                label="95% HPD",
                alpha=0.2,
                linestyle="--",
            )
            axes[ax].set_title(f"t={time_point + 1} ")
        fig.suptitle(f"Posterior for baseline {label} function")
        return fig

    def plot_posterior_probas_for_n_users(
        self,
        probabilities: pd.DataFrame,
        user_ids: Iterable,
        label: str,
        n_subrows: conint(ge=1) = 4,
    ):
        """
        Plot the conditional posterior distribution
        for baseline survival/hazard probability at time points
        along with HPD area for n different users.

        Parameters
        ----------
        probabilities: pd.DataFrame
            probabilities dataframe as sampled from the model
        user_ids: Iterable
            list with user ids' indices
        label: str
            extra labeling for the main plot title
        n_subrows: int
            number of rows to consider in the subplot. Default is set to 4

        Returns
        ----------
            matplotlib.pyplot
        """
        n_time_points = probabilities.shape[2]
        sub_cols = n_time_points // n_subrows

        fig, axes = plt.subplots(n_subrows, sub_cols, figsize=(30, 15))
        axes = axes.flatten()

        uid_colors = dict(
            zip(
                range(len(user_ids)),
                sns.color_palette("Set2", n_colors=len(user_ids)).as_hex(),
            )
        )

        for time_point, ax in zip(range(n_time_points), range(len(axes))):
            for i, uid in enumerate(user_ids):
                # KDE on the posterior distribution
                sns.kdeplot(
                    probabilities[uid, :, time_point],
                    ax=axes[ax],
                    cumulative=False,
                    color=uid_colors[i],
                    alpha=0.5,
                )
                # plot vertical lines for HPD and median
                axes[ax].axvline(
                    x=np.quantile(
                        probabilities[uid, :, time_point], self.hpd_alpha / 2
                    ),
                    color=uid_colors[i],
                    label=f"user {i + 1} {100 * (1 - self.hpd_alpha)}% HPD",
                    alpha=0.2,
                    linestyle="--",
                )
                axes[ax].axvline(
                    x=np.quantile(
                        probabilities[uid, :, time_point], 1 - self.hpd_alpha / 2
                    ),
                    color=uid_colors[i],
                    alpha=0.2,
                    linestyle="--",
                )
                axes[ax].axvline(
                    x=np.quantile(probabilities[uid, :, time_point], 0.5),
                    color=uid_colors[i],
                    label=f"user {i + 1} median",
                    alpha=0.2,
                )
                axes[ax].set_title(f"time={time_point}")

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig.suptitle(f"Posterior for {label} probability for different users")
        return fig

    def plot_baseline_survival_with_KM(
        self, y: pd.DataFrame, time_col: str, event_col: str
    ):
        """
        Plot the conditional posterior distribution
        for baseline survival at time points
        along with HPD area and the Kaplan - Meier non
        parametric estimator.

        Parameters
        ----------
        y: pd.DataFrame
            Dataframe including time to event and event columns
        time_col: str
            time to event column name
        event_col: str
            event column name

        Returns
        -------
            matplotlib.pyplot
        """
        fig = plt.figure(figsize=(5, 5))
        base_survival = self.model.posterior_parameters["baseline_survival_probability"]
        n_time_points = base_survival.shape[1]

        plt.plot(
            np.arange(1, n_time_points),
            np.quantile(base_survival, q=0.5, axis=0)[:-1],
            color="blue",
            label="Posterior baseline survival",
        )
        ax = plt.gca()

        ax.fill_between(
            # set same index (time)
            np.arange(1, n_time_points),
            # set lower bound
            np.quantile(base_survival, q=self.hpd_alpha / 2, axis=0)[:-1],
            # set upper bound
            np.quantile(base_survival, q=1 - self.hpd_alpha / 2, axis=0)[:-1],
            alpha=0.3,
        )

        kmf = KaplanMeierFitter()
        kmf.fit(y[time_col], y[event_col])
        ci = kmf.confidence_interval_survival_function_
        ts = ci.index
        low, high = np.transpose(ci.values)

        ax.fill_between(ts[1:], low[1:], high[1:], color="green", alpha=0.3)
        kmf.survival_function_[1:].plot(ax=ax, color="green", label="KM fitter")

        plt.ylabel("Survival baseline function $S(t|.)$")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        return fig

    @staticmethod
    def plot_surv_trajectory_n_users(dfs: list[pd.DataFrame], user_ids: Iterable):
        """
        Plot the personalised survival probability's trajectory
        at time points along with HPD area for n different users.

        Parameters
        ----------
        dfs: list of pandas.DataFrame
          list of dataframes incorporating the median and HPD areas
          (low and high) of survival probabilities per time point per user.

        user_ids: Iterable
            list of user indices to be selected for plotting

        Returns
        -------
            matplotlib.pyplot
        """

        uid_colors = dict(
            zip(user_ids, sns.color_palette("Set2", n_colors=len(user_ids)).as_hex())
        )

        df_med, df_low, df_high = dfs[0], dfs[1], dfs[2]

        n_time_points = df_med.time.nunique()

        _ = plt.figure(figsize=(5, 5))
        # For each id
        for user in user_ids:
            # Plot the median survival probability by time point
            plt.plot(
                np.arange(1, n_time_points + 1),
                df_med[df_med["index"] == user].survival_probability,
                alpha=1,
                color=uid_colors[user],
            )
            ax = plt.gca()
            # Fill with the 95% CDI to show uncertainty
            ax.fill_between(
                # set same index (time)
                np.arange(1, n_time_points + 1),
                # set lower bound
                df_low[df_low["index"] == user].survival_probability,
                # set upper bound
                df_high[df_high["index"] == user].survival_probability,
                alpha=0.1,
                color=uid_colors[user],
            )
            plt.title("Personalised Survival probability for different users")
            plt.ylabel("Survival probability $S(t|x,.)$")

    def plot_with_hpd(
        self,
        x: np.array,
        hazard: pd.DataFrame,
        f: staticmethod,
        interval_length: conint(ge=1),
        ax: matplotlib.axes,
        color: str = None,
        label: str = None,
    ):
        """
        This function plots the piecewise hazard function with its hpd values.

        Parameters
        ----------
        x: np.array
            interval points (bounds) to plot
        hazard: pd.DataFrame
            hazard function of NxT dimension
        f: method
            function (cumulative) to calculate cumulative hazard/survival
        interval_length: int
            integer specifying the length of the intervals
        ax: matplotlib.axes
        color: str
        label: str

        Returns
        -------
            matplotlib.pyplot
        """
        percentiles = 100 * np.array(
            [self.hpd_alpha / 2.0, 0.5, 1 - self.hpd_alpha / 2.0]
        )
        if not f:
            hpd = np.percentile(hazard, percentiles, axis=0)
        else:
            hpd = np.percentile(f(hazard, interval_length), percentiles, axis=0)
        ax.fill_between(x, hpd[0], hpd[2], color=color, alpha=0.1)
        ax.plot(x, hpd[1], color=color, label=label)

    def plot_baseline_vs_partial_effects(
        self,
        feature: str,
        values: Iterable,
        type_var: str,
        min_max_scaler: sklearn.preprocessing.MinMaxScaler,
        num_feats: list[str],
    ):
        """
        Plots baseline cumulative hazard and survival functions
        with partial effects of covariate or categorical values
        with HPD areas.

        Parameters
        ----------
        feature: str
            feature names to calculate partial effects
        values: Iterable
            list of values to calculate the partial effects on.
            If feature is categorical, it should be set to [1].
        type_var: str
            string indicating whether the variable is
            continuous or categorical.
        min_max_scaler: sklearn.preprocessing.MinMaxScaler
            Min-max scaler used for data pre-processing before
            training.
        num_feats: list[str]
            list of numerical features

        """

        fig, (hazard_ax, surv_ax) = plt.subplots(
            ncols=2, sharex=True, sharey=False, figsize=(16, 6)
        )

        colors = dict(
            zip(
                values + ["baseline"],
                sns.color_palette("Set2", n_colors=len(values) + 1).as_hex(),
            )
        )

        # Plot hpd area for the baseline cumulative hazard.
        # That is, the sum of baseline hazard for all time points
        self.plot_with_hpd(
            x=self.model.interval_bounds[:-1],
            hazard=self.model.posterior_parameters["baseline_hazard"],
            f=get_cumulative_hazard,
            interval_length=1,
            ax=hazard_ax,
            color=colors["baseline"],
            label="Baseline",
        )

        # Plot hpd area for the cumulative hazard for a specific covariate
        # value, conditonally on the rest.

        for val in values:
            if type_var == "continuous":
                self.plot_with_hpd(
                    x=self.model.interval_bounds[:-1],
                    # As the features were scaled, we need to
                    # scale that value too in get_covariate_effect_continuous.
                    hazard=get_covariate_effect_continuous(
                        self.model.posterior_parameters["betas"],
                        self.model.posterior_parameters["baseline_hazard"],
                        feature,
                        val,
                        min_max_scaler,
                        num_feats,
                    ),
                    f=get_cumulative_hazard,
                    interval_length=1,
                    ax=hazard_ax,
                    color=colors[val],
                    label=f"{feature}={val}",
                )
            else:
                self.plot_with_hpd(
                    x=self.model.interval_bounds[:-1],
                    hazard=get_covariate_effect_dummy(
                        self.model.posterior_parameters["betas"],
                        self.model.posterior_parameters["baseline_hazard"],
                        feature,
                    ),
                    f=get_cumulative_hazard,
                    interval_length=1,
                    ax=hazard_ax,
                    color=colors[val],
                    label=f"{feature}=1",
                )

        hazard_ax.set_xlim(0, self.model.interval_bounds[-2])
        hazard_ax.set_xlabel("Days")

        hazard_ax.set_ylabel(r"Cumulative hazard $\Lambda(t)$")

        hazard_ax.legend(loc=2)

        # Plot hpd area for the baseline survival.

        self.plot_with_hpd(
            x=self.model.interval_bounds[:-1],
            hazard=self.model.posterior_parameters["baseline_hazard"],
            f=get_baseline_cumulative_survival,
            interval_length=1,
            ax=surv_ax,
            color=colors["baseline"],
            label="Baseline",
        )

        # Plot hpd area for the survival probability for a specific covariate
        # value, conditonally on the rest.
        for val in values:
            if type_var == "continuous":
                self.plot_with_hpd(
                    x=self.model.interval_bounds[:-1],
                    # As the features were scaled, we need to
                    # scale that value too in get_covariate_effect_continuous.
                    hazard=get_covariate_effect_continuous(
                        self.model.posterior_parameters["betas"],
                        self.model.posterior_parameters["baseline_hazard"],
                        feature,
                        val,
                        min_max_scaler,
                        num_feats,
                    ),
                    f=get_baseline_cumulative_survival,
                    interval_length=1,
                    ax=surv_ax,
                    color=colors[val],
                    label=f"{feature}={val}",
                )
            else:
                self.plot_with_hpd(
                    x=self.model.interval_bounds[:-1],
                    hazard=get_covariate_effect_dummy(
                        self.model.posterior_parameters["betas"],
                        self.model.posterior_parameters["baseline_hazard"],
                        feature,
                    ),
                    f=get_baseline_cumulative_survival,
                    interval_length=1,
                    ax=surv_ax,
                    color=colors[val],
                    label=f"{feature}=1",
                )

        surv_ax.set_xlim(0, self.model.interval_bounds[-2])
        surv_ax.set_xlabel("Days")

        surv_ax.set_ylabel("Survival function $S(t)$")

        fig.suptitle("Bayesian survival model")
