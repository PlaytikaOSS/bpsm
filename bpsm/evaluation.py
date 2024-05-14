from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
from pydantic import BaseModel, confloat, conint, root_validator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    confusion_matrix,
    f1_score,
)


class Evaluation(BaseModel):
    """
    This class accepts predictions from Bayesian survival models
    calculates evaluation metrics, i.e., c-index, dynamic brier and
    f1 scores, confusion matrix at the end of time to event and
    count plots for real labels vs predicted.

    Parameters
    ----------
    y_train_predicted: pd.DataFrame
        pd.DataFrame with predicted target values of train set
    y_train_real: pd.DataFrame
        pd.DataFrame with observed target values of train set
    y_test_predicted: pd.DataFrame
        pd.DataFrame with predicted target values of test set
    y_test_real: pd.DataFrame
        pd.DataFrame with observed target values of test set
    surv_probas_train : pandas.DataFrame
        Input train set pandas.DataFrame that incorporates survival
        probabilities per user per time point as estimated
        from Bayesian survival model.
    surv_probas_test : pandas.DataFrame
        Input test set pandas.DataFrame that incorporates survival
        probabilities per user per time point as estimated
        from Bayesian survival model.
    time_to_event_col : str
        string specifying the time to event column name, i.e.
        number of days for first churn, first purchase etc.
    event_col : str
        string specifying the event column name, i.e., churn, first purchase etc
    pred_median_col: str
        column name of prediction median
    pred_lower_col: str
        column name of prediction lower value
    pred_upper_col: str
        column name of prediction higher value
    probability_threshold: confloat(ge=0, le=1)
        float between 0 and 1 indicating the probability threshold that someone
        can be predicted as time to event
    hpd_alpha: confloat(ge=0, le=1)
        float between 0 and 1 representing the quantile level
        of getting the HPD interval, i.e.,
        extracting the [hpd_alpha/2, 1-hpd_alpha/2]
    test_set_label: str = "test"
        string for labeling if the second set is test or validation set
    max_time_point: Optional[conint(ge=1)]
        integer specifying the maximum time to event point that exists
        in the data
    merged_train: Optional[pd.DataFrame]
        pd.DataFrame incorporating the merged dataset of predictions
        and observed data for train set
    merged_test: Optional[pd.DataFrame]
        pd.DataFrame incorporating the merged dataset of predictions
        and observed data for test set
    """

    y_train_predicted: pd.DataFrame
    y_train_real: pd.DataFrame
    y_test_predicted: pd.DataFrame
    y_test_real: pd.DataFrame
    surv_probas_train: List[pd.DataFrame]
    surv_probas_test: List[pd.DataFrame]
    time_to_event_col: str
    event_col: str
    pred_median_col: str
    pred_lower_col: str
    pred_upper_col: str
    probability_threshold: confloat(ge=0, le=1)
    hpd_alpha: confloat(ge=0, le=1)
    test_set_label: str = "test"
    max_time_point: Optional[conint(ge=1)]
    merged_train: Optional[pd.DataFrame]
    merged_test: Optional[pd.DataFrame]

    @root_validator(pre=False)
    def initiate_evaluation(cls, values: Dict) -> Dict:
        """
        This function calculates merges predictions and
        observed data and calculates maximum time to event
        observed in the train data.

        Parameters
        ----------
        values: Dict
            Dictionary including all the constructor variables

        Returns
        -------
            Dict
        Updated dictionary for constructor variables
        """
        values["merged_train"] = pd.concat(
            [values["y_train_real"], values["y_train_predicted"]], axis=1
        )
        values["merged_test"] = pd.concat(
            [values["y_test_real"], values["y_test_predicted"]], axis=1
        )
        values["max_time_point"] = (
            values["merged_train"][values["time_to_event_col"]].astype(int).max()
        )
        return values

    class Config:
        arbitrary_types_allowed = True

    def get_concordance(self):
        """
        Calculate a crude concordance interval
        comparing the predicted time to event
        for lower bound of survival probabilities, median and upper bound.
        It uses the 95% HDI from the predictive distributions.

        Returns
        -------
        list
            [c-index train, c-index test]

        """

        cindexes_train = [
            round(
                concordance_index(
                    self.merged_train[f"{100 * (self.hpd_alpha / 2)}%"],
                    self.merged_train[self.time_to_event_col],
                    event_observed=self.merged_train[self.event_col],
                ),
                3,
            ),
            round(
                concordance_index(
                    self.merged_train["50%"],
                    self.merged_train[self.time_to_event_col],
                    event_observed=self.merged_train[self.event_col],
                ),
                3,
            ),
            round(
                concordance_index(
                    self.merged_train[f"{100 * (1 - self.hpd_alpha / 2)}%"],
                    self.merged_train[self.time_to_event_col],
                    event_observed=self.merged_train[self.event_col],
                ),
                3,
            ),
        ]
        if self.merged_test.empty:
            return cindexes_train

        else:
            cindexes_test = [
                round(
                    concordance_index(
                        self.merged_test[f"{100 * (self.hpd_alpha / 2)}%"],
                        self.merged_test[self.time_to_event_col],
                        event_observed=self.merged_test[self.event_col],
                    ),
                    3,
                ),
                round(
                    concordance_index(
                        self.merged_test["50%"],
                        self.merged_test[self.time_to_event_col],
                        event_observed=self.merged_test[self.event_col],
                    ),
                    3,
                ),
                round(
                    concordance_index(
                        self.merged_test[f"{100 * (1 - self.hpd_alpha / 2)}%"],
                        self.merged_test[self.time_to_event_col],
                        event_observed=self.merged_test[self.event_col],
                    ),
                    3,
                ),
            ]

            return cindexes_train, cindexes_test

    def _get_brier_score(self, df: pd.DataFrame, probas: pd.DataFrame):
        """
        Calculate dynamic brier score for a single dataframe.
        $BS_t = 1/N sum_{j=1}^N (f_j - o_j)^2$
        where $f_j$ and $o_tj are the forecast
        (survival probability) and the observed label
        per user.

        Parameters
        ----------
        df: pd.DataFrame
            pd.DataFrame incorporating the event
            label column and the predicted probability
        probas : pd.DataFrame
            survival probabilities

        Returns
        -------
            list
                list of float
                with scores per time point
        """
        brier_score = []
        # Loop over all the days
        for time in range(self.max_time_point):
            # Calculate brier score per time point
            # If estimated hazard probability is closer to 1
            # then the user is correctly classified.
            # Here we used the predicted probabilities as
            # scores.
            brier_score.append(
                brier_score_loss(
                    df[self.event_col],
                    1
                    - np.array(
                        probas.query(f"time=='t_{time}'").survival_probability
                    ).T,
                    pos_label=1,
                )
            )
        return brier_score

    def get_brier_scores(self):
        """
        Calculate dynamic brier scores for
        train and test predictions with 95%
        prediction intervals.

        Returns
        -------
            list
                list of floats
        """
        brier_score_train_low = self._get_brier_score(
            self.merged_train, self.surv_probas_train[1]
        )
        brier_score_train_median = self._get_brier_score(
            self.merged_train, self.surv_probas_train[0]
        )
        brier_score_train_high = self._get_brier_score(
            self.merged_train, self.surv_probas_train[2]
        )
        if self.merged_test.empty:
            return [
                brier_score_train_low,
                brier_score_train_median,
                brier_score_train_high,
            ]
        else:
            brier_score_test_low = self._get_brier_score(
                self.merged_test, self.surv_probas_test[1]
            )
            brier_score_test_median = self._get_brier_score(
                self.merged_test, self.surv_probas_test[0]
            )
            brier_score_test_high = self._get_brier_score(
                self.merged_test, self.surv_probas_test[2]
            )

            return (
                [
                    brier_score_train_low,
                    brier_score_train_median,
                    brier_score_train_high,
                ],
                [brier_score_test_low, brier_score_test_median, brier_score_test_high],
            )

    def plot_brier_scores(self):
        """
        Plots dynamic brier scores for train and test
        predictions with the 95% HDI.
        """
        briers = self.get_brier_scores()

        if self.merged_test.empty:
            _ = plt.figure(figsize=(10, 5))
            plt.plot(
                range(1, self.max_time_point + 1),
                briers[1],
                color="blue",
                label="train",
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                briers[0],
                briers[2],
                color="blue",
                alpha=0.3,
            )
            ax = plt.gca()
            ax.set(
                xlabel="Day",
                ylabel="Brier Score",
                title="Bayesian Model Calibration Over Time",
            )
            ax.grid()
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.show()
        else:
            _ = plt.figure(figsize=(10, 5))
            plt.plot(
                range(1, self.max_time_point + 1),
                briers[0][1],
                color="blue",
                label="train",
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                briers[0][0],
                briers[0][2],
                color="blue",
                alpha=0.3,
            )

            plt.plot(
                range(1, self.max_time_point + 1),
                briers[1][1],
                color="orange",
                label=self.test_set_label,
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                briers[1][0],
                briers[1][2],
                color="orange",
                alpha=0.3,
            )
            ax = plt.gca()
            ax.set(
                xlabel="Day",
                ylabel="Brier Score",
                title="Bayesian Model Calibration Over Time",
            )
            ax.grid()
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.show()

    def _get_dynamic_f1_score(
        self, df: pd.DataFrame, probas: pd.DataFrame, label_type: str, weighted: bool
    ):
        """
        Calculate dynamic f1-score for a single dataframe.
        $F1_t = TP/ (TP + 1/2(FP+FN)$ for all the
        users up to time t.
        The score can be calculated for two definitions:
        i) for the real event where users churn at time t
        and ii) for the real event where users churn at time
        up to t. The predicted class is predicted on the 50%
        threshold.

        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame incorporating the event
            label column and the predicted probability
        probas : pd.DataFrame
            survival probabilities
        label_type: str
            can take values "==t" for using the observed event
            at time t, or "<=t" for using the observed
            event at up to time t.
        weighted : bool
            if set to true then macro f1-score is calculated,
            otherwise f1-score for minority class is calculated.

        Returns
        -------
            list
                list of floats
                with scores per time point
        """
        f1 = []

        for time in range(self.max_time_point):
            # Assign new label, if user churned at
            # time t then it's 1, otw 0.
            if label_type == "==t":
                tmp_df = df.assign(event=time == df[self.time_to_event_col])
            else:
                # Assign new label, if user churned from first
                # time point up to time t,
                # then it's 1, otw 0.
                tmp_df = df.assign(event=time <= df[self.time_to_event_col])

            # Calculate f1-score, if predicted
            # survival probability at time t < 50% then it's churn
            f1.append(
                f1_score(
                    tmp_df["event"],
                    (
                        1
                        - np.array(
                            probas.query(f"time=='t_{time}'").survival_probability
                        )
                    )
                    > self.probability_threshold,
                    pos_label=1,
                    average="weighted" if weighted else "binary",
                )
            )
        return f1

    def get_f1_scores(self, label_type: str, weighted: bool):
        """
        Calculates dynamic f1 scores for
        train and test sets with 95% HDI intervals.

        Parameters
        ----------
        label_type: str
            can take values "==t" for using the observed event
            at time t, or "<=t" for using the observed
            event at up to time t.
        weighted : bool
            if set to true then macro f1-score is calculated,
            otherwise f1-score for minority class is calculated.

        Returns
        -------
            list
                list of floats

        """
        f1_score_train_low = self._get_dynamic_f1_score(
            self.merged_train, self.surv_probas_train[1], label_type, weighted
        )
        f1_score_train_median = self._get_dynamic_f1_score(
            self.merged_train, self.surv_probas_train[0], label_type, weighted
        )
        f1_score_train_high = self._get_dynamic_f1_score(
            self.merged_train, self.surv_probas_train[2], label_type, weighted
        )
        if self.merged_test.empty:
            return [f1_score_train_low, f1_score_train_median, f1_score_train_high]
        else:
            f1_score_test_low = self._get_dynamic_f1_score(
                self.merged_test, self.surv_probas_test[1], label_type, weighted
            )

            f1_score_test_median = self._get_dynamic_f1_score(
                self.merged_test, self.surv_probas_test[0], label_type, weighted
            )
            f1_score_test_high = self._get_dynamic_f1_score(
                self.merged_test, self.surv_probas_test[2], label_type, weighted
            )

            return (
                [f1_score_train_low, f1_score_train_median, f1_score_train_high],
                [f1_score_test_low, f1_score_test_median, f1_score_test_high],
            )

    def plot_f1_scores(self, weighted: bool):
        """
        Plots f1 scores for train and
        test sets for both real event definitions.

        Parameters
        ----------
        weighted : bool
            if set to true then macro f1-score is plotted,
            otherwise f1-score for minority class is plotted.
        """
        f1s_equ = self.get_f1_scores("==t", weighted)
        f1s_leq = self.get_f1_scores("<=t", weighted)

        if self.merged_test.empty:
            _ = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(16, 6))
            plt.plot(
                range(1, self.max_time_point + 1),
                f1s_equ[1],
                color="blue",
                label="train",
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                f1s_equ[0],
                f1s_equ[2],
                color="blue",
                alpha=0.3,
            )
            plt.title("Dynamic f1-score label defined churn at t=t")
            plt.legend()
            plt.show()

            plt.plot(
                range(1, self.max_time_point + 1),
                f1s_leq[1],
                color="blue",
                label="train",
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                f1s_leq[0],
                f1s_leq[2],
                color="blue",
                alpha=0.3,
            )
            plt.title("Dynamic f1-score label defined churn at t<=t")
            plt.legend()
            plt.show()
        else:
            _ = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(16, 6))
            plt.plot(
                range(1, self.max_time_point + 1),
                f1s_equ[0][1],
                color="blue",
                label="train",
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                f1s_equ[0][0],
                f1s_equ[0][2],
                color="blue",
                alpha=0.3,
            )

            plt.plot(
                range(1, self.max_time_point + 1),
                f1s_equ[1][1],
                color="orange",
                label=self.test_set_label,
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                f1s_equ[1][0],
                f1s_equ[1][2],
                color="orange",
                alpha=0.3,
            )

            plt.title("Dynamic f1-score label defined churn at t=t")
            plt.legend()
            plt.show()

            plt.plot(
                range(1, self.max_time_point + 1),
                f1s_leq[0][1],
                color="blue",
                label="train",
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                f1s_leq[0][0],
                f1s_leq[0][2],
                color="blue",
                alpha=0.3,
            )

            plt.plot(
                range(1, self.max_time_point + 1),
                f1s_leq[1][1],
                color="orange",
                label=self.test_set_label,
            )
            plt.fill_between(
                range(1, self.max_time_point + 1),
                f1s_leq[1][0],
                f1s_leq[1][2],
                color="orange",
                alpha=0.3,
            )

            plt.title("Dynamic f1-score label defined churn at t<=t")
            plt.legend()
            plt.show()

    def _plot_confusion_matrix(self, df: pd.DataFrame, extra_label: str):
        """
        This function accepts a data frame with the median and
        95% HDI of the predictions vs
        real class and outputs the confusion matrices.

        Parameters
        ----------
        df: pd.DataFrame
            pd.DataFrame that includes event col and predicted columns
        extra_label: str
            string to add extra labeling on title

        """

        cm_display_median = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                df[self.event_col], df[self.pred_median_col]
            ),
            display_labels=[False, True],
        )

        cm_display_low = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                df[self.event_col],
                df[self.pred_lower_col],
            ),
            display_labels=[False, True],
        )

        cm_display_high = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                df[self.event_col],
                df[self.pred_upper_col],
            ),
            display_labels=[False, True],
        )

        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(15, 5))
        cm_display_median.plot(ax=ax[0])
        cm_display_low.plot(ax=ax[1])
        cm_display_high.plot(ax=ax[2])
        ax[0].set_xlabel("survival_class, 50%")
        ax[0].set_ylabel("real_class")
        ax[1].set_xlabel(f"survival_class, {100 * (self.hpd_alpha / 2)}%")
        ax[1].set_ylabel("real_class")
        ax[2].set_xlabel(f"survival_class, {100 * (1 - self.hpd_alpha / 2)}%")
        ax[2].set_ylabel("real_class")
        plt.suptitle(f"Confusion matrix for {extra_label}")
        plt.show()

    def plot_confusion_matrices(self):
        """
        This function plots confusion
        matrices for train and test sets.
        """
        if self.merged_test.empty:
            self._plot_confusion_matrix(self.merged_train, "train set")
        else:
            self._plot_confusion_matrix(self.merged_train, "train set")
            self._plot_confusion_matrix(self.merged_test, f"{self.test_set_label} set")

    def _get_count_plot(self, _df: pd.DataFrame, extra_label: str):
        """
        This function plots the count distribution of observed
        time_to_event versus predicted with 95% HDI.

        Parameters
        ----------
        _df: pd.DataFrame
            pd.DataFrame that includes
            time_to_event column and predicted columns
        extra_label: str
            string to add extra labeling on title

        """
        fig, ax = plt.subplots(1, 4, sharey=True, sharex=False, figsize=(40, 5))
        _df = _df.replace(np.inf, self.max_time_point)
        sns.countplot(data=_df, x=self.time_to_event_col, hue=self.event_col, ax=ax[0])
        ax[0].set_xlabel(f"Real time labels {extra_label}")
        sns.countplot(data=_df, x=self.pred_median_col, hue=self.event_col, ax=ax[1])
        ax[1].set_xlabel(f"Predicted time labels median {extra_label}")
        sns.countplot(data=_df, x=self.pred_lower_col, hue=self.event_col, ax=ax[2])
        ax[2].set_xlabel(
            f"Predicted time labels {100 * (self.hpd_alpha / 2)}% {extra_label}"
        )
        sns.countplot(data=_df, x=self.pred_upper_col, hue=self.event_col, ax=ax[3])
        ax[3].set_xlabel(
            f"Predicted time labels {100 * (1 - self.hpd_alpha / 2)}% {extra_label}"
        )
        plt.show()

    def plot_count_plots(self):
        """
        This function plots the count distribution of observed
        time_to_event versus predicted with 95% HDI
        for train and test sets.
        """
        if self.merged_test.empty:
            self._get_count_plot(self.merged_train, "train set")
        else:
            self._get_count_plot(self.merged_train, "train set")
            self._get_count_plot(self.merged_test, f"{self.test_set_label} set")
