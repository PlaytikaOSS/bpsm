import collections
from itertools import combinations
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import scipy
from dython.nominal import associations
from lifelines import CoxPHFitter
from loguru import logger
from pydantic import BaseModel, confloat, conint

METHOD = Literal["stepwise", "sparse_correlation", "correlation"]
CORRELATION = Literal["pearson", "spearman"]


class FeatureSelector(BaseModel):
    """

    Parameters
    ----------
        df : pd.DataFrame
            dataset including features and target columns
        time_to_event_col: str
            string specifying time to event column
        event_col: str
            string specifying event column
        method: Literal
            feature selection method to be used. It can be
            stepwise, sparse_correlation or correlation
        correlation_method: Literal
            correlation method to be used if either sparse_correlation or correlation
            methods are used for feature selection
        corr_cutoff: confloat
            float between 0 and 1 indicating the correlation
            threshold for the features to be dropped
            if either sparse_correlation or correlation
            methods are used for feature selection.
            It will remove one of pairs of features with
            a correlation greater than this value.
        p_val_thresh: float, between 0 and 1
            consider a feature significant or correlation significant
            if its p-value < p_val_thresh
        features_to_keep: list[str]
            list of features to keep and not contact
            any selection
        initial_list: list[str]
            features to be used initially in stepwise
            selection if stepwise method is used.
        threshold_in: confloat
            float between 0 and 1 indicating the p-value
            in stepwise selection for a feature to be considered
            significant. It is used only if method is stepwise.
            Default value will be set to 0.01.
            Always set threshold_in < threshold_out to avoid infinite looping.
            See https://en.wikipedia.org/wiki/Stepwise_regression for the details.
        threshold_out: confloat
            float between 0 and 1 indicating the p-value
            in stepwise selection for a feature to be dropped as non-significant
            significant. It is used only if method is stepwise.
            Default value will be set to 0.05.
            Always set threshold_in < threshold_out to avoid infinite looping.
            See https://en.wikipedia.org/wiki/Stepwise_regression for the details.
        n_vars_drop: conint
            positive integer indicating number of features to be dropped
            if correlation method is used.
    """

    df: pd.DataFrame
    time_to_event_col: str
    event_col: str
    use_univariate_model: Optional[bool] = False
    method: METHOD
    features_to_keep: Optional[List[str]]
    initial_list: Optional[List[str]]
    correlation_method: Optional[CORRELATION]
    corr_cutoff: Optional[confloat(ge=0, le=1)]
    p_val_thresh: Optional[confloat(ge=0, le=1)] = 0.05
    threshold_in: Optional[confloat(ge=0, le=1)] = 0.01
    threshold_out: Optional[confloat(ge=0, le=1)] = 0.05
    n_vars_drop: Optional[conint(ge=1)]

    class Config:
        arbitrary_types_allowed = True

    def feature_selection(self):
        """
        Conducts feature selection

        Returns
        -------
            list[str]
        List of features to be kept
        """
        if self.use_univariate_model:
            logger.info("Starting univarite model tests")
            feats = self.remove_non_significant_features(
                self.df
                if not self.features_to_keep
                else self.df.drop(self.features_to_keep, axis=1)
            )
        else:
            feats = (
                self.df.columns
                if not self.features_to_keep
                else self.df.drop(self.features_to_keep, axis=1).columns
            )
        if self.method == "stepwise":
            logger.info("Starting stepwise selection")
            feats = self.stepwise_selection(
                self.df[feats], decision_metric="p", verbose=True
            )
        elif self.method == "sparse_correlation":
            logger.info("Removing correlated features")
            remove_features = self.find_correlation_sparse(
                self.df[feats].drop([self.time_to_event_col, self.event_col], axis=1)
            )
            feats = [c for c in self.df.columns if c not in remove_features]
        elif self.method == "correlation":
            logger.info("Removing correlated features")
            remove_features = self.find_correlation(
                self.df[feats].drop([self.time_to_event_col, self.event_col], axis=1)
            )
            feats = [
                c
                for c in self.df.columns
                if c not in remove_features + [self.time_to_event_col, self.event_col]
            ]
        else:
            raise ValueError(f"Feature selection method {self.method} not supported.")

        # Delete df to save space
        del self.df
        return feats

    def remove_non_significant_features(self, df: pd.DataFrame):
        """
        Perform feature selection
        based on p-value from CoxPH fitter.
        If p-value greater than threshold then the
        features are removed.

        Parameters
        ----------
        df: pd.DataFrame

        Return
        ------
        Returns: list
            list of selected features

        """
        sign_feats = []
        for feat in df.columns:
            try:
                # Define univariate cox model
                mod = CoxPHFitter()
                mod.fit(
                    df[[feat] + [self.time_to_event_col, self.event_col]],
                    duration_col=self.time_to_event_col,
                    event_col=self.event_col,
                )
                # Extract p-value
                p_val_feat = mod.summary.p.loc[feat]
                exp_coef = mod.summary["exp(coef)"].loc[feat]
                # Append if feature is significant
                if p_val_feat < self.p_val_thresh:
                    sign_feats.append(feat)
                    logger.info(
                        f"{feat} significant with p val {round(p_val_feat, 3)} "
                        f"and  exp(coef) {round(exp_coef, 3)}"
                    )
            except:  # noqa: E722
                logger.info(f"{feat} couldn't be modeled")
        return sign_feats

    def stepwise_selection(self, df: pd.DataFrame, decision_metric="p", verbose=True):
        """
        Perform a forward-backward feature selection
        based on p-value from standard CoxPH fitter.

        Parameters
        ----------
            df : pandas.DataFrame
                pandas.DataFrame with candidate features
            decision_metric: str
                p specifying p-value. Will be extended to
                accept AIC, partial log-likelihood or
                concordance index.
            verbose: bool
                whether to print the sequence of inclusions and exclusions

        Return
        ------
        Returns: list
            list of selected features

        """
        if self.initial_list is None:
            self.initial_list = []
        included = list(self.initial_list)
        cols_for_step = list(
            set(df.columns) - set([self.time_to_event_col, self.event_col])
        )
        while True:
            changed = False
            # forward step
            excluded = list(set(cols_for_step) - set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                try:
                    # Define model
                    model = CoxPHFitter()
                    # fit model with new column
                    model.fit(
                        df[
                            included
                            + [self.time_to_event_col, self.event_col]
                            + [new_column]
                        ],
                        duration_col=self.time_to_event_col,
                        event_col=self.event_col,
                    )
                    # extract p-value
                    new_pval[new_column] = model.summary[decision_metric].loc[
                        new_column
                    ]
                except:  # noqa: E722
                    continue
            # detect best p-value
            best_pval = new_pval.min()

            # include feature if satisfies threshold_in rule
            if best_pval < self.threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(
                        "Add  {:30} with p-value {:.6}".format(best_feature, best_pval)
                    )

            # Backward step
            # Define again the model
            model = CoxPHFitter()

            # Fit included features
            model.fit(
                df[included + [self.time_to_event_col, self.event_col]],
                duration_col=self.time_to_event_col,
                event_col=self.event_col,
            )

            pvalues = model.summary[decision_metric]

            # Extract worst feature
            worst_pval = pvalues.max()  # null if pvalues is empty

            # If feature is not significant anymore, drop it
            if worst_pval > self.threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print(
                        "Drop {:30} with p-value {:.6}".format(
                            worst_feature, worst_pval
                        )
                    )
            # Stop process if features list doesn't change
            if not changed:
                break
        return included

    def find_correlation_sparse(self, df: pd.DataFrame):
        """
        Find highly correlated features,
        and return a list of features to remove

        Parameters
        ----------
        df : pandas.DataFrame
            pandas.DataFrame with candidate features
        """

        cat = df.select_dtypes(
            include="category"
        ).columns.tolist()  # categorical features

        if cat:
            # dython correlation matrix
            qtt = [f for f in df.columns.tolist() if f not in cat]
            # Convert all the columns in float to integer for correlation plot and
            # category / boolean to object
            df.loc[:, qtt] = df.loc[:, qtt].astype(int)
            df.loc[:, cat] = df.loc[:, cat].astype(object)
            corrMatrix = associations(df, compute_only=True)["corr"]
        else:
            # correlation matrix for numeric features
            corrMatrix = np.abs(self.df.corr(method=self.correlation_method))

        corrMatrix.loc[:, :] = np.tril(corrMatrix, k=-1)

        already_in = set()
        result = []

        for col in corrMatrix:
            # drop features that satisfy the threshold rule
            perfect_corr = corrMatrix[col][
                corrMatrix[col] > self.p_val_thresh
            ].index.tolist()
            if perfect_corr and col not in already_in:
                already_in.update(set(perfect_corr))
                perfect_corr.append(col)
                result.append(perfect_corr)

        # Drop the first feature from correlated pairs
        select_nested = [f[1:] for f in result]
        select_flat = [i for j in select_nested for i in j]
        return select_flat

    def _get_pearson_correlation(self, df: pd.DataFrame, pairs):
        """
        This function accepts a dataframe and a list of
        pairs of features within that dataframe
        and calculates pairwise pearson correlation
        coefficients between features and average
        absolute correlations of a feature with respect
        the rest.

        Parameters
        ----------
        df : pandas.DataFrame
            pandas.DataFrame with candidate features
        pairs: list
            list of combined features in pairs of 2.

        Returns
        -------
            dict, list
        dictionary with pairwise correlation,
        list of average correlations per feature
        """

        all_cors_by_pair = collections.defaultdict(list)
        corrs = dict()

        for pair in pairs:
            # Calculate pearson correlation coefficient and p-val
            corr, p_val = scipy.stats.pearsonr(df[pair[0]], df[pair[1]])

            # Save the correlation with indexing each feature
            all_cors_by_pair[pair[0]].append(np.abs(corr))
            all_cors_by_pair[pair[1]].append(np.abs(corr))

            # Create dictionary with absolute correlations
            corrs[f"{pair[0]}-{pair[1]}"] = {
                "var_1": pair[0],
                "var_2": pair[1],
                "correlation": corr,
                "absolute_correlation": np.abs(corr),
                "p_val": p_val,
                # Is correlation higher that the cutoff?
                "above_threshold": np.abs(corr) > self.corr_cutoff,
                "significant": p_val < self.p_val_thresh,
            }

        return corrs, all_cors_by_pair

    def _get_spearman_correlation(self, df: pd.DataFrame, pairs):
        """
        This function accepts a dataframe and a list of
        pairs of features within that dataframe
        and calculates pairwise spearman correlation
        coefficients between features and average
        absolute correlations of a feature with respect
        the rest.

        Parameters
        ----------
        df : pandas.DataFrame
            pandas.DataFrame with candidate features
        pairs: list
            list of combined features in pairs of 2.

        Returns
        -------
            dict, list
        dictionary with pairwise correlation,
        list of average correlations per feature
        """
        all_cors_by_pair = collections.defaultdict(list)
        corrs = dict()

        for pair in pairs:
            # Calculate spearman correlation coefficient and p-val
            corr, p_val = scipy.stats.spearmanr(df[pair[0]], df[pair[1]])

            # Save the correlation with indexing each feature
            all_cors_by_pair[pair[0]].append(np.abs(corr))
            all_cors_by_pair[pair[1]].append(np.abs(corr))

            # Create dictionary with absolute correlations
            corrs[f"{pair[0]}-{pair[1]}"] = {
                "var_1": pair[0],
                "var_2": pair[1],
                "correlation": corr,
                "absolute_correlation": np.abs(corr),
                "p_val": p_val,
                # Is correlation higher that the cutoff?
                "above_threshold": np.abs(corr) > self.corr_cutoff,
                "significant": p_val < self.p_val_thresh,
            }

            return corrs, all_cors_by_pair

    def find_correlation(self, df: pd.DataFrame):
        """
        This function accepts a dataframe
        and calculates pairwise spearman or pearson correlation
        coefficients between features and average
        absolute correlations of a feature with respect
        the rest. Then based on a correlation threshold and
        how many features we want to drop, it returns a list of
        the most correlated features to drop.

        Parameters
        ----------
        df: pd.DataFrame
            pd.DataFrame including features

        Returns
        -------
           list[str]
            list of feature names to drop
        """

        # Get all possible pairwise combinations

        pairs = list(combinations(df.columns, 2))

        if self.correlation_method == "pearson":
            corrs, all_cors_by_pair = self._get_pearson_correlation(df, pairs)
        elif self.correlation_method == "spearman":
            corrs, all_cors_by_pair = self._get_spearman_correlation(df, pairs)
        else:
            raise ValueError(
                f"Correlation method {self.correlation_method} not supported."
            )

        # Get average absolute correlation per feature
        avg_corrs = (
            pd.DataFrame.from_dict(all_cors_by_pair)
            .mean(axis=0)
            .reset_index()
            .rename(columns={"index": "variable", 0: "avg_corr"})
            .sort_values("avg_corr", ascending=False)
        )
        # Fetch pairs with correlation > threshold
        corrs_above_threshold = pd.DataFrame.from_dict(corrs).T.query(
            "above_threshold==True"
        )

        # Merge pairwise correlations with averages per feature
        combined_mat = (
            corrs_above_threshold.merge(
                avg_corrs, left_on="var_1", right_on="variable", how="left"
            )
            .merge(
                avg_corrs,
                left_on="var_2",
                right_on="variable",
                how="left",
                suffixes=("_var_1", "_var_2"),
            )
            .sort_values("absolute_correlation", ascending=False)
        )

        # Calculate which is the most multicollinear variable among the pair based on
        # the average absolute correlation
        combined_mat["drop_var"] = combined_mat.apply(
            lambda row: row["var_1"]
            if row["avg_corr_var_1"] > row["avg_corr_var_2"]
            else row["var_2"],
            axis=1,
        )

        # Drop features until we reach the n_vars_drop
        drop_vars = []
        i = 0
        while (
            len(drop_vars) < self.n_vars_drop
            and len(drop_vars) < combined_mat.shape[0]
            and i <= len(drop_vars)
        ):
            var = combined_mat.iloc[i]["drop_var"]
            if var not in drop_vars:
                drop_vars.append(var)
                print(f"Dropping {var}")
            i += 1
        return drop_vars
