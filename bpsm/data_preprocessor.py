import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from pydantic import BaseModel, confloat, conint
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor(BaseModel):
    """
    This class preprocesses a feature set by splitting
    into training, validation and test sets, applying Min-Max
    scaling, SMOTE if desired and returns the scaled
    train, validation and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        pandas.DataFrame of features and target.
    train_date_range: list
        list of strings indicating the start and end of the train set period
    validation_date_range: list
        list of strings indicating the start and end of the validation set period
    test_date_range: list
        list of strings indicating the start and end of the test set period
    time_to_event_col: str
        time to event column name
    time_to_event_cutoff: int
        cut-off point for survival model. Any users that churned or didn't churn
        after that point will be considered censored at time_to_event_cutoff.
    event_col_name: str
        event column name
    date_col_name: str
        date column name
    smote: bool
        boolean indicating where over/undersampling should be done
        for balancing event_col
    seed: int
        integer for reproducibility of smote

    """

    df: pd.DataFrame
    train_date_range: List[str]
    validation_date_range: List[str]
    test_date_range: List[str]
    time_to_event_col: str
    time_to_event_cutoff: conint(ge=1)
    event_col_name: str
    user_id_col: str
    date_col_name: str
    drop_columns: List[str]
    categorical_columns: List[str]
    scaler: Optional[MinMaxScaler]
    smote: Optional[bool] = False
    sample_train_frac: Optional[confloat(gt=0, le=1)] = 1
    seed: Optional[int] = 11

    class Config:
        arbitrary_types_allowed = True

    def _drop_features(self):
        """
        Drops features specified in the class

        Returns
        -------
            pd.DataFrame
        """
        return self.df.drop(self.drop_columns, axis=1)

    def _preprocess_label(self):
        """
        Creates the event column based on the time to
        event cut off point. NaNs in time to event
        and any time to event above the cut-off is marked
        as event = 0 and time_to_event = time_to_event_cut_off.
        """

        self.df[self.time_to_event_col] = np.where(
            self.df[self.time_to_event_col] > self.time_to_event_cutoff,
            np.nan,
            self.df[self.time_to_event_col],
        )
        # There is time to event only if the user churned.
        # nan means that the user is right censored from t=time_to_event_cutoff

        self.df[self.event_col_name] = ~self.df[self.time_to_event_col].isna()
        self.df[self.event_col_name] = self.df[self.event_col_name].astype(np.int64)

        # If nan it means right-censored.
        # Fill time to event censored users based on time_to_event_cutoff
        self.df[self.time_to_event_col] = self.df[self.time_to_event_col].fillna(
            self.time_to_event_cutoff
        )

    def _split_data(self):
        """
        Splits the data into train, validation and test sets.

        Returns
        ----------
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
        """
        self.df = self._drop_features()
        return (
            self.df[self.df[self.date_col_name].between(*self.train_date_range)].sample(
                frac=self.sample_train_frac
            ),
            self.df[self.df[self.date_col_name].between(*self.validation_date_range)],
            self.df[self.df[self.date_col_name].between(*self.test_date_range)],
        )

    def _get_smote_oversampling(self, X: pd.DataFrame, y: pd.Series):
        """
        Apply Synthetic Minority Over-sampling Technique as presented in
        N. V. Chawla et al. (2002).

        Parameters
        ----------
        X: pd.DataFrame
            feature set of train data
        y: pd.Series
            event column of train data

        Returns
        ----------
            pandas.DataFrame, pandas.DataFrame
        """
        smote = SMOTE(random_state=self.seed)
        X_oversample, y_oversample = smote.fit_resample(X, y)
        y_oversample = pd.concat(
            [X_oversample[self.time_to_event_col], y_oversample], axis=1
        )
        y_oversample[f"{self.time_to_event_col}_round"] = (
            y_oversample[self.time_to_event_col].astype(float).round()
        )
        y_oversample[self.event_col_name] = y_oversample[self.event_col_name].astype(
            int
        )
        return X_oversample, y_oversample

    def _scale_train_data(self, X: DataFrame) -> DataFrame:
        """
        Fits and transforms train set
        with min max scaling.

        Parameters
        ----------
        X: pd.DataFrame
            feature set of train data

        Returns
        ----------
            pandas.DataFrame
        """
        self.scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X))
        X_scaled.columns = X.columns
        return X_scaled

    def _scale_test_data(self, X):
        """
        Transforms validation/test set
        with min max scaling.

        Parameters
        ----------
        X: pd.DataFrame
            feature set of validation/test data

        Returns
        ----------
            pandas.DataFrame
        """
        X_scaled = pd.DataFrame(self.scaler.transform(X))
        X_scaled.columns = X.columns
        return X_scaled

    def save_scaler(self, file_name):
        """
        Saves scaler as pickle

        Parameters
        ----------
        file_name: str
            File name or path to file name
        """
        with open(f"{file_name}.pkl", "wb") as file_pi:
            pickle.dump(self.scaler, file_pi)

    def return_scaled_sets_with_labels(self):
        """
        Transforms and returns scaled train, validation and test sets
        with min max scaling and target columns for event and time to
        event.
        """
        target_and_id_cols = [
            self.event_col_name,
            self.user_id_col,
            self.date_col_name,
            *self.categorical_columns,
        ]

        self._preprocess_label()
        df_train, df_validation, df_test = self._split_data()

        # Delete df to save space
        del self.df

        if self.smote:
            X_train, y_train = self._get_smote_oversampling(
                df_train.drop(target_and_id_cols, axis=1), df_train[self.event_col_name]
            )
        else:
            X_train = pd.concat(
                [
                    df_train[
                        [
                            self.user_id_col,
                            self.date_col_name,
                            *self.categorical_columns,
                        ]
                    ].reset_index(drop=True),
                    self._scale_train_data(
                        df_train.drop(target_and_id_cols, axis=1)
                    ).reset_index(drop=True),
                ],
                axis=1,
            )
            y_train = df_train[[self.time_to_event_col, self.event_col_name]]

        X_val = pd.concat(
            [
                df_validation[
                    [self.user_id_col, self.date_col_name, *self.categorical_columns]
                ].reset_index(drop=True),
                self._scale_test_data(df_validation.drop(target_and_id_cols, axis=1)),
            ],
            axis=1,
        )
        X_test = pd.concat(
            [
                df_test[
                    [self.user_id_col, self.date_col_name, *self.categorical_columns]
                ].reset_index(drop=True),
                self._scale_test_data(df_test.drop(target_and_id_cols, axis=1)),
            ],
            axis=1,
        )

        return (
            X_train,
            y_train.reset_index(drop=True),
            X_val,
            df_validation[[self.time_to_event_col, self.event_col_name]].reset_index(
                drop=True
            ),
            X_test,
            df_test[[self.time_to_event_col, self.event_col_name]].reset_index(
                drop=True
            ),
        )
