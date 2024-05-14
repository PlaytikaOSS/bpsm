import aesara.tensor as at
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import conint

from .bayesian_survival import BayesianSurvivalModel


class BayesianSurvivalSimple(BayesianSurvivalModel):
    def _specify_model(
        self,
        n_covars: conint(ge=1),
        hyper_param_lambda0: list[float],
        X_train: pd.DataFrame,
        event: np.array,
        exposure: np.array,
    ):
        """
        Constructs the model hierarchy of the Bayesian piecewise survival
        model without reguralisation
        https://wiki.playtika.com/display/AILAB/Bayesian+Implementation

        Parameters
        ----------
        n_covars: int
            number of covariates/features
        hyper_param_lambda0: list[float]
            list of positive floats indicating the hyperparameter priors for
            baseline hazard
        X_train: pd.DataFrame
            feature set
        event: np.array
            array matrix specifying the event flag
            for each user at each time point
        exposure: np.array
            array matrix specifying whether
            the user is participating in each time point.
        Returns
        -------
            pm.Model
        """
        super()._specify_model(n_covars, hyper_param_lambda0, X_train, event, exposure)

        with pm.Model() as model:
            # Prior for piecewise hazards
            lambda0 = pm.Gamma(
                "lambda0",
                hyper_param_lambda0[0],
                hyper_param_lambda0[1],
                shape=self.n_intervals,
            )

            # Non-infrmative prior for precision
            precision = pm.Gamma("precision", 0.001, 1000, shape=n_covars)
            # Transform precision to variance
            sigma = pm.Deterministic("sigma", precision**-2)

            # Centered at 0 with feature specific variances
            beta = pm.MvNormal("beta", 0, sigma * np.eye(n_covars), shape=(1, n_covars))

            # Partial hazard calculation
            lambda_ = pm.Deterministic(
                "lambda_", at.outer(at.exp(pm.math.dot(X_train, beta.T)), lambda0)
            )
            # Mean churn events by time
            # mu = t_ij * lambdaij
            mu = pm.Deterministic("mu", exposure * lambda_)

            # likelihood
            _ = pm.Poisson("obs", mu, observed=event)
        return model

    def get_model_specific_params(
        self, n_samples: conint(ge=0, le=1), parameters_dict: dict
    ):
        """
        Updates the posterior parameters' dictionary

        Parameters
        ----------
        n_samples: conint
            number of samples to extract from posterior distributions
        parameters_dict: dict
            dictionary incorporating the common parameters
        Returns
        -------
        dict
        """
        super().get_model_specific_params(n_samples, parameters_dict)

        sigmas = self.idata.posterior["sigma"].values.reshape(
            n_samples, len(self.feature_names)
        )
        parameters_dict["sigmas_sq"] = sigmas
        return parameters_dict
