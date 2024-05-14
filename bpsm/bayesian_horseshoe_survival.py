import aesara.tensor as at
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import confloat, conint

from .bayesian_survival import BayesianSurvivalModel


class BayesianHorseshoeSurvival(BayesianSurvivalModel):
    def _specify_model(
        self,
        n_covars: int,
        hyper_param_lambda0: list[confloat(gt=0)],
        X_train: pd.DataFrame,
        event: np.array,
        exposure: np.array,
    ):
        """
        Constructs the model hierarchy of the Bayesian piecewise survival
        model with Horseshoe regularisation defined in
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
            matrix specifying the event flag
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

            # Feature specific variance regularisation
            omega_1 = pm.HalfCauchy("omega_1", beta=1, shape=n_covars)
            # Global regularisation
            omega_2 = pm.HalfCauchy("omega_2", beta=1)
            # get the vector of variances
            sigma = pm.Deterministic("horseshoe", omega_1 * omega_1 * omega_2 * omega_2)

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

    def get_model_specific_params(self, n_samples: conint(ge=1), parameters_dict: dict):
        """
        Updates the posterior parameters' dictionary

        Parameters
        ----------
        n_samples: int
            number of samples to extract from posterior distributions
        parameters_dict: dict
            dictionary incorporating the common parameters
        Returns
        -------
        dict
        """
        super().get_model_specific_params(n_samples, parameters_dict)

        omegas_1 = self.idata.posterior["omega_1"].values.reshape(
            n_samples, len(self.feature_names)
        )
        omega_2 = self.idata.posterior["omega_2"].values.reshape(n_samples)
        parameters_dict["omegas_local"] = omegas_1
        parameters_dict["omega_global"] = omega_2
        return parameters_dict
