import gc
import os

import fire
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from matplotlib import pyplot as plt

from bpsm.bayesian_horseshoe_survival import BayesianHorseshoeSurvival
from bpsm.bayesian_survival_simple import BayesianSurvivalSimple
from bpsm.data_preprocessor import DataPreprocessor
from bpsm.evaluation import Evaluation
from bpsm.feature_selector import FeatureSelector
from bpsm.plot_bayesian_survival import PlotBayesianSurvival
from bpsm.script_config import ScriptConfig


def train_survival_model(config_file: str):
    """
    It iterates over different hyperparameters of baseline
    hazard, fits the model and extracts plots for
    train and validation sets. If desired, it extracts for
    the test set and saves the predictions.

    Parameters
    ----------
    config_file: str
        Path to the config file of the experiments that you want to do
    """
    # read config file
    logger.info("Config read")

    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    config = ScriptConfig(**yaml_config)
    logger.info(config)

    logger.info("Reading data")
    data = pd.read_parquet(config.data_input_path).sample(frac=config.data_sample_frac)
    logger.info("Reading data finished")

    categorical_cols = []

    for col in data.columns:
        for feat in config.categorical_features:
            if col.startswith(feat + "_lag"):
                categorical_cols.append(col)
                break

    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor(
        df=data,
        train_date_range=config.train_dates,
        validation_date_range=config.validation_dates,
        test_date_range=config.test_dates,
        time_to_event_col=config.time_to_event_column,
        time_to_event_cutoff=config.time_to_event_cutoff,
        event_col_name=config.event_column,
        user_id_col=config.id_column,
        date_col_name=config.date_column,
        categorical_columns=categorical_cols,
        drop_columns=config.drop_columns,
        smote=False,
        sample_train_frac=1,
        seed=11,
    )

    if not config.use_test_set:
        (
            X_train,
            y_train,
            X_validation,
            y_validation,
            _,
            _,
        ) = preprocessor.return_scaled_sets_with_labels()
    else:
        (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        ) = preprocessor.return_scaled_sets_with_labels()
    logger.info("Data preprocessing finished")

    # Delete df to save space
    del data

    if config.apply_feature_selection:
        logger.info("Starting feature selection")
        features_to_model = FeatureSelector(
            df=pd.concat([X_train, y_train], axis=1).drop(
                [config.id_column, config.date_column], axis=1
            ),
            time_to_event_col=config.time_to_event_column,
            event_col=config.event_column,
            use_univariate_model=config.feature_selection_args["use_univariate_model"],
            method=config.feature_selection_args["method"],
            features_to_keep=config.features_select,
            correlation_method=config.feature_selection_args["correlation_method"],
            corr_cutoff=config.feature_selection_args["corr_cutoff"],
            n_vars_drop=config.feature_selection_args["n_vars_drop"],
        ).feature_selection()
        features_to_model = features_to_model + config.features_select
        logger.info("Final features selected:{}".format(features_to_model))
    else:
        features_to_model = config.features_select
        logger.info("Features to be considered:{}".format(features_to_model))

    for experiment, hyperparams in config.hyper_param_lambda0.items():
        logger.info(
            "Starting experiment {} with shape={} and rate={}".format(
                experiment, hyperparams[0], hyperparams[1]
            )
        )
        if config.model_type == "regularised":
            logger.info("Specifying Regularised Horseshoe")
            model = BayesianHorseshoeSurvival()
        elif config.model_type == "simple":
            logger.info("Specifying non regularised model")
            model = BayesianSurvivalSimple()
        else:
            raise ValueError(f"Model {config.model_type} not supported")

        logger.info("Model Specified")

        experiment_path = config.output_path + experiment

        os.makedirs(f"{experiment_path}/", exist_ok=True)
        logger.info("Fitting model")
        model.fit(
            X=X_train[features_to_model],
            y=y_train,
            hyper_param_lambda0=hyperparams,
            interval_length=1,
            time_to_event_col=config.time_to_event_column,
            event_col=config.event_column,
            n_its=config.n_its,
            n_samples=config.n_samples,
        )

        plot_obj = PlotBayesianSurvival(
            model=model, model_name="horseshoe", hpd_alpha=config.hpd_alpha
        )

        os.makedirs(f"{experiment_path}/plots", exist_ok=True)

        # Plot coefficients contribution
        logger.info("Saving forest plot")
        f = plot_obj.plot_forest_betas()
        f.savefig(experiment_path + "/plots/forest_betas.png", bbox_inches="tight")
        f.clf()
        logger.info("Saving baseline hazard plot")
        # Plot baseline hazard for all time points
        f = plot_obj.plot_baseline_time_points("hazard")
        f.savefig(experiment_path + "/plots/baseline_hazard.png", bbox_inches="tight")
        f.clf()
        # Plot survival baseline for all time points
        logger.info("Saving baseline survival plot")
        f = plot_obj.plot_baseline_time_points("survival")
        f.savefig(experiment_path + "/plots/baseline_survival.png", bbox_inches="tight")
        f.clf()
        # Plot baseline survival with KM
        logger.info("Saving baseline survival with KM plot")
        f = plot_obj.plot_baseline_survival_with_KM(
            y_train, config.time_to_event_column, config.event_column
        )
        f.savefig(
            experiment_path + "/plots/baseline_survival_with_KM.png",
            bbox_inches="tight",
        )
        f.clf()

        for feature, values in config.continuous_partial_effects.items():
            if feature in features_to_model:
                # Plot partial effects on survival
                logger.info("Saving partial effects for {}".format(feature))
                plot_obj.plot_baseline_vs_partial_effects(
                    feature=feature,
                    values=values,
                    type_var="continuous",
                    min_max_scaler=preprocessor.scaler,
                    num_feats=list(
                        X_train.drop(
                            [config.id_column, config.date_column, *categorical_cols],
                            axis=1,
                        ).columns
                    ),
                )

                plt.savefig(
                    experiment_path + f"/plots/partial_effects_{feature}.png",
                    bbox_inches="tight",
                )
                plt.clf()

        for feature in config.categorical_partial_effects:
            if feature in features_to_model:
                logger.info("Saving partial effects for {}".format(feature))
                plot_obj.plot_baseline_vs_partial_effects(
                    feature=feature,
                    values=[1],
                    type_var="categorical",
                    min_max_scaler=preprocessor.scaler,
                    num_feats=list(
                        X_train.drop(
                            [config.id_column, config.date_column, *categorical_cols],
                            axis=1,
                        ).columns
                    ),
                )
                plt.savefig(
                    experiment_path + f"/plots/partial_effects_{feature}.png",
                    bbox_inches="tight",
                )
                plt.clf()

        logger.info("Calculating predictions for train and validation sets")
        # Predict
        (
            predicted_survival_probabilities,
            predicted_hpd_survival,
            predicted_hpd_time_to_event,
        ) = model.predict(X_train[features_to_model])

        (
            predicted_survival_probabilities_val,
            predicted_hpd_survival_val,
            predicted_hpd_time_to_event_val,
        ) = model.predict(X_validation[features_to_model])

        # Plot player trajectories (train set)
        example_ids = np.random.choice(
            X_train.reset_index()["index"].unique(), size=config.n_users_trajectories
        )
        plot_obj.plot_posterior_probas_for_n_users(
            predicted_survival_probabilities, example_ids, "survival"
        )
        plt.savefig(
            experiment_path + "/plots/n_users_example_distributions.png",
            bbox_inches="tight",
        )
        plt.clf()

        logger.info(
            "Saving trajectory plots for {} users".format(config.n_users_trajectories)
        )
        plot_obj.plot_surv_trajectory_n_users(
            [
                predicted_hpd_survival[0],
                predicted_hpd_survival[1],
                predicted_hpd_survival[2],
            ],
            example_ids,
        )
        plt.savefig(
            experiment_path + "/plots/n_users_example_trajectories.png",
            bbox_inches="tight",
        )
        plt.clf()

        # Plot evaluation between train and validation, train and test
        plot_eval_obj = Evaluation(
            y_train_predicted=predicted_hpd_time_to_event,
            y_train_real=y_train.reset_index(drop=True),
            y_test_predicted=predicted_hpd_time_to_event_val,
            y_test_real=y_validation.reset_index(drop=True),
            surv_probas_train=predicted_hpd_survival,
            surv_probas_test=predicted_hpd_survival_val,
            time_to_event_col=config.time_to_event_column,
            event_col=config.event_column,
            pred_median_col="",
            pred_lower_col="",
            pred_upper_col="",
            probability_threshold=0.5,
            hpd_alpha=config.hpd_alpha,
            test_set_label="validation",
        )
        logger.info("Calculating concordances for train and validation sets")
        concord_train, concord_val = plot_eval_obj.get_concordance()

        logger.info("Saving dynamic brier score")
        plot_eval_obj.plot_brier_scores()
        plt.savefig(
            experiment_path + "/plots/hpd_dynamic_brier_score_w_validation.png",
            bbox_inches="tight",
        )
        plt.clf()

        # delete data to save space
        del (
            predicted_survival_probabilities_val,
            predicted_hpd_survival_val,
            predicted_hpd_time_to_event_val,
        )

        gc.collect()

        if config.use_test_set:
            logger.info("Calculating predictions for test set")
            (
                predicted_survival_probabilities_test,
                predicted_hpd_survival_test,
                predicted_hpd_time_to_event_test,
            ) = model.predict(X_test[features_to_model])

            plot_eval_obj = Evaluation(
                y_train_predicted=predicted_hpd_time_to_event,
                y_train_real=y_train.reset_index(drop=True),
                y_test_predicted=predicted_hpd_time_to_event_test,
                y_test_real=y_test.reset_index(drop=True),
                surv_probas_train=predicted_hpd_survival,
                surv_probas_test=predicted_hpd_survival_test,
                time_to_event_col=config.time_to_event_column,
                event_col=config.event_column,
                pred_median_col="",
                pred_lower_col="",
                pred_upper_col="",
                probability_threshold=0.5,
                hpd_alpha=config.hpd_alpha,
                test_set_label="test",
            )
            logger.info("Calculating concordance for test set")
            _, concord_test = plot_eval_obj.get_concordance()
            logger.info("Saving dynamic brier score")
            briers = plot_eval_obj.get_brier_scores()
            plot_eval_obj.plot_brier_scores()
            plt.savefig(
                experiment_path + "/plots/hpd_dynamic_brier_score_w_test.png",
                bbox_inches="tight",
            )
            plt.clf()

        del plot_eval_obj

        logger.info("Calculating summaries with HPD for the parameters")

        lower, upper = config.hpd_alpha / 2, 1 - config.hpd_alpha / 2
        summary_base_survival = (
            pd.DataFrame(model.posterior_parameters["baseline_survival_probability"])
            .quantile([0.5, lower, upper])
            .T
        )

        results = {
            "hazard_hyperparameters": {"shape": hyperparams[0], "rate": hyperparams[1]},
            "baseline_cumulative_survival": summary_base_survival.to_dict("index"),
            "betas": model.posterior_parameters["summary_betas"].to_dict("index"),
            "exp_betas": model.posterior_parameters["summary_exp_betas"].to_dict(
                "index"
            ),
            "concordance_train": pd.DataFrame(concord_train).to_dict("records"),
            "concordance_validation": pd.DataFrame(concord_val).to_dict("records"),
        }

        if config.model_type == "regularised":
            summary_omega_global = (
                pd.DataFrame(model.posterior_parameters["omega_global"])
                .quantile([0.5, lower, upper])
                .T
            )
            summary_omega_local = (
                pd.DataFrame(model.posterior_parameters["omegas_local"])
                .quantile([0.5, lower, upper])
                .T
            )
            summary_omega_local.index = model.feature_names

            summary_base_survival.columns = [
                "median",
                f"{100 * lower}%",
                f"{100 * upper}%",
            ]
            summary_omega_global.columns = [
                "median",
                f"{100 * lower}%",
                f"{100 * upper}%",
            ]
            summary_omega_local.columns = [
                "median",
                f"{100 * lower}%",
                f"{100 * upper}%",
            ]
            results.update(
                {
                    "local_variances": summary_omega_local.to_dict("index"),
                    "global_variance": summary_omega_global.to_dict("records"),
                }
            )
        else:
            summary_sigma_sq_local = (
                pd.DataFrame(model.posterior_parameters["sigmas_sq"])
                .quantile([0.5, lower, upper])
                .T
            )
            summary_sigma_sq_local.index = model.feature_names

            results.update({"sigmas": summary_sigma_sq_local.to_dict("index")})

        logger.info("Saving model")

        os.makedirs(experiment_path + "/model", exist_ok=True)

        model.save_model(experiment_path + "/model/model", False)
        preprocessor.save_scaler(experiment_path + "/model/scaler")

        # Delete to save space
        del model

        gc.collect()

        if config.use_test_set:
            results.update(
                {"concordance_test": pd.DataFrame(concord_test).to_dict("records")}
            )
            results.update(
                {
                    "brier_scores": {
                        "train": {
                            lower: pd.DataFrame(briers[0][0]).T.to_dict("index"),
                            "0.5": pd.DataFrame(briers[0][1]).T.to_dict("index"),
                            upper: pd.DataFrame(briers[0][2]).T.to_dict("index"),
                        },
                        "test_or_validation": {
                            lower: pd.DataFrame(briers[1][0]).T.to_dict("index"),
                            "0.5": pd.DataFrame(briers[1][1]).T.to_dict("index"),
                            upper: pd.DataFrame(briers[1][2]).T.to_dict("index"),
                        },
                    }
                }
            )

            logger.info("Preparing test set predictions")

            predicted_hpd_survival_test[0]["time_int"] = predicted_hpd_survival_test[0][
                "time"
            ].map(lambda x: int(x.split("_")[1]) + 1)
            predicted_hpd_survival_test[1]["time_int"] = predicted_hpd_survival_test[1][
                "time"
            ].map(lambda x: int(x.split("_")[1]) + 1)
            predicted_hpd_survival_test[2]["time_int"] = predicted_hpd_survival_test[2][
                "time"
            ].map(lambda x: int(x.split("_")[1]) + 1)

            predictions_merged = (
                X_test[[config.id_column, config.date_column]]
                .merge(y_test, left_index=True, right_index=True)
                .merge(
                    predicted_hpd_time_to_event_test.rename(
                        columns={
                            "50%": "time_to_event_median",
                            f"{100 * lower}%": "time_to_event_lower",
                            f"{100 * upper}%": "time_to_event_upper",
                        }
                    ),
                    left_index=True,
                    right_index=True,
                )
                .merge(
                    predicted_hpd_survival_test[0].rename(
                        columns={"survival_probability": "median_survival_probability"}
                    ),
                    left_on=["index", "time_to_event_median"],
                    right_on=["index", "time_int"],
                )
                .drop(["time_int", "time"], axis=1)
                .merge(
                    predicted_hpd_survival_test[1].rename(
                        columns={"survival_probability": "lower_survival_probability"}
                    ),
                    left_on=["index", "time_to_event_lower"],
                    right_on=["index", "time_int"],
                )
                .drop(["time_int", "time"], axis=1)
                .merge(
                    predicted_hpd_survival_test[2].rename(
                        columns={"survival_probability": "upper_survival_probability"}
                    ),
                    left_on=["index", "time_to_event_upper"],
                    right_on=["index", "time_int"],
                )
                .drop(["time_int", "time"], axis=1)
            ).drop("index", axis=1)

            logger.info("Writing test set predictions")
            predictions_merged.to_csv(experiment_path + "/predictions.csv", index=False)

        print(results)
        # Writing to sample.json
        logger.info("Writing model results")
        with open(experiment_path + "/model_results.yml", "w") as outfile:
            yaml.dump(results, outfile, default_flow_style=False)

        logger.info("Writing config file")
        with open(experiment_path + "/config.yml", "w") as outfile:
            yaml.dump(yaml_config, outfile)


if __name__ == "__main__":
    fire.Fire(train_survival_model)
