# Bayesian Piecewise Survival model

This folder provides a tool for training Bayesian survival piecewise exponential models.

![img/architecture.png"](https://github.com/Playtika/bpsm/blob/develop/img/architecture.png)

This script performs the following steps:

* Read input configuration file
* Load data
* Preprocess data
* Apply feature selection
* Fit model
* Extract performance and posterior estimates plots
* Make predictions for train and validation sets
* Make predictions for test set if selected
* Save model and results in the output directory

## Configuration file

The configuration file should respect the following structure.
Examples of config files are also at ```lausanne_player_360/scripts/survival/docs/dummy_experiment/config```.

* ```config.yaml```
    - ```train_dates```: date range of train set to split the data before preprocessing
    - ```validation_dates```: date range of validation set to split the data before preprocessing
    - ```test_dates```: date range of test set to split the data before preprocessing
    - ```data_sample_frac```: float between 0 and 1 to perform the modelling and analysis on a sample of the data.
    - ```data_input_path```: path of the input dataset, e.g. "path/data.parquet"
    - ```model_type```: can be either regularised or simple. If regularised then it uses Horseshoe prior on the feature coefficients. If simple then independent gamma priors are placed on features' precisions.
    - ```use_test_set```: boolean true or false. If true then performance metrics are calculated for test set and predictions are extracted.
    - ```time_to_event_cutoff```: positive integer that cuts off time censoring
    - ```id_column```: name indicating id column, e.g. user_id
    - ```date_column```: name indicating date column, e.g date
    - ```time_to_event_column```: name indicating time to event column, e.g. time_to_event
    - ```event_column```: name to give to event column, e.g churn. This column is constructed during data preprocessing
    - ```drop_columns```: columns to drop, e.g. ["age", "gender"]
    - ```features_select```: features to be selected for modelling
    - ```feature_selection_args```: dictionary of arguments for feature selection.
      * ```method``` : It can be `stepwise`, `sparse_correlation` or `correlation`
      * ```correlation_method```: If sparse_correlation or correlation method is used select between `pearson` and `spearman` correlations.
      * ```corr_cutoff```: defined between [0,1], if either sparse_correlation or correlation methods are used for feature selection. It will remove one of pairs of features with a correlation greater than this value.
      * ```p_val_thresh```: p-value to consider a feature significant or correlation significant
      * ```initial_list```: features to be used initially in stepwise selection if stepwise method is used.
      * ```threshold_in```: defined between [0,1], in stepwise selection for a feature to be considered
            significant to enter. It is used only if method is stepwise.
      * ```threshold_out```: defined between [0,1], in stepwise selection for a feature to be considered
            not significant to leave. It is used only if method is stepwise. lways set threshold_in < threshold_out to avoid infinite looping.
      * ```n_vars_drop```: positive integer indicating number of features to be dropped
            if `correlation` method is used.
    - ```categorical_features```: categorical features to be separated during data scaling
    - ```hyper_param_lambda0```: dictionary indicating experiment name and shape, rate parameters for baseline hazard, e.g. exp1: [0.1, 0,1]
    - ```n_its```: positive integer number of variational inference iterations. It is recommended to between 20K and 100K
    - ```n_samples```: positive integer for sampling from the posterior distribution. It is recommended to be between 2K and 5K.
    - ```continuous_partial_effects```: dictionary of numerical features with a list of values to plot partial effects along with the baselines, e.g. level_end: [ 20, 50, 200 ]
    - ```categorical_partial_effects```: list of categorical features to plot partial effects along with the baselines, e.g. ["is_active", "has_subscription"]
    - ```n_users_trajectories```: positive integer in order to plot random users' estimate survival trajectories.
    - ```hpd_alpha```: float between 0 and 1 for HPD alpha to calculate the high density posterior area as [100 * alpha/2, 100 * (1-alpha/2)], e.g. 0.05
    - ```output_path```: path of the output results dataset, e.g. "data/"

## Usage

Run the following command line to execute the tool. Be careful to choose the right config file according to the specific task (either "regularised", "simple").

```commandline
$ python3 train_survival_model.py [--config_file CONFIG_FILE]

optional arguments:
    --config_file CONFIG_FILE
                        configuration path of the specific process to run
```

e.g.
```commandline
$ python3 train_survival_model.py --config_file path/config.yaml
```


## Code structure

Source code
```
├── docs                                                  <- Documentation.
│   ├── dummy_experiment                                  <- Dummy experiment directory.
│   │   │   └── config.yaml                               <- Dummy experiment configuration.
│   ├── prior_sensitivity_experiments                     <- Real experiment configuration.
│   │   │   └── config.yaml                               <- Dummy experiment configuration.
├── src                                                   <- Directory of the source code.
│   ├── BayesianHorseshoeSurvival.py                      <- Bayesian Survival Horseshoe code.
│   ├── BayesianSurvival.py                               <- Bayesian Survival superclass code.
│   ├── BayesianSurvivalSimple.py                         <- Bayesian Survival unregularised code.
│   ├── DataPreprocessor.py                               <- Data preprossecing code.
│   ├── Evaluation.py                                     <- Evalutation plots and metrics code.
│   ├── FeatureSelector.py                                <- Feature selection code.
│   ├── interval_constructors.py                          <- Interval constructors needed for modelling code.
│   ├── plots_utils.py                                    <- Shared plot utils.
│   ├── PlotBayesianSurvival.py                           <- Code accessing arviz and pymc plots.
│   ├── predictive_functions.py                           <- Predictive functions for BayesianSurvival.
│   └── utils.py                                          <- Shared Python utils.
├── main.py                                               <- Python script to execute.
└── README.md                                             <- Info bayesian piecewise survival model and how to use it.
```

Experiment results
```
└── dummy_experiment                                      <- Directory for the experiment results.
    │── experiment_1                                      <- Results plot for the first experiment.
    │   │── model                                         <- Directory for storing model.
    │   │   ├── model.pkl                                 <- Pickle file of model.
    │   │── plots                                         <- Directory of results' plots for the experiment.
    │       ├── baseline_hazard.png                       <- Baseline hazard posteriors.
    │       ├── baseline_survival.png                     <- Baseline survival posteriors.
    │       └── ...                                       <- ...
    │   ├── config.yaml                                   <- Configuration file.
    │   ├── model_results.yaml                            <- Model's summarized results.
    │   └── predictions.csv                               <- Dataset with predictions.
    │── experiment_2                                      <- Results plot for the second experiment.
    │   │── model                                         <- Directory for storing model.
    │   │   ├── model.pkl                                 <- Pickle file of model.
    │   │── plots                                         <- Directory of results' plots for the experiment.
    │       ├── baseline_hazard.png                       <- Baseline hazard posteriors.
    │       ├── baseline_survival.png                     <- Baseline survival posteriors.
    │       └── ...                                       <- ...
    │   ├── config.yaml                                   <- Configuration file.
    │   ├── model_results.yaml                            <- Model's summarized results.
    │   └── predictions.csv                               <- Dataset with predictions.
```
