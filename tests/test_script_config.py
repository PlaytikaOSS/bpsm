from bpsm.script_config import ScriptConfig


def test_script_config_instantiation(tmp_path):
    input_path = tmp_path / "input.csv"
    input_path.write_text("test")

    train_dates = ["2020-01-01", "2020-01-02"]

    sc = ScriptConfig(
        train_dates=train_dates,
        validation_dates=["2020-01-03"],
        test_dates=["2020-01-04"],
        data_sample_frac=0.8,
        data_input_path=input_path,
        model_type="regularised",
        time_to_event_cutoff=5,
        id_column="id",
        date_column="date",
        time_to_event_column="time_to_event",
        event_column="event",
        drop_columns=["drop1", "drop2"],
        categorical_features=["cat1", "cat2"],
        hyper_param_lambda0={"lambda0": 0.1},
        n_its=10,
        n_samples=100,
        continuous_partial_effects={"cont1": 0.1},
        categorical_partial_effects=["cat1"],
        n_users_trajectories=10,
        hpd_alpha=0.05,
        output_path="output",
    )
    assert sc.train_dates == train_dates
