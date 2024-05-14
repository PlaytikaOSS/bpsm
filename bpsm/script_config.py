from typing import Dict, List, Optional

from pydantic import BaseModel, FilePath, confloat, conint
from typing_extensions import Literal

MODEL_TYPE = Literal["regularised", "simple"]


class ScriptConfig(BaseModel):
    train_dates: List[str]
    validation_dates: List[str]
    test_dates: List[str]
    data_sample_frac: confloat(ge=0, le=1)
    data_input_path: FilePath
    model_type: MODEL_TYPE
    use_test_set: bool = True
    time_to_event_cutoff: conint(ge=1)
    id_column: str
    date_column: str
    time_to_event_column: str
    event_column: str
    drop_columns: List[str]
    features_select: Optional[List[str]]
    apply_feature_selection: Optional[bool] = False
    feature_selection_args: Optional[Dict]
    categorical_features: List[str]
    hyper_param_lambda0: Dict
    n_its: conint(gt=0)
    n_samples: conint(gt=0)
    continuous_partial_effects: Dict
    categorical_partial_effects: List[str]
    n_users_trajectories: conint(gt=1)
    hpd_alpha: confloat(ge=0, le=1)
    output_path: str
