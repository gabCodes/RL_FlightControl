from dataclasses import dataclass
import yaml
from typing import List, Union, Dict


@dataclass
class GlobalConfig:
    layer_neurons: int
    max_action: float
    min_action: float
    gamma: float
    tau: float
    lr: float
    batch_size: int
    buffer_size: int
    lam_s: List[int]
    lam_t: List[int]
    dropout: int
    Q_Nr: List[int]
    UTD: List[int]
    trim_inputs: List[float]
    nr_runs: int
    ep_num: List[int]
    resolution: List[int]


@dataclass
class TaskConfig:
    state_dim: int
    action_dim: int
    angle_idx: Union[int, List[int]]
    rate_idx: Union[int, List[int]]
    reward_weight: Union[float, List[float]]
    offset: Union[float, List[float]]

@dataclass
class FaultConfig:
    ep_length: int
    start_maxd: int | None = None
    end_maxd: int | None = None
    start_mind: int | None = None
    end_mind: int | None = None

@dataclass
class Config:
    globals: GlobalConfig
    tasks: Dict[str, TaskConfig]
    faults: Dict[str, FaultConfig]

# Load YAML config file into a Config dataclass
def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Extract global hyperparameters (everything that is not 'tasks' or 'faults')
    globals_keys = [k for k in raw if k not in ('tasks', 'faults')]
    globals_dict = {k: raw[k] for k in globals_keys}
    globals_config = GlobalConfig(**globals_dict)

    tasks = {name: TaskConfig(**params) for name, params in raw['tasks'].items()}
    faults = {name: FaultConfig(**params) for name, params in raw['faults'].items()}

    return Config(globals=globals_config, tasks=tasks, faults=faults)


if __name__ == "__main__":
    config = load_config("config.yml")
    print(config.tasks['pitch'].state_dim)    # 2
    print(config.tasks['pitchroll'].angle_idx) # [7,6]
    print(config.faults['jolt'].ep_length)    # 20
    print(config.globals.gamma)
    print(config.globals.dropout)