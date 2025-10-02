from dataclasses import dataclass
import yaml
from typing import List, Union, Dict

@dataclass
class GlobalConfig:
    hidden_dim: int
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
    trim_inputs: List[float]


@dataclass
class AgentConfig:
    utd: int | None = None
    q_nr: int | None = None

@dataclass
class PhaseConfig:
    ep_length: int
    nr_runs: int | None = None
    ep_num: List[int] | None = None
    resolution: List[int] | None = None
    save_dir: str | None = None

@dataclass
class TaskConfig:
    state_dim: int
    action_dim: int
    angle_idx: Union[int, List[int]]
    rate_idx: Union[int, List[int]]
    reward_weight: Union[float, List[float]]
    offset: Union[float, List[float]]
    lam_s: int
    lam_t: int

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
    agents: Dict[str, AgentConfig]
    phases: Dict[str, PhaseConfig]
    tasks: Dict[str, TaskConfig]
    faults: Dict[str, FaultConfig]

# Load YAML config file into a Config dataclass
def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Extract global hyperparameters (everything that is not 'tasks' or 'faults')
    globals_keys = [k for k in raw if k not in ('agents', 'phases', 'tasks', 'faults')]
    globals_dict = {k: raw[k] for k in globals_keys}
    globals_config = GlobalConfig(**globals_dict)

    agents = {name: AgentConfig(**params) for name, params in raw['agents'].items()}
    phases = {name: PhaseConfig(**params) for name, params in raw['phases'].items()}
    tasks = {name: TaskConfig(**params) for name, params in raw['tasks'].items()}
    faults = {name: FaultConfig(**params) for name, params in raw['faults'].items()}

    return Config(globals=globals_config, agents=agents, phases=phases, tasks=tasks, faults=faults)


if __name__ == "__main__":
    config = load_config("config.yml")
    print(config.tasks['pitch'].state_dim)    # 2
    print(config.tasks['pitchroll'].angle_idx) # [7,6]
    print(config.faults['jolt'].ep_length)    # 20
    print(config.globals.gamma)
    print(config.globals.dropout)
    print(config.agents['RED3Q'].utd)
    print(config.agents['RED5Q'].q_nr)
    print(config.phases['train'].ep_num)
    print(config.phases['train'].resolution)
    print(config.phases['eval'].ep_num)
    print(config.phases['eval'].ep_length)
    print(config.phases['run'].resolution)
    print(config.phases['run'].save_dir)
    print(config.phases['train'].save_dir)