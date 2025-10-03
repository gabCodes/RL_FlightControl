from config import load_config
from .helpers import train_agents, fault_cases
from .HPO import HPO


if __name__ == "__main__":
    HPO()

    config = load_config("config.yml")
    
    train_agents(config)

    fault_cases(config)