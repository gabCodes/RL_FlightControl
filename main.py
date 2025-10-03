from config import load_config
from .helpers import train_agents, fault_cases


if __name__ == "__main__":
    config = load_config("config.yml")

    train_agents(config)

    fault_cases(config)