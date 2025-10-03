import optuna
import numpy as np
from src.operations.util import _agentChooser
from config import load_config
from src.operations import train, evaluate

def objective(trial) -> float:
    config = load_config("config.yml")
    GAMMA = trial.suggest_float("GAMMA", 0.9, 0.99)
    TAU = trial.suggest_float("TAU", 0.01, 0.3)
    LR = trial.suggest_float("LR", 1E-5, 1E-3)
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [64, 128, 256])

    config.globals['gamma'] = GAMMA
    config.globals['tau'] = TAU
    config.globals['lr'] = LR
    config.globals['batch_size'] = BATCH_SIZE
    config.phases['train'].ep_length = 50
    config.phases['train'].nr_runs = 1
    config.phases['train'].ep_num[1] = 1

    agent = _agentChooser('SAC', 'pitch', config)
    reward_list = []

    for episode in range(20):
        agent = train(agent, 'SAC', 'pitch', config, save = False)
        ep_reward = evaluate(agent, 'pitch', None, config)
        reward_list.append(ep_reward.tolist())
        print(f"Episode: {episode + 1}, Reward: {ep_reward}")
    trial.set_user_attr("notes", str(reward_list))
    
    return np.mean(reward_list)

def redq_objective(trial) -> float:
    config = load_config("config.yml")

    GAMMA = trial.suggest_float("GAMMA", 0.9, 0.99)
    TAU = trial.suggest_float("TAU", 0.01, 0.3)
    LR = trial.suggest_float("LR", 1E-5, 1E-3)
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [64, 128, 256])
    BUFFER_SIZE = trial.suggest_categorical("BUFFER_SIZE", [50000, 100000, 150000, 200000, 250000])
    UTD = trial.suggest_int("UTD", 2, 5)
    NR_CRITICS = trial.suggest_int("NR_CRITICS", 2, 5)
    DROPOUT = trial.suggest_float("DROPOUT", 1e-10, 1e-5, log=True)

    config.globals['gamma'] = GAMMA
    config.globals['tau'] = TAU
    config.globals['lr'] = LR
    config.globals['batch_size'] = BATCH_SIZE
    config.globals['buffer_size'] = BUFFER_SIZE
    config.globals['dropout'] = DROPOUT
    config.agents['RED3Q'].utd = UTD
    config.agents['RED3Q'].q_nr = NR_CRITICS
    config.phases['train'].ep_length = 50
    config.phases['train'].nr_runs = 1
    config.phases['train'].ep_num[1] = 1

    agent = _agentChooser('RED3Q', 'pitch', config)
    reward_list = []

    for episode in range(20):
        agent = train(agent, 'RED3Q', 'pitch', config, save = False)
        ep_reward = evaluate(agent, 'pitch', None, config)
        reward_list.append(ep_reward.tolist())
        print(f"Episode: {episode + 1}, Reward: {ep_reward}")
    trial.set_user_attr("notes", str(reward_list))

    return np.mean(reward_list)


def HPO():
    # Use SQLite database
    print("Conducting hyperparameter optimisation")
    storage_url = "sqlite:///STUDY.db"

    # Run optimization
    sac_study = optuna.create_study(
        study_name="SAC_HYPERPARAMS",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True
    )
    sac_study.optimize(objective, n_trials=30, n_jobs=1)

    # Print best trial with extra metric optuna-dashboard sqlite:///STUDY.db
    best_trial = sac_study.best_trial
    print(f"Best trial params: {best_trial.params}")
    print(f"Best objective value: {best_trial.value}")
    print(f"Extra metric in best trial: {best_trial.user_attrs.get('extra_metric')}")

    redq_study = optuna.create_study(
        study_name="REDQ_HYPERPARAMS",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True
    )
    redq_study.optimize(redq_objective, n_trials=30, n_jobs=1)

    # Print best trial with extra metric optuna-dashboard sqlite:///STUDY.db
    best_trial = redq_study.best_trial
    print(f"Best trial params: {best_trial.params}")
    print(f"Best objective value: {best_trial.value}")
    print(f"Extra metric in best trial: {best_trial.user_attrs.get('extra_metric')}")
