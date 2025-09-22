import optuna
import numpy as np
from util_training import *
from sac_torch import SACAgent
from redq_sac_torch import REDQSACAgent

def objective(trial) -> float:
    # torch.manual_seed(1)
    # np.random.seed(1)
    # Hyperparameter search space

    GAMMA = trial.suggest_float("GAMMA", 0.9, 0.99)
    TAU = trial.suggest_float("TAU", 0.01, 0.3)
    LR = trial.suggest_float("LR", 1E-5, 1E-3)
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [64, 128, 256])
    lam_s = 5
    lam_t = 10
    BUFFER_SIZE = 200000

    
    num_states = 2
    max_action = 0.26

    agent = SACAgent(
            state_dim=num_states,
            action_dim=1,
            max_action=max_action,
            gamma=GAMMA,
            tau=TAU,
            lr=LR,
            batch_size=BATCH_SIZE,
            buffer_size = BUFFER_SIZE,
            lam_s = lam_s,
            lam_t = lam_t
            )
    
    num_episodes = 20
    ep_length = 50
    reward_list = []

    for episode in range(num_episodes):
        train_sac_pitch(agent, "", ep_num=1, resolution=50, run_dir="", ep_length=ep_length, SAC=True)
        ep_reward, _, _ = evaluate_pitch(agent)
        reward_list.append(ep_reward.tolist())
        print(f"Episode: {episode + 1}, Reward: {ep_reward}")
    trial.set_user_attr("notes", str(reward_list))
    
    return np.mean(reward_list)

def redq_objective(trial) -> float:
    # torch.manual_seed(1)
    # np.random.seed(1)
    # Hyperparameter search space

    GAMMA = trial.suggest_float("GAMMA", 0.9, 0.99)
    TAU = trial.suggest_float("TAU", 0.01, 0.3)
    LR = trial.suggest_float("LR", 1E-5, 1E-3)
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [64, 128, 256])
    BUFFER_SIZE = trial.suggest_categorical("BUFFER_SIZE", [50000, 100000, 150000, 200000, 250000])
    UTD = trial.suggest_int("UTD", 2, 5)
    NR_CRITICS = trial.suggest_int("NR_CRITICS", 2, 5)
    DROPOUT = trial.suggest_float("DROPOUT", 1e-10, 1e-5, log=True)

    lam_s = 5
    lam_t = 15

    
    num_states = 2
    max_action = 0.26

    agent = REDQSACAgent(
            state_dim=num_states,
            action_dim=1,
            max_action=max_action,
            gamma=GAMMA,
            tau=TAU,
            lr=LR,
            batch_size=BATCH_SIZE,
            buffer_size =BUFFER_SIZE,
            utd= UTD,
            nr_critics=NR_CRITICS,
            dropout=DROPOUT,
            lam_s = lam_s,
            lam_t = lam_t
            )
    
    num_episodes = 20
    ep_length = 25
    reward_list = []

    for episode in range(num_episodes):
        train_pitch(agent, "", ep_num=1, resolution=50, run_dir="", ep_length=ep_length, SAC=False)
        ep_reward, _, _ = evaluate_pitch(agent)
        reward_list.append(ep_reward.tolist())
        print(f"Episode: {episode + 1}, Reward: {ep_reward}")
    trial.set_user_attr("notes", str(reward_list))

    return np.mean(reward_list)


if __name__ == "__main__":
    # Use SQLite database
    storage_url = "sqlite:///CAPS_STUDY.db"

    # Run optimization
    study = optuna.create_study(
        study_name="REDQ_HYPERPARAMS",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True
    )
    study.optimize(redq_objective, n_trials=20, n_jobs=1)

    # Print best trial with extra metric optuna-dashboard sqlite:///CAPS_STUDY.db
    #best_trial = study.best_trial
    # print(f"Best trial params: {best_trial.params}")
    # print(f"Best objective value: {best_trial.value}")
    # print(f"Extra metric in best trial: {best_trial.user_attrs.get('extra_metric')}")
