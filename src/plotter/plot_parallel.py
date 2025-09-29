import optuna
import plotly.express as px


# Function for creating the hyperparameter plots, takes in the hyperparameter storage URL and the study name
def plot_parallel(storage_url: str, study_name: str) -> None:

    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    for trial in study.trials:
        print(f"Trial {trial.number}: Value={trial.value}")

    # Convert the study trials to a dataFrame
    df = study.trials_dataframe(attrs=("number", "params", "value", "state"))

    print(df.columns)


    # Filter out only the completed trials
    df = df[df['state'] == 'COMPLETE']
    df = df[df["number"] != 95]

    # Drop unnecessary columns
    if "REDQ" in study_name:
        df = df[['number', 'params_BATCH_SIZE', 'params_BUFFER_SIZE', 'params_DROPOUT', 'params_GAMMA', 'params_LR', 'params_NR_CRITICS', 'params_TAU', 'params_UTD', 'value']]
        dimensions=['params_BATCH_SIZE', 'params_BUFFER_SIZE', 'params_DROPOUT', 'params_GAMMA', 'params_LR', 'params_NR_CRITICS', 'params_TAU', 'params_UTD']
    
    else:
        df = df[['number', 'params_BATCH_SIZE', 'params_GAMMA', 'params_LR', 'params_TAU', 'value']]
        dimensions=['params_BATCH_SIZE', 'params_GAMMA', 'params_LR', 'params_TAU'] 

    fig = px.parallel_coordinates(
    df,
    dimensions=dimensions,
    color="value", 
    color_continuous_scale=px.colors.diverging.RdBu
    )

    fig.update_layout(
        title="",
        coloraxis_colorbar=dict(title="Objective Value")
    )
    fig.show()

    return