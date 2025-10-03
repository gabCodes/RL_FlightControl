from src.operations import runs, fault_test
from src.plotter import plot_learningcurve, plot_parallel
from src.plotter.plot_box import plot_box_singlets, plot_box_pairs
from config import Config

"""
Training the agents, it's important to initially set training to true as this will generate the weights
and performance logs to create the learning curves. After they have been trained, can set training to False
to just evaluate and generate graphs.
"""
def train_agents(config: Config):
    runs('SAC', 'pitch', config, training=True)
    runs('SAC', 'roll', config, training=True)
    runs('SAC', 'pitchroll', config, training=True)

    runs('RED3Q', 'pitch', config, training=True)
    runs('RED3Q', 'roll', config, training=True)
    runs('RED3Q', 'pitchroll', config, training=True)

    runs('RED5Q', 'pitch', config, training=True)
    runs('RED5Q', 'roll', config, training=True)
    runs('RED5Q', 'pitchroll', config, training=True)


"""
Evaluating the agents per fault case, outputs of these are needed to generate the boxplots.
"""
def fault_cases(config: Config):
    # PITCH - NOM
    err_nom_pitch1 = fault_test('SAC', 'pitch', 'nominal', config)
    err_nom_pitch2 = fault_test('RED3Q', 'pitch', 'nominal', config)
    err_nom_pitch3 = fault_test('RED5Q', 'pitch', 'nominal', config)

    # PITCH - EFF
    err_eff_pitch1 = fault_test('SAC', 'pitch', 'quarter', config)
    err_eff_pitch2 = fault_test('RED3Q', 'RED3Q', 'quarter', config)
    err_eff_pitch3 = fault_test('RED5Q', 'RED5Q', 'quarter', config)

    # PITCH - JOLT
    err_jolt_pitch1 = fault_test('SAC', 'pitch', 'jolt', config)
    err_jolt_pitch2 = fault_test('RED3Q', 'pitch', 'jolt', config)
    err_jolt_pitch3 = fault_test('RED5Q', 'pitch', 'jolt', config)

    # ROLL - NOM
    err_nom_roll1 = fault_test('SAC', 'roll', 'nominal', config)
    err_nom_roll2 = fault_test('RED3Q', 'roll', 'nominal', config)
    err_nom_roll3 = fault_test('RED5Q', 'roll', 'nominal', config)

    # ROLL - EFF
    err_eff_roll1 = fault_test('SAC', 'roll', 'quarter', config)
    err_eff_roll2 = fault_test('RED3Q', 'roll', 'quarter', config)
    err_eff_roll3 = fault_test('RED5Q', 'roll', 'quarter', config)

    # ROLL - JOLT
    err_jolt_roll1 = fault_test('SAC', 'roll', 'jolt', config)
    err_jolt_roll2 = fault_test('RED3Q', 'roll', 'jolt', config)
    err_jolt_roll3 = fault_test('RED5Q', 'roll', 'jolt', config)

    groups = {
        'box_pitch_nominal': [err_nom_pitch1, err_nom_pitch2, err_nom_pitch3],
        'box_pitch_quarter': [err_eff_pitch1, err_eff_pitch2, err_eff_pitch3],
        'box_pitch_jolt': [err_jolt_pitch1, err_jolt_pitch2, err_jolt_pitch3],
        'box_roll_nominal': [err_nom_roll1, err_nom_roll2, err_nom_roll3],
        'box_roll_quarter': [err_eff_roll1, err_eff_roll2, err_eff_roll3],
        'box_roll_jolt': [err_jolt_roll1, err_jolt_roll2, err_jolt_roll3],
    }

    # Generate boxplots for pitch and roll
    plot_box_singlets(groups)

    #Pitchroll nom
    err_nom_pitchroll1 = fault_test('SAC', 'pitchroll', 'nominal', config)
    err_nom_pitchroll2 = fault_test('RED3Q', 'pitchroll', 'nominal', config)
    err_nom_pitchroll3 = fault_test('RED5Q', 'pitchroll', 'nominal', config)

    #Pitchroll eff
    err_eff_pitchroll1 = fault_test('SAC', 'pitchroll', 'quarter', config)
    err_eff_pitchroll2 = fault_test('RED3Q', 'pitchroll', 'quarter', config)
    err_eff_pitchroll3 = fault_test('RED5Q', 'pitchroll', 'quarter', config)

    # Pitchroll jolt
    err_jolt_pitchroll1 = fault_test('SAC', 'pitchroll', 'jolt', config)
    err_jolt_pitchroll2 = fault_test('RED3Q', 'pitchroll', 'jolt', config)
    err_jolt_pitchroll3 = fault_test('RED5Q', 'pitchroll', 'jolt', config)

    #Pitchroll
    nom_data = [err_nom_pitchroll1, err_nom_pitchroll2, err_nom_pitchroll3]
    eff_data = [err_eff_pitchroll1, err_eff_pitchroll2, err_eff_pitchroll3]
    jolt_data = [err_jolt_pitchroll1, err_jolt_pitchroll2, err_jolt_pitchroll3]

    # Generate boxplots for pitchroll
    plot_box_pairs(nom_data, eff_data, jolt_data)


if __name__ == "__main__":

    # Pitch short scope learning curves
    files = ['SAC30v2PITCH50.txt', 'RED3Q30v3PITCH50.txt', 'RED5Q30v3PITCH50.txt']

    # Pitch long scope learning curves
    # files = ['SAC30v2PITCH250.txt', 'RED3Q30v2PITCH250.txt', 'RED5Q30v2PITCH250.txt']

    # Roll short scope learning curves
    # files = ['SAC30v2ROLL50.txt', 'RED3Q30v2ROLL50.txt', 'RED5Q30v2ROLL50.txt']

    # Roll long scope learning curves
    # files = ['SAC30v2ROLL250.txt', 'RED3Q30v2ROLL250.txt', 'RED5Q30v2ROLL250.txt']

    # Bi-axial short scope learning curves
    # files = ['SAC30v2PITCHROLL50.txt', 'RED3Q30v2PITCHROLL50.txt', 'RED5Q30v2PITCHROLL50.txt']

    # Bi-axial long scope learning curves
    # files = ['SAC30v2PITCHROLL250.txt', 'RED3Q30v2PITCHROLL250.txt', 'RED5Q30v2PITCHROLL250.txt']

    plot_learningcurve(files)

    #storage_url = "sqlite:///CAPS_STUDY.db"
    #study_name = "SAC_30_RUNS"
    #study_name = "REDQ_HYPERPARAMS"

    # create_parallel(storage_url, study_name)
