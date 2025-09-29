from handlers import PitchHandler, RollHandler, PitchRollHandler
from faults import QuarterEfficiencyFault, JoltFault
from agents import SACAgent, REDQSACAgent
from config import Config

# Chooses the correct handler per agent per task per fault
def _choose_handler(agent: SACAgent | REDQSACAgent, task: str, ep_length: int, fault = None):
    handler = None

    mapping = {
        "pitch": PitchHandler,
        "roll": RollHandler,
        "pitchroll": PitchRollHandler
    }

    handler = mapping[task.lower()](agent, ep_length)

    if fault:
        mapping = {
            "eff": QuarterEfficiencyFault,
            "jolt": JoltFault
        }
        handler = mapping[fault.lower()](handler)

    return handler

def _agentChooser(agent: str, task: str, config: Config):
    if "RED" in agent:
        a = REDQSACAgent(agent, task.lower(), config)

    else:
        a = SACAgent(task, config)

    return a

# Loads the evaluation phase specific parameters for the evaluation function to use
def _evalLoader(config: Config) -> tuple[int, list[float]]:

    short_eps = config.phases['eval'].ep_num[0]
    long_eps = config.phases['eval'].ep_num[1]
    short_res = config.phases['eval'].resolution[0]
    long_res = config.phases['eval'].resolution[1]
    ep_length = config.phases['eval'].ep_length

    return short_eps, long_eps, short_res, long_res, ep_length






