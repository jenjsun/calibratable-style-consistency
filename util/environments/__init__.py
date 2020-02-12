from .core import BaseEnvironment, generate_rollout
from .bball import BBallEnv
from .mouse_v1 import MouseV1Env

environment_dict = {
    'bball' : BBallEnv(),
    'mouse_v1' : MouseV1Env()
}


def load_environment(env_name):
	if env_name in environment_dict:
		return environment_dict[env_name]
	else:
		raise NotImplementedError
