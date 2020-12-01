"""
    Utility functions
"""
import gym
import controllers.slideEnv as slideEnv
import controllers.pushEnv as pushEnv
import controllers.pickAndPlaceEnv as pickPlaceEnv
import requests
from mpi4py import MPI

def connected_to_internet(url:str='http://www.google.com/', timeout:int=5):
    """
        Check if system is connected to the internet
        Used when running code on MIT Supercloud
    """
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("No internet connection available.")
    return False

def make_env(env_name:str):
    """
        Make an environment and return
        This will check if the environment exists 
        within gym or our custom environment files
    """
    try:
        # check if environment exists in gym
        return gym.make(env_name)
    except:
        pass
    try:
        # else import from custom slide environment file
       return getattr(slideEnv, env_name)()
    except:
        pass
    try:
       return getattr(pushEnv, env_name)()
    except:
        pass
    try:
       return getattr(pickPlaceEnv, env_name)()
    except:
        pass 

def get_pretty_env_name(env_name:str):
    if 'FetchPickAndPlace' in env_name:
        new_env_name = 'pickPlace'
        exp_name = env_name
        exp_name = exp_name.replace('FetchPickAndPlace','')
        return f"{new_env_name}{exp_name}"
    if 'FetchSlide' in env_name:
        new_env_name = 'slide'
        exp_name = env_name
        exp_name = exp_name.replace('FetchSlide','')
        exp_name = exp_name.replace('Control','')
        return f"{new_env_name}{exp_name}"
    if 'FetchPush' in env_name:
        new_env_name = 'push'
        exp_name = env_name
        exp_name = exp_name.replace('FetchPush','')
        return f"{new_env_name}{exp_name}"

if __name__ == "__main__":
    env_names = ['FetchSlide-v1','FetchSlide','FetchSlideImperfectControl']
    for name in env_names:
        env = make_env(name)
        print(f"{name}: {env.reset()}")