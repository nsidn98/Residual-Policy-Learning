"""
    Utility functions
"""
import gym
import controllers.slideEnv as slideEnv
import controllers.pushEnv as pushEnv
import controllers.pickAndPlaceEnv as pickPlaceEnv
import requests

def connected_to_internet(url:str='http://www.google.com/', timeout:int=5):
    """
        Check if system is connected to the internet
        Used when running code on MIT Supercloud
    """
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
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
    # try:
    #     # check if environment exists in gym
    #     return gym.make(env_name)
    # except AttributeError:
    #     # else import from custom environment file
    #     return getattr(slideEnv, env_name)()
    # except (gym.error.Error, AttributeError):
    #     return getattr(pushEnv, env_name)()
    # except (gym.error.Error, AttributeError):
    #     return getattr(pickPlaceEnv, env_name)()
    # except Exception as e:
        
    # except:
    #     return getattr(pickPlaceEnv, env_name)()

if __name__ == "__main__":
    env_names = ['FetchSlide-v1','FetchSlide','FetchSlideImperfectControl']
    for name in env_names:
        env = make_env(name)
        print(f"{name}: {env.reset()}")