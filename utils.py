"""
    Utility functions
"""
import requests

def connected_to_internet(url='http://www.google.com/', timeout=5):
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
