import gym
from gym.envs.registration import register

from .frozen_lake import *
from .taxi import *

__all__ = ['FrozenLakeEnv', 'TaxiEnv']

register(
    id='FrozenLake8x8-v1',
    entry_point='environments:FrozenLakeEnv',
    kwargs={'map_name': '8x8'}
)
register(
    id='FrozenLake20x20-v1',
    entry_point='environments:FrozenLakeEnv',
    kwargs={'map_name': '20x20'}
)
register(
    id='TaxiEnv-v1',
    entry_point='environments:TaxiEnv'
)


def get_frozen_lake_environment():
    return 'FrozenLake-v0'


def get_small_frozen_lake_environment():
    return 'FrozenLake8x8-v1'


def get_large_frozen_lake_environment():
    return 'FrozenLake20x20-v1'


def get_taxi_environment():
    return 'TaxiEnv-v1'
