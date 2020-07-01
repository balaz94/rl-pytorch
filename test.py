from utils.breakout_wrapper import make_env
import gym

env = make_env('BreakoutNoFrameskip-v4')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
