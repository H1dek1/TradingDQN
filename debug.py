#!/usr/bin/env python3

import gym
import simple_trading

def main():
    action_list = [0, 0, 1, 0, 0, 0, 2]
    env = gym.make('SimpleTrading-v0')
    #env.debug()
    #env.random_play()
    obs = env.reset()
    print('+++++++ Reset Return +++++++')
    print('series')
    print(obs[0])
    print('latent_gain')
    print(obs[1])

    for action in action_list:
        print('+++++++ Step Return : action = {} +++++++'.format(action))
        obs, reward, done, info = env.step(action)
        print('series')
        print(obs[0])
        print('latent_gain')
        print(obs[1])
        print('reward')
        print(reward)
        print('done')
        print(done)
        print('info')
        print(info)

if __name__ == '__main__':
    main()
