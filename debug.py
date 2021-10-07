#!/usr/bin/env python3

import gym
import simple_trading

def main():
    env = gym.make('SimpleTrading-v0')
    #env.debug()
    env.random_play()
    print('Success!')


if __name__ == '__main__':
    main()
