#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from rl.dqn import DQN
import simple_trading

def main():
    window_length = 12
    env = gym.make('SimpleTrading-v0', window_length=window_length)
    #env.random_play()

    model = DQN(
            env=env, 
            window_size=window_length,
            use_target_network=False,
            use_doubleDQN=False, 
            use_dueling=True,
            initial_eps=0.8,
            update_interval=4,
            target_update_interval=20000,
            replay_buffer_size=1000000,
            final_eps=0.05,
            learning_starts=1000,
            replay_batch_size=4,
            learning_rate=0.01,

            #learning_starts=1,
            #update_interval=1,
            #target_update_interval=1,
            #replay_batch_size=4
            )
    #model.test_episode()
    #sys.exit()

    history = model.learn(200000)
    #history = model.learn(2000)
    #model.save('models/sample_model')
    
    eval_env = gym.make('SimpleTrading-v0', window_length=window_length)
    model.test_episode(eval_env)
    df_hist = pd.DataFrame.from_dict(history, orient='index').T                                                                                     
    df_hist.to_csv('history.csv')
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True)
    ax[0].set_xlabel('total_step', fontsize=15)
    ax[0].set_ylabel('epi_rew', fontsize=15)
    ax[0].plot(df_hist['total_step'], df_hist['epi_rew'])
    ax[1].set_xlabel('total_step', fontsize=15)
    ax[1].set_ylabel('ave_loss', fontsize=15)
    ax[1].set_yscale('log')
    ax[1].plot(df_hist['total_step'], df_hist['ave_loss'])
    fig.savefig('result.png')



if __name__ == '__main__':
    main()
