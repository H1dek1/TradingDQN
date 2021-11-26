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
    n_share_max = 2
    env = gym.make('SimpleTrading-v0', window_length=window_length, max_share=n_share_max, reward_gain=10.0, latent_gain=10.0)
    # env.random_play()
    # sys.exit()

    model = DQN(
            env=env, 
            use_target_network=True,
            use_doubleDQN=False, 
            use_dueling=True,
            initial_eps=1.0,
            target_update_interval=30000,
            final_eps=0.02,
            eps_change_length=3000,
            learning_rate=0.003,
            gamma=0.99,
            )

    eval_env = gym.make(
            'SimpleTrading-v0',
            window_length=window_length,
            max_share=n_share_max,
            reward_gain=10.0,
            latent_gain=10.0
            )

    history = model.learn(400000, eval_env=eval_env, eval_interval=300)
    model.save('models/sample_model')
    
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
