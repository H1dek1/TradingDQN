from collections import deque
import copy
import sys
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import json

import keras
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Dense, LSTM, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import model_from_config
     
class DQN:
    """
    Simple Deep Q-Network for time series Reinforcement Learning
    :param env: (Gym.Env) The environment constructed with Gym format.
    :param window_size: (int) The size of time series data input to this model.
    :param replay_buffer_size: (int) The max size of replay buffer.
    :param replay_batch_size: (int) size of a batched sample from replay buffer for training.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning.
    :param learning_rate: (float) learning rate for adam optimizer.
    :param gamma: (float) discount factor
    :param initial_eps: (float) initial value of random action probability.
    :param final_eps: (float) final value of random action probability.
    :param eps_change_length: (int) The number of episodes that is taken to change epsilon.
    """
    
    def __init__(self, env, replay_buffer_size=1000000, replay_batch_size=32, n_replay_epoch=1, learning_starts=10000, learning_rate=0.0001, gamma=0.99, initial_eps=1.0, final_eps=0.05, eps_change_length=1000, load_Qfunc=False, Qfunc_path=None, use_target_network=False, use_doubleDQN=False, update_interval=4, target_update_interval=20000, use_dueling=False, load_model=False, load_model_path=None):
        """
        Environment
        """
        self._env = env
        self._n_action = self._env.action_space.n
        
        """
        Experience Replay
        """
        self._replay_buffer_size = replay_buffer_size
        self._replay_batch_size = replay_batch_size
        self._replay_buffer = deque(maxlen=self._replay_buffer_size)
        self._n_replay_epoch = n_replay_epoch

        """
        Learning
        """
        self._use_dueling = use_dueling
        self._use_target_network = use_target_network
        self._use_doubleDQN = use_doubleDQN
        self._learning_starts = learning_starts
        self._learning_rate = learning_rate
        self._gamma = gamma
        # For epsilon greedy
        self._initial_eps = initial_eps
        self._final_eps = final_eps
        self._eps_change_length = eps_change_length
        self._update_interval = update_interval
        self._target_update_interval = target_update_interval

        """
        Q-Function
        """
        if not load_Qfunc:
            self._Qfunc = self._init_Qfunc(self._use_dueling)
            self._Qfunc.compile(optimizer=Adam(learning_rate=self._learning_rate), loss='mean_squared_error')
        else:
            self._Qfunc = None

        if self._use_target_network or self._use_doubleDQN:
            self._target_Qfunc = self._init_Qfunc(self._use_dueling)
            self._target_Qfunc.compile(optimizer=Adam(learning_rate=self._learning_rate), loss='mean_squared_error')

        plot_model(self._Qfunc, show_shapes=True, show_layer_names=True)
        print(self._Qfunc.summary())

        #if self._use_doubleDQN:
        #    self._target_Qfunc = self._clone_network(self._Qfunc)

    def _init_Qfunc(self, use_dueling=False):
        if use_dueling:
            if len(self._env.observation_space['sub_input']) == 0:
                series_input = Input(shape=self._env.observation_space['series_data'].shape[0], name='series_data')
                series_net = Dense(16, activation='relu')(series_input)

                # state and advantage
                state_value = Dense(1, activation='linear', name='state_value')(series_net)
                advantage = Dense(self._n_action, activation='linear', name='advantage')(series_net)

                # concatenate
                concat = concatenate([state_value, advantage])

                # dueling layer
                output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self._n_action,), name='Q_value')(concat)
                return Model(inputs=series_input, outputs=output)

            else:
                series_input = Input(shape=self._env.observation_space['series_data'].shape[0], name='series_data')
                series_net = Dense(16, activation='relu')(series_input)
                series_net = Dense(4, activation='relu')(series_net)

                sub_net = {}
                for sub_input_name in self._env.observation_space['sub_input']:
                    sub_net[sub_input_name] = Input(shape=self._env.observation_space['sub_input'][sub_input_name].shape, name=sub_input_name)

                concat_list = [series_net]
                input_list = [series_input]
                for net in sub_net.values():
                    concat_list.append(net)
                    input_list.append(net)

                concat_1 = concatenate(concat_list)
                concat_1 = Dense(16, activation='relu')(concat_1)

                # state and advantage
                state_value = Dense(1, activation='linear', name='state_value')(concat_1)
                advantage = Dense(self._n_action, activation='linear', name='advantage')(concat_1)

                # concatenate
                concat_2 = concatenate([state_value, advantage])

                output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self._n_action,), name='Q_value')(concat_2)
                return Model(inputs=input_list, outputs=output)

        else: # not dueling
            if len(self._env.observation_space['sub_input']) == 0:
                series_input = Input(shape=self._env.observation_space['series_data'].shape[0], name='series_data')
                series_net = Dense(16, activation='relu')(series_input)
                series_net = Dense(16, activation='relu')(series_net)

                output = Dense(self._n_action, activation='linear')(series_net)
                return Model(inputs=series_input, outputs=output)
            else:
                series_input = Input(shape=self._env.observation_space['series_data'].shape[0], name='series_data')
                series_net = Dense(16, activation='relu')(series_input)
                series_net = Dense(4, activation='relu')(series_net)

                sub_net = {}
                for sub_input_name in self._env.observation_space['sub_input']:
                    sub_net[sub_input_name] = Input(shape=self._env.observation_space['sub_input'][sub_input_name].shape, name=sub_input_name)

                concat_list = [series_net]
                input_list = [series_input]
                for net in sub_net.values():
                    concat_list.append(net)
                    input_list.append(net)

                concat = concatenate(concat_list)
                concat = Dense(16, activation='relu')(concat)
                output = Dense(self._n_action, activation='linear')(concat)
                return Model(inputs=input_list, outputs=output)



    def learn(self, total_timesteps):
        self._step_count = 0
        episode_count = 0
        history = {'epi_len': [], 'epi_rew': [], 'total_step': [], 'ave_loss': []}

        while True: # loop episodes
            epi_len = 0
            epi_rew = 0.0
            loss_history =[]
            obs = self._env.reset()
            while True: # loop steps
                # decide action
                action = self._decide_action(obs, episode_count)
                # proceed environment
                next_obs, reward, done, _ = self._env.step(action)

                # store experience
                self._replay_buffer.append( 
                        np.array([obs, action, reward, next_obs], dtype=object) 
                        )
                # update observation
                obs = next_obs
                # increments
                self._step_count += 1
                epi_len += 1
                epi_rew += reward
                # experience replay
                if self._step_count > self._learning_starts \
                        and self._step_count%self._update_interval == 0:
                    if self._step_count%self._target_update_interval == 0:
                        update_targetQ = True
                    else:
                        update_targetQ = False
                    loss = self._experience_replay(update_targetQ)
                    loss_history.extend(loss.history['loss'])

                # judgement
                #if done or step_count == total_timesteps:
                if done:
                    if len(loss_history) != 0:
                        print(f'Episode {episode_count}:  reward: {epi_rew:.3f}, remain step: {total_timesteps-self._step_count}, loss: {np.average(loss_history):.5f}')
                    else:
                        print(f'Episode {episode_count}:  reward: {epi_rew:.3f}, remain step: {total_timesteps-self._step_count}')
                    break # from inside loop

            # each episode
            episode_count += 1
            history['total_step'].append(self._step_count)
            history['epi_len'].append(epi_len)
            history['epi_rew'].append(epi_rew)
            if len(loss_history) != 0:
                history['ave_loss'].append( sum(loss_history)/len(loss_history) )
            else:
                history['ave_loss'].append( np.nan )
            # End inside loop
            if self._step_count >= total_timesteps:
                break # from outside loop

        return history

    def _decide_action(self, obs, episode_count):
        if episode_count < self._eps_change_length:
            eps = self._initial_eps + (self._final_eps - self._initial_eps) * (episode_count/self._eps_change_length)
        else:
            eps = self._final_eps

        if eps < np.random.rand():
            # greedy
            series_input = obs[0].reshape(1, -1)
            sub_input = np.array([obs[1]])
            Q_values = self._Qfunc.predict([series_input, sub_input]).flatten()
            action = np.argmax(Q_values)
            
        else:
            # random
            action = np.random.randint(0, self._n_action)

        return action
    
    def _experience_replay(self, update_targetQ):
        obs_minibatch = []
        target_minibatch = []
        action_minibatch = []

        # define sampling size
        minibatch_size = min(len(self._replay_buffer), self._replay_batch_size)
        # choose experience batch randomly
        minibatch = np.array(random.sample(self._replay_buffer, minibatch_size),
                dtype=object)
        obs_batch = np.stack(minibatch[:,0], axis=0)
        action_batch = minibatch[:, 1]
        reward_batch = minibatch[:, 2]
        next_obs_batch = np.stack(minibatch[:,3], axis=0)

        main_obs_batch, sub_obs_batch = self._split_observation(obs_batch)
        next_main_obs_batch, next_sub_obs_batch = self._split_observation(next_obs_batch)

        # make target value
        current_Q_values = self._Qfunc.predict([main_obs_batch, sub_obs_batch])
        target_Q_values = current_Q_values.copy()


        if self._use_doubleDQN:
            """
            main_current_Q = main_Qfunc.predict([main_obs_batch, sub_obs_batch])
            main_next_Q    = main_Qfunc.predict([next_main_obs_batch, next_sub_obs_batch])
            target_next_Q    = target_Qfunc.predict([next_main_obs_batch, next_sub_obs_batch])
            next_action = np.argmax(main_next_Q)
            maxQ = target_next_Q[next_action]
            """
            main_next_Q = self._Qfunc.predict(
                    [next_main_obs_batch, next_sub_obs_batch])
            target_next_Q = self._target_Qfunc.predict(
                    [next_main_obs_batch, next_sub_obs_batch])
            next_actions = np.argmax(main_next_Q, axis=1)
            #print('main_next_Q')
            #print(main_next_Q)
            #print('target_next_Q')
            #print(target_next_Q)
            #print('next_actions')
            #print(next_actions)

            for i, (next_obs, next_action, target_Q_value, action, reward) \
                    in enumerate(zip(
                            next_obs_batch, 
                            next_actions, 
                            target_Q_values, 
                            action_batch, 
                            reward_batch)):
                maxQ = target_next_Q[i, next_action]
                #print('maxQ', maxQ)
                target_Q_value[action] = reward + self._gamma*maxQ


            if update_targetQ:
                print('Updating target Q function ...')
                # parameter value from main Q to target Q
                self._target_Qfunc.set_weights(self._Qfunc.get_weights())

        elif self._use_target_network:
            """
            target_next_Q = target_Qfunc.predict(next_state)
            next_action = np.argmax(target_next_Q)
            maxQ = target_next_Q[next_action]

            """
            target_next_Q_batch = self._target_Qfunc.predict(
                    [next_main_obs_batch, next_sub_obs_batch])
            maxQ_batch = np.max(target_next_Q_batch, axis=1)

            for target_Q_value, action, reward, maxQ \
                    in zip(target_Q_values, action_batch, reward_batch, maxQ_batch):
                target_Q_value[action] = reward + self._gamma*maxQ

            if update_targetQ:
                print('Updating target Q function ...')
                # parameter value from main Q to target Q
                self._target_Qfunc.set_weights(self._Qfunc.get_weights())

        else: # not use double DQN
            """
            main_next_Q = main_Qfunc.predict(next_state)
            next_action = np.argmax(main_next_Q)
            maxQ = main_next_Q[next_action]

            """
            #debug = self._Qfunc.predict([next_main_obs_batch, next_sub_obs_batch])
            main_next_Q_batch = self._Qfunc.predict(
                    [next_main_obs_batch, next_sub_obs_batch])
            maxQ_batch = np.max(main_next_Q_batch, axis=1)
            #print('target_Q_values')
            #print(target_Q_values)
            #print('main_next_Q_batch')
            #print(main_next_Q_batch)
            #print('maxQ_batch')
            #print(maxQ_batch)

            #next_target_Q = np.max(self._Qfunc.predict([next_main_obs_batch, next_sub_obs_batch]), axis=1)
            for target_Q_value, action, reward, maxQ \
                    in zip(target_Q_values, action_batch, reward_batch, maxQ_batch):
                target_Q_value[action] = reward + self._gamma*maxQ
            #print('target_Q_values')
            #print(target_Q_values)
            #sys.exit()


        # update parameters
        loss = self._Qfunc.fit(
                       [main_obs_batch, sub_obs_batch],
                       target_Q_values,
                       epochs=self._n_replay_epoch,
                       verbose=0
                       )
        return loss

    def simulate(self, visualize=False):
        obs = self._env.reset()
        done = False
        while not done:
            obs, reward, done, info = self._env.step(self._action(obs.reshape(1, -1, 1)))
            if visualize:
                self._env.render()

    def _action(self, obs):
        Q_values = self._Qfunc.predict(obs).flatten()
        return np.argmax(Q_values)

    def _clone_network(self, model):
        config = {
                'class_name': model.__class__.__name__,
                'config': model.get_config(),
                }
        clone = model_from_config(config, custom_objects={})
        clone.set_weights(model.get_weights())
        return clone

    def _split_observation(self, obs_batch):
        main_obs_batch = []
        sub_obs_batch = []
        for i in range(len(obs_batch)):
            main_obs_batch.append(list(obs_batch[i][0]))
            sub_obs_batch.append(list(obs_batch[i][1:]))
        main_obs_batch = np.array(main_obs_batch)
        sub_obs_batch = np.array(sub_obs_batch)

        return main_obs_batch, sub_obs_batch

    def _greedy_action(self, obs):
        series_input = obs[0].reshape(1, -1)
        sub_input = np.array([obs[1]])
        Q_values = self._Qfunc.predict([series_input, sub_input]).flatten()
        action = np.argmax(Q_values)
        return action

    def test_episode(self, eval_env):
        obs = eval_env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        prices = []
        taken_actions = []

        while not done:
            print('obs: ', obs[0][-1], obs[1])
            action = self._greedy_action(obs)
            print('action: ', action)
            prices.append(obs[0][-1])
            taken_actions.append(action)

            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            print('reward: ', reward)

            step_count += 1
            
        print('total_reward', total_reward)
        fig, ax = plt.subplots(1, 1, figsize=(10,8))
        ax.set_ylim(-1.2, 1.2)
        ax.plot(range(len(prices)), prices)
        action_color = ['C{}'.format(i) for i in taken_actions]
        ax.scatter(range(len(prices)), np.zeros(len(prices)), c=action_color)
        ax.scatter(len(prices)/2.0-0.3, -10.8, c='C0', label='Hold')
        ax.scatter(len(prices)/2.0+0.0, -10.8, c='C1', label='Buy')
        ax.scatter(len(prices)/2.0+0.3, -10.8, c='C2', label='Sell')
        ax.legend(fontsize=15)
        fig.savefig('test_episode.png')

    def load(cls, model_path):
        pass
        

    def save(self, path):
        if path[-1] != '/':
            path += '/'
        self._Qfunc.save(path+'mainQ')
        if self._use_doubleDQN or self._use_target_network:
            self._target_Qfunc.save(path+'targetQ')

        df_parameters = {
                'learning_rate': self._learning_rate,
                'learned_steps': self._step_count,
                'initial_eps': self._initial_eps,
                'final_eps': self._final_eps,
                'eps_change_length': self._eps_change_length,
                'update_interval': self._update_interval,
                'target_update_interval': self._target_update_interval,
                'use_target_network': self._use_target_network,
                'use_doubleDQN': self._use_doubleDQN,
                'gamma': self._gamma,
                'n_replay_epoch': self._n_replay_epoch,
                }

        with open(path+'parameters.json', mode='wt', encoding='utf-8') as file:
              json.dump(df_parameters, file, ensure_ascii=False, indent=2)






