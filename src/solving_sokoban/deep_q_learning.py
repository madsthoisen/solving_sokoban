import copy
import heapq
import random

import numpy as np
import tensorflow as tf

from collections import defaultdict, deque
from itertools import permutations
from tensorflow import keras
from tensorflow.keras import initializers, losses

from .game_engine import State


def agent(state_shape, action_shape):
    learning_rate = 0.001
    #init = initializers.Zeros()
    init = initializers.HeUniform()
    #loss = losses.MeanSquaredError()
    loss = losses.Huber()

    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(action_shape))
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model


def train(state, replay_memory, model, target_model, completed, state_shape):
    learning_rate = 0.7
    discount_factor = 0.95

    MIN_REPLAY_SIZE = 128*2
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, completed) in enumerate(mini_batch):
        if not completed:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
