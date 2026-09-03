from dataclasses import dataclass

import minari
import numpy as np

@dataclass
class Dataset:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray

def qlearning_dataset(dataset: minari.MinariDataset) -> Dataset:
    obs, next_obs, actions, rewards, dones = [], [], [], [], []

    for episode in dataset.iterate_episodes():
        obs.append(episode.observations[:-1].astype(np.float32))
        next_obs.append(episode.observations[1:].astype(np.float32))
        actions.append(episode.actions.astype(np.float32))
        rewards.append(episode.rewards)
        dones.append(np.logical_or(episode.terminations, episode.truncations))

    return Dataset(
        observations=np.concatenate(obs),
        actions=np.concatenate(actions),
        next_observations=np.concatenate(next_obs),
        rewards=np.concatenate(rewards),
        terminals=np.concatenate(dones),
    )