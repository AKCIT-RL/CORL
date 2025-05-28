import minari
import numpy as np
from typing import Dict
import torch

def qlearning_dataset(dataset: minari.MinariDataset) -> Dict[str, np.ndarray]:
    obs, next_obs, actions, rewards, dones = [], [], [], [], []

    for episode in dataset.iterate_episodes():
        obs.append(episode.observations[:-1].astype(np.float32))
        next_obs.append(episode.observations[1:].astype(np.float32))
        actions.append(episode.actions.astype(np.float32))
        rewards.append(episode.rewards)
        dones.append(episode.terminations | episode.truncations)

    return {
        "observations": np.concatenate(obs),
        "actions": np.concatenate(actions),
        "next_observations": np.concatenate(next_obs),
        "rewards": np.concatenate(rewards),
        "terminals": np.concatenate(dones),
    }


class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, dataset_dict):
        self.observations = dataset_dict["observations"]
        self.actions = dataset_dict["actions"]
        self.rewards = dataset_dict["rewards"]
        self.next_observations = dataset_dict["next_observations"]
        self.terminals = dataset_dict["terminals"]
        self.size = len(self.observations)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.observations[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.tensor([self.rewards[idx]], dtype=torch.float32),
            torch.from_numpy(self.next_observations[idx]),
            torch.tensor([self.terminals[idx]], dtype=torch.float32),
        ]     