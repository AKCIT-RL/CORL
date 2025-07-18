# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
# 2. https://github.com/karpathy/minGPT
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

# import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import trange

import minari

from algorithms.utils.wrapper_gym import get_env
from algorithms.utils.dataset import qlearning_dataset, ReplayBuffer
from algorithms.utils.save_video import save_video

# os.environ["MINARI_DATASETS_PATH"] = "/home/luanagbmartins/Documents/CEIA/offline_to_online/CORL/datasets"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "DT-D4RL"
    # wandb run name
    name: str = "DT"
    # transformer hidden dim
    embedding_dim: int = 128
    # depth of the transformer model
    num_layers: int = 3
    # number of heads in the attention
    num_heads: int = 1
    # maximum sequence length during training
    seq_len: int = 20
    # maximum rollout length, needed for the positional embeddings
    episode_len: int = 1000
    # attention dropout
    attention_dropout: float = 0.1
    # residual dropout
    residual_dropout: float = 0.1
    # embeddings dropout
    embedding_dropout: float = 0.1
    # maximum range for the symmetric actions, [-1, 1]
    max_action: float = 1.0
    # training dataset and evaluation environment
    env_name: str = "halfcheetah-medium-v2"
    dataset_id: str = "halfcheetah-medium-v2"
    # AdamW optimizer learning rate
    learning_rate: float = 1e-4
    # AdamW optimizer betas
    betas: Tuple[float, float] = (0.9, 0.999)
    # AdamW weight decay
    weight_decay: float = 1e-4
    # maximum gradient norm during training, optional
    clip_grad: Optional[float] = 0.25
    # training batch size
    batch_size: int = 64
    # total training steps
    update_steps: int = 100_000
    # warmup steps for the learning rate scheduler
    warmup_steps: int = 10_000
    # reward scaling, to reduce the magnitude
    reward_scale: float = 0.001
    # number of workers for the pytorch dataloader
    num_workers: int = 4
    # target return-to-go for the prompting durint evaluation
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    # number of episodes to run during evaluation
    eval_episodes: int = 100
    # evaluation frequency, will evaluate eval_every training steps
    eval_every: int = 10_000
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # configure PyTorch to use deterministic algorithms instead
    # of nondeterministic ones
    deterministic_torch: bool = False
    # training random seed
    train_seed: int = 10
    # evaluation random seed
    eval_seed: int = 42
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def custom_collate_fn(batch):
    """Custom collate function to handle the dataset items properly."""
    try:
        states, actions, returns, time_steps, masks = zip(*batch)
        
        # Convert to numpy arrays first to ensure consistent shapes
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        time_steps = np.array(time_steps, dtype=np.int64)
        masks = np.array(masks, dtype=np.float32)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        time_steps = torch.tensor(time_steps, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.float32)
        
        return states, actions, returns, time_steps, masks
    except Exception as e:
        print(f"Error in custom_collate_fn: {e}")
        print(f"Batch size: {len(batch)}")
        if len(batch) > 0:
            print(f"First item shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in batch[0]]}")
        raise


def load_d4rl_trajectories(
    dataset_id: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = minari.load_dataset(dataset_id)
    
    traj, traj_len = [], []

    obs = []

    data_ = defaultdict(list)
    for episode in dataset.iterate_episodes():
        data_["observations"] = episode.observations
        data_["actions"] = episode.actions
        data_["rewards"] = episode.rewards
        data_["terminals"] = episode.terminations | episode.truncations

        episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
        # return-to-go if gamma=1.0, just discounted returns else
        episode_data["returns"] = discounted_cumsum(
            episode_data["rewards"], gamma=gamma
        )
        traj.append(episode_data)
        traj_len.append(episode_data["actions"].shape[0])
        # reset trajectory buffer
        data_ = defaultdict(list)

        # Ensure observations are numpy arrays
        obs.append(np.array(episode.observations, dtype=np.float32))

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": np.concatenate(obs).mean(0, keepdims=True),
        "obs_std": np.concatenate(obs).std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(self, dataset_id: str, seq_len: int = 10, reward_scale: float = 1.0):
        self.dataset, info = load_d4rl_trajectories(dataset_id, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        
        # Get the actual length available from this starting point
        available_length = traj["observations"].shape[0] - start_idx
        actual_seq_len = min(self.seq_len, available_length)
        
        states = traj["observations"][start_idx : start_idx + actual_seq_len]
        actions = traj["actions"][start_idx : start_idx + actual_seq_len]
        returns = traj["returns"][start_idx : start_idx + actual_seq_len]
        time_steps = np.arange(start_idx, start_idx + actual_seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        
        # Always pad to seq_len, even if we have fewer elements
        mask = np.hstack(
            [np.ones(actual_seq_len), np.zeros(self.seq_len - actual_seq_len)]
        )
        
        # Pad all arrays to seq_len
        states = pad_along_axis(states, pad_to=self.seq_len)
        actions = pad_along_axis(actions, pad_to=self.seq_len)
        returns = pad_along_axis(returns, pad_to=self.seq_len)
        time_steps = pad_along_axis(time_steps, pad_to=self.seq_len, fill_value=0)

        # Ensure all arrays have the correct shape and dtype
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)
        time_steps = np.asarray(time_steps, dtype=np.int64)
        mask = np.asarray(mask, dtype=np.float32)

        # Validate shapes
        assert states.shape == (self.seq_len, states.shape[-1]), f"States shape: {states.shape}, expected: ({self.seq_len}, {states.shape[-1]})"
        assert actions.shape == (self.seq_len, actions.shape[-1]), f"Actions shape: {actions.shape}, expected: ({self.seq_len}, {actions.shape[-1]})"
        assert returns.shape == (self.seq_len,), f"Returns shape: {returns.shape}, expected: ({self.seq_len},)"
        assert time_steps.shape == (self.seq_len,), f"Time steps shape: {time_steps.shape}, expected: ({self.seq_len},)"
        assert mask.shape == (self.seq_len,), f"Mask shape: {mask.shape}, expected: ({self.seq_len},)"

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3]) * self.max_action
        return out


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if 'state' in obs:
            obs = obs['state']
        elif 'privileged_state' in obs:
            obs = obs['privileged_state']
        elif 'observation' in obs:
            obs = obs['observation']
        else:
            obs = list(obs.values())[0]
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs, dtype=np.float32)
    states[:, 0] = torch.as_tensor(obs, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        predicted_actions = model(
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        step_result = env.step(predicted_action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_state, reward, done, truncated, info = step_result
            done = done or truncated
        else:
            next_state, reward, done, info = step_result
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        if isinstance(next_state, dict):
            if 'state' in next_state:
                next_state = next_state['state']
            elif 'privileged_state' in next_state:
                next_state = next_state['privileged_state']
            elif 'observation' in next_state:
                next_state = next_state['observation']
            else:
                next_state = list(next_state.values())[0]
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        # Convert reward to tensor on the correct device before subtraction
        reward_tensor = torch.as_tensor(reward, device=device)
        returns[:, step + 1] = returns[:, step] - reward_tensor

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    wandb_init(asdict(config))

    # data & dataloader setup
    dataset = SequenceDataset(
        config.dataset_id, seq_len=config.seq_len, reward_scale=config.reward_scale
    )
    
    # Try with custom collate function first, fallback to single worker if issues persist
    try:
        trainloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
        )
    except Exception as e:
        print(f"Warning: Falling back to single worker due to: {e}")
        trainloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=0,  # Single worker
            collate_fn=custom_collate_fn,
        )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=get_env(config.env_name, config.device),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    # model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            for target_return in config.target_returns:
                # Try to seed the environment, but handle cases where it might not work
                try:
                    eval_env.seed(config.eval_seed)
                except (AttributeError, TypeError):
                    # Some environments don't have a seed method or it's not callable
                    pass
                eval_returns = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)

                # normalized_scores = (
                #     eval_env.get_normalized_score(np.array(eval_returns)) * 100
                # )
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        # f"eval/{target_return}_normalized_score_mean": np.mean(
                        #     normalized_scores
                        # ),
                        # f"eval/{target_return}_normalized_score_std": np.std(
                        #     normalized_scores
                        # ),
                    },
                    step=step,
                )
            model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))


if __name__ == "__main__":
    train()
