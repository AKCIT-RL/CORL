import numpy as np
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Any
import jax

class Batch(NamedTuple):
   observations: jnp.ndarray | np.ndarray
   actions: jnp.ndarray | np.ndarray
   rewards: jnp.ndarray | np.ndarray
   next_observations: jnp.ndarray | np.ndarray
   dones: jnp.ndarray | np.ndarray

class ReplayBuffer:
   def __init__(
      self, 
      state_dim: int, 
      action_dim: int, 
      buffer_size: int,
      stratify: bool = False,
      seed: int = 0
   ):
      self._buffer_size = buffer_size
      self._pointer = 0
      self._size = 0
      self._stratify = stratify
      self._rng = np.random.default_rng(seed)
      self._minari_loaded: bool = False
      self._minari_pointers: tuple[int, int] = (0, 0)

      self._data = Batch(
         observations=np.zeros((buffer_size, state_dim), dtype=np.float32),
         actions=np.zeros((buffer_size, action_dim), dtype=np.float32),
         rewards=np.zeros((buffer_size,), dtype=np.float32),
         next_observations=np.zeros((buffer_size, state_dim), dtype=np.float32),
         dones=np.zeros((buffer_size,), dtype=np.float32),
      )

   def size(self) -> int:
      return self._size

   def max_size(self) -> int:
      return self._buffer_size

   def _to_numpy(self, value) -> np.ndarray:
      return np.asarray(value, dtype=np.float32)

   def load_dataset(
      self, 
      dataset,
      config,
      clip_to_eps: bool = True,
      eps: float = 1e-5
   ) -> Tuple[Any, Any]:
      if self._size != 0:
         raise ValueError("Trying to load data into non-empty replay buffer")

      obs = self._to_numpy(dataset.observations)
      act = self._to_numpy(dataset.actions)
      rew = self._to_numpy(dataset.rewards)
      next_obs = self._to_numpy(dataset.next_observations)
      
      if clip_to_eps:
         lim = 1 - eps
         act = np.clip(act, -lim, lim)

      imputed_next_observations = np.roll(obs, -1, axis=0)
      same_obs = np.all(
         np.isclose(imputed_next_observations, next_obs, atol=1e-5),
         axis=-1,
      )
      dones = 1.0 - same_obs.astype(np.float32)
      dones[-1] = 1.0
      
      n_transitions = min(self._buffer_size, len(obs))

      perm = self._rng.permutation(len(obs))[:n_transitions]

      self._data.observations[:n_transitions] = obs[perm]
      self._data.actions[:n_transitions] = act[perm]
      self._data.rewards[:n_transitions] = rew[perm]
      self._data.next_observations[:n_transitions] = next_obs[perm]
      self._data.dones[:n_transitions] = dones[perm]
      self._minari_loaded = True
      self._minari_pointers = (0, n_transitions)

      self._size = n_transitions
      self._pointer = n_transitions % self._buffer_size

      obs_mean, obs_std = 0, 1
      if getattr(config, 'normalize_state', False) or getattr(config, 'normalize', False):
         valid_obs = self._data.observations[:self._size]
         obs_mean = valid_obs.mean(0)
         obs_std = valid_obs.std(0) + eps

         self._data.observations[:self._size] = (self._data.observations[:self._size] - obs_mean) / obs_std
         self._data.next_observations[:self._size] = (self._data.next_observations[:self._size] - obs_mean) / obs_std

      return obs_mean, obs_std

   def _update_minari_pointers(self, write_start: int, count: int) -> None:
      """Update the contiguous [start, end) slice of Minari data after writing."""
      if not self._minari_loaded or count <= 0:
         return
      if count >= self._buffer_size:
         self._minari_pointers = (0, 0)
         self._minari_loaded = False
         return

      start, end = self._minari_pointers

      def _shrink_for_range(s: int, e: int, a: int, b: int) -> tuple[int, int]:
         if s >= e or b <= s or a >= e:
            return s, e
         if a <= s:
            return (0, 0) if b >= e else (b, e)
         if b >= e:
            return s, a
         # Overwrite splits [s, e) into [s, a) and [b, e); keep the larger contiguous slice.
         return (s, a) if (a - s) >= (e - b) else (b, e)

      end_idx = write_start + count
      if end_idx <= self._buffer_size:
         start, end = _shrink_for_range(start, end, write_start, end_idx)
      else:
         start, end = _shrink_for_range(start, end, write_start, self._buffer_size)
         start, end = _shrink_for_range(start, end, 0, end_idx - self._buffer_size)

      self._minari_pointers = (start, end)
      if start >= end:
         self._minari_loaded = False

   def add_transition(self, observation, action, reward, next_observation, done):
      write_start = self._pointer
      self._data.observations[write_start] = self._to_numpy(observation)
      self._data.actions[write_start] = self._to_numpy(action)
      self._data.rewards[write_start] = self._to_numpy(reward)
      self._data.next_observations[write_start] = self._to_numpy(next_observation)
      self._data.dones[write_start] = self._to_numpy(done)

      self._pointer = (write_start + 1) % self._buffer_size
      self._size = min(self._size + 1, self._buffer_size)
      self._update_minari_pointers(write_start, 1)

   def add_transitions(self, observations, actions, rewards, next_observations, dones):
      obs = self._to_numpy(observations)
      acts = self._to_numpy(actions)
      rews = self._to_numpy(rewards)
      next_obs = self._to_numpy(next_observations)
      dns = self._to_numpy(dones)

      n = len(obs)
      if n == 0:
         return
         
      if n > self._buffer_size:
         obs = obs[-self._buffer_size:]
         acts = acts[-self._buffer_size:]
         rews = rews[-self._buffer_size:]
         next_obs = next_obs[-self._buffer_size:]
         dns = dns[-self._buffer_size:]
         n = self._buffer_size

      write_start = self._pointer
      end_idx = write_start + n

      if end_idx <= self._buffer_size:
         self._data.observations[write_start:end_idx] = obs
         self._data.actions[write_start:end_idx] = acts
         self._data.rewards[write_start:end_idx] = rews
         self._data.next_observations[write_start:end_idx] = next_obs
         self._data.dones[write_start:end_idx] = dns
      else:
         overflow = end_idx - self._buffer_size
         first_part = n - overflow

         self._data.observations[write_start:] = obs[:first_part]
         self._data.actions[write_start:] = acts[:first_part]
         self._data.rewards[write_start:] = rews[:first_part]
         self._data.next_observations[write_start:] = next_obs[:first_part]
         self._data.dones[write_start:] = dns[:first_part]

         self._data.observations[:overflow] = obs[first_part:]
         self._data.actions[:overflow] = acts[first_part:]
         self._data.rewards[:overflow] = rews[first_part:]
         self._data.next_observations[:overflow] = next_obs[first_part:]
         self._data.dones[:overflow] = dns[first_part:]

      self._pointer = end_idx % self._buffer_size
      self._size = min(self._size + n, self._buffer_size)
      self._update_minari_pointers(write_start, n)

   def all(self) -> Batch:
      return jax.tree_util.tree_map(
         lambda x: jnp.asarray(x[:self._size], dtype=jnp.float32), 
         self._data
      )

   def sample(self, batch_size: int, minari_percent: float = 0.0) -> Batch:
      if self._size == 0:
         raise ValueError("Replay buffer is empty, cannot sample.")

      if self._stratify and self._minari_loaded and minari_percent > 0.0:
         minari_start, minari_end = self._minari_pointers
         minari_local_size = minari_end - minari_start
         non_minari_local_size = self._size - minari_local_size

         if minari_local_size <= 0:
            indices = self._rng.integers(0, self._size, size=batch_size)
         elif non_minari_local_size <= 0:
            indices = self._rng.integers(minari_start, minari_end, size=batch_size)
         else:
            minari_size = int(batch_size * minari_percent)
            non_minari_size = batch_size - minari_size

            if minari_size > 0:
               minari_indices = self._rng.integers(
                  minari_start,
                  minari_end,
                  size=minari_size,
               )
            else:
               minari_indices = np.empty(0, dtype=np.int64)

            if non_minari_size > 0:
               u = self._rng.integers(0, non_minari_local_size, size=non_minari_size)
               non_minari_indices = np.where(
                  u < minari_start,
                  u,
                  u - minari_start + minari_end,
               )
            else:
               non_minari_indices = np.empty(0, dtype=np.int64)

            indices = np.concatenate([minari_indices, non_minari_indices])
            self._rng.shuffle(indices)
      else:
         indices = self._rng.integers(0, self._size, size=batch_size)

      return jax.tree_util.tree_map(
         lambda x: jnp.asarray(x[indices], dtype=jnp.float32), 
         self._data
      )