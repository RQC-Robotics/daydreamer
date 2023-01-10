import embodied
import numpy as np

from ur_env.remote import RemoteEnvClient


class UR5e(embodied.Env):

    def __init__(self, address, repeat=1):
        print("All images must be of a size (64, 64)")
        self._env = RemoteEnvClient(address)
        self._repeat = repeat

        self._done = True
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):
        spaces = self._env.observation_spec().copy()
        for key, space in spaces.items():
            spaces[key] = embodied.Space(space.dtype, space.shape)
        spaces.update(
            reward=embodied.Space(np.float32),
            is_first=embodied.Space(bool),
            is_last=embodied.Space(bool),
            is_terminal=embodied.Space(bool),
        )
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        return {
            'action': embodied.Space(
                np.float32, spec.shape, spec.minimum, spec.maximum),
            'reset': embodied.Space(bool)
        }

    def step(self, action):
        if action['reset'] or self._done:
            time_step = self._env.reset()
            self._done = False
            return self._obs(time_step, 0.0)
        assert np.isfinite(action['action']).all()
        reward = 0.0
        for _ in range(self._repeat):
            time_step = self._env.step(action['action'])
            reward += time_step.reward or 0.0
            if time_step.last():
                self._done = True
                break
        assert time_step.discount in (0, 1)
        return self._obs(time_step, reward)

    def _obs(self, time_step, reward):
        obs = {
            k: v[None] if v.shape == () else v
            for k, v in dict(time_step.observation).items()
            if k not in self._ignored_keys
        }
        return dict(
            reward=reward,
            is_first=time_step.first(),
            is_last=time_step.last(),
            is_terminal=time_step.discount == 0,
            **obs,
        )

    def close(self):
        self._env.close()
