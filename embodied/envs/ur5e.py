from typing import Tuple

import embodied
import numpy as np

from ur_env.remote import RemoteEnvClient

Address = Tuple[str, int]


class UR5e(embodied.Env):

    def __init__(self, task, address: Address, repeat=1):
        # TODO: dummy and real version needed to obtain proper env_spaces.
        # describe env_specs by hand
        if task == 'real':
            self._env = RemoteEnvClient(address)
        else:
            self._env = None
        self._task = task
        self._repeat = repeat
        self._done = True

    @property
    def obs_space(self):
        # One cannot obtain env.observation_spec() from learners dummy env.
        # So obs_space remains to be hardcoded.
        spaces = {
            'kinect/image': embodied.Space(np.uint8, (64, 64, 3)),
            'kinect/depth': embodied.Space(np.uint8, (64, 64, 1)),
            'arm/ActualTCPPose': embodied.Space(np.float32, (6,)),
            'arm/ActualQ': embodied.Space(np.float32, (6,)),
            'gripper/pos': embodied.Space(np.float32, (1,)),
            'gripper/object_detected': embodied.Space(np.float32, (1,)),
        }
        spaces.update(
            reward=embodied.Space(np.float32),
            is_first=embodied.Space(bool),
            is_last=embodied.Space(bool),
            is_terminal=embodied.Space(bool),
        )
        return spaces

    @property
    def act_space(self):
        lim = np.full((4,), 1.)
        return {
            'action': embodied.Space(
                np.float32, None, -lim, lim),
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
        return dict(
            reward=reward,
            is_first=time_step.first(),
            is_last=time_step.last(),
            is_terminal=time_step.discount == 0,
            **time_step.observation,
        )

    def close(self):
        if self._task == 'real':
            self._env.close()
