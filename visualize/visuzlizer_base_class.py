import sys
from typing import List, Optional, cast

import cv2  # type: ignore
import evogym.envs  # type: ignore
import numpy as np
import torch

sys.path.append(".")

from alg import Structure
from alg.a2c_ppo_acktr.model import Policy  # type: ignore
from alg.ppo.envs import make_vec_envs  # type: ignore
from alg.ppo.utils import get_vec_normalize  # type: ignore


class Visualizer:
    def __init__(
        self,
        structure: Structure,
        actor_critic: Policy,
        env_name: str,
        movie_path: Optional[str] = None,
        num_episodes: int = 1,
        envs=None,
        obs_rms=None,
    ):
        """
        visualize the motion of a robot.

        Args:
            structure: structure of the robot
            actor_critic: controller of the robot
            movie_path: path to save movie (if is None, simply render)
            num_episodes: the number of episodes
        """
        self.structure = structure
        self.actor_critic = actor_critic
        self.env_name = env_name
        self.movie_path = movie_path
        self.num_episodes = num_episodes

        if envs is not None:
            self.envs = envs
        else:
            self.envs = make_vec_envs(
                env_name=self.env_name,
                robot_structure=self.structure.as_tuple(),
                seed=100,
                num_processes=1,
                gamma=None,
                log_dir="./visualize/log_dir",
                device="cpu",
                allow_early_resets=True,
            )

        self.obs_rms = obs_rms

    def run(self):

        vec_norm = get_vec_normalize(self.envs)
        if self.obs_rms is not None:
            vec_norm.obs_rms = self.obs_rms

        # initial input for the neural network
        recurrent_hidden_states = torch.zeros(
            1, self.actor_critic.recurrent_hidden_state_size
        )
        masks = torch.zeros(1, 1)
        obs = self.envs.reset()

        # log culminative reward of each episode
        sum_reward = torch.tensor([[0.0]])
        sum_reward_list: List[torch.Tensor] = []

        if self.movie_path is not None:
            # The frames of the episode that earned the highest cumulative reward.
            best_frames: Optional[List[np.ndarray]] = None
        else:
            self.envs.render("screen")

        frames: List[np.ndarray] = []

        while len(sum_reward_list) < self.num_episodes:

            # act
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = self.actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True
                )

            obs, reward, done, infos = self.envs.step(action)

            sum_reward += reward

            if self.movie_path is not None:
                frame = self.envs.render(mode="img")
                # cv2.putText(
                #     frame,
                #     f"culminative reward: {round(sum_reward.numpy().flatten()[0], 2)}",
                #     (20, 50),
                #     cv2.FONT_HERSHEY_DUPLEX,
                #     1.0,
                #     (0, 0, 0),
                # )
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
            else:
                self.envs.render("screen")

            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device="cpu",
            )

            for info in infos:
                # end episode
                if "episode" in info.keys() or "TimeLimit.truncated" in info.keys():
                    if self.movie_path is not None and (
                        best_frames is None
                        or all([sum_reward > val for val in sum_reward_list])
                    ):
                        best_frames = frames
                    frames = []
                    sum_reward_list.append(sum_reward)
                    print(f"sum_reward: {sum_reward.numpy().flatten()[0]:.03f}")
                    sum_reward = torch.tensor(0)

        self.envs.close()

        # write video
        if self.movie_path is not None:
            print("Writing Video...")
            assert best_frames is not None
            best_frames_ = cast(List[np.ndarray], best_frames)
            frame = best_frames_[0]
            shape = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            writer = cv2.VideoWriter(self.movie_path, fourcc, 50, shape)

            for frame in best_frames_:
                writer.write(frame)

            writer.release()
