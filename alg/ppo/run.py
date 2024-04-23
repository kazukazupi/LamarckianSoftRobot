import csv
import os
import time
from collections import deque
from pathlib import Path

import evogym.envs  # type: ignore
import numpy as np
import torch

from alg import Config
from alg.a2c_ppo_acktr import algo
from alg.a2c_ppo_acktr.model import Policy
from alg.a2c_ppo_acktr.storage import RolloutStorage
from alg.ppo import utils
from alg.ppo.envs import make_vec_envs
from alg.ppo.evaluate import evaluate
from alg.structure import Structure

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

LOG_DIR_NAME = "log_dir"
RL_CSV_FILE_NAME = "log.csv"
ACTOR_CRITIC_FILE_NAME = "actor_critic.pt"


def run_ppo(structure: Structure, saving_dir: Path, config: Config):

    termination_condition = utils.TerminationCondition(config.max_iters)

    if config.print_en:
        print(f"Starting training on \n{structure.body}\nat {saving_dir}...\n")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.cuda and torch.cuda.is_available() and config.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # ---------------------
    #  1. prepare logging
    # ---------------------

    # save log at saving_dir/log.csv
    csv_file = saving_dir / RL_CSV_FILE_NAME
    if config.print_en:
        print(f"write log at {csv_file}")
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["updates", "num_episodes", "num_timesteps", "mean", "median", "min", "max"]
        )

    # save neural network
    actor_critic_file = saving_dir / ACTOR_CRITIC_FILE_NAME
    if config.print_en:
        print(f"save actor_critic at {actor_critic_file}")

    # ------------------------------------
    #  2. get environment and controller
    # ------------------------------------
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.cuda else "cpu")
    log_dir = os.path.join(saving_dir, LOG_DIR_NAME)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    envs = make_vec_envs(
        env_name=config.env_name,
        robot_structure=structure,
        seed=None,
        num_processes=config.num_processes,
        gamma=config.gamma,
        log_dir=log_dir,
        device=device,
        allow_early_resets=False,
    )

    # from ga.inherit_controller import get_controller

    # actor_critic = get_controller(
    #     body=body,
    #     observation_space_shape=envs.observation_space.shape,
    #     action_space=envs.action_space,
    #     parents=parents,
    #     crossover_info=crossover_info,
    # )

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={"recurrent": config.recurrent_policy},
    )

    # # if parent_robot_dir is not None:
    # #     assert(config.inherit_en)
    # #     utils.inherit_policy(actor_critic, body, parent_robot_dir, config.env_name)
    # #     if config.print_en: print(f'inherit policy from {parent_robot_dir}')

    actor_critic.to(device)

    # ----------------------
    #  3. train controller
    # ----------------------

    agent = algo.PPO(
        actor_critic,
        config.clip_param,
        config.ppo_epoch,
        config.num_mini_batch,
        config.value_loss_coef,
        config.entropy_coef,
        lr=config.lr,
        eps=config.eps,
        max_grad_norm=config.max_grad_norm,
    )

    rollouts = RolloutStorage(
        config.num_steps,
        config.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)  # type: ignore
    num_episdoes = 0

    start = time.time()
    num_updates = int(config.num_env_steps) // config.num_steps // config.num_processes

    rewards_tracker = []
    avg_rewards_tracker = []
    sliding_window_size = 10
    max_determ_avg_reward = float("-inf")

    # end when j == min{tc.max_iters, num_updates}
    for j in range(num_updates):

        if config.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, num_updates, config.lr)

        for step in range(config.num_steps):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # track rewards
            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])
                    rewards_tracker.append(info["episode"]["r"])
                    num_episdoes += 1
                    if len(rewards_tracker) < 10:
                        avg_rewards_tracker.append(
                            np.average(np.array(rewards_tracker))
                        )
                    else:
                        avg_rewards_tracker.append(
                            np.average(np.array(rewards_tracker[-10:]))
                        )

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            config.use_gae,
            config.gamma,
            config.gae_lambda,
            config.use_proper_time_limits,
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # print status
        if j % config.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config.num_processes * config.num_steps
            end = time.time()
            if config.print_en:
                print(
                    "Updates {}, num episodes {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                        j,
                        num_episdoes,
                        total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                    )
                )

            with open(csv_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        j,
                        num_episdoes,
                        total_num_steps,
                        round(np.mean(episode_rewards), 2),
                        round(np.median(episode_rewards), 2),
                        round(np.min(episode_rewards), 2),
                        round(np.max(episode_rewards), 2),
                    ]
                )

        # evaluate the controller and save it if it does the best so far
        if len(episode_rewards) > 1 and (
            j % config.eval_interval == 0 or termination_condition(j)
        ):

            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(
                num_evals=config.num_evals,
                actor_critic=actor_critic,
                obs_rms=obs_rms,
                env_name=config.env_name,
                robot_structure=structure,
                seed=config.seed,
                num_processes=config.num_processes,
                eval_log_dir=log_dir,
                device=device,
            )

            if config.print_en:
                print(
                    f"Evaluated using {config.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n"
                )

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward

                if config.print_en:
                    print(
                        f"Saving {actor_critic_file} with avg reward {max_determ_avg_reward}\n"
                    )
                torch.save(
                    [
                        actor_critic,
                        getattr(utils.get_vec_normalize(envs), "obs_rms", None),
                    ],
                    actor_critic_file,
                )

        # return upon reaching the termination condition
        if not termination_condition == None:
            if termination_condition(j):
                if config.print_en:
                    print(f"met termination condition ({j})...terminating...\n")
                return max_determ_avg_reward
