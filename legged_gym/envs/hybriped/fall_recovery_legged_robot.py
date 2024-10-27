# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .fall_recovery_config import FallRecoveryCfg

import json

class FallRecovery(LeggedRobot):
    cfg : FallRecoveryCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, teleop=False):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, teleop)
        with open(os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'scripts', "init_poses_collected.json"), 'r') as json_file:
            self.data = json.load(json_file)
        self.succesful_recoveries = 0

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self.collect_poses_idx_random = np.random.randint(0, len(self.data["base_pos"]), size=len(env_ids))
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self.extras["episode"] = {}
        self.extras["episode"]['Succesful recoveries'] = self.succesful_recoveries
        

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = torch.tensor(np.array(self.data["dof_pos"])[self.collect_poses_idx_random], device=self.device, dtype=torch.float)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] = self.env_origins[env_ids][:, :2]
        self.root_states[env_ids, 2] = torch.tensor(np.array(self.data["base_pos"])[self.collect_poses_idx_random], device=self.device, dtype=torch.float)[:, 2]
        self.root_states[env_ids, 3:7] = torch.tensor(np.array(self.data["base_orientation"])[self.collect_poses_idx_random], device=self.device, dtype=torch.float)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
              
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def check_termination(self):
        """ Check if environments need to be reset: Time_out or big shift in position
        """
        self.reset_buf = torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1) > self.cfg.asset.distance_threshold_termination
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        #self.standing_pose_reached = torch.logical_and(torch.norm(self.dof_pos-self.default_dof_pos, dim=-1)<self.cfg.asset.pose_error_threshold_termination, torch.abs(self.root_states[:, 2] - self.cfg.rewards.base_height_target)<self.cfg.asset.height_error_threshold_termination)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.standing_pose_reached = torch.logical_and(torch.norm(self.dof_pos-self.default_dof_pos, dim=-1)<self.cfg.asset.pose_error_threshold_termination, torch.sum(contact_filt, dim=1)==4)
        self.succesful_recoveries += torch.sum(self.standing_pose_reached).item()
        #print(self.reset_buf, self.time_out_buf, self.standing_pose_reached)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.standing_pose_reached
        

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.dof_pos * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)   
        
    def _reward_base_height_flat(self):
        #print(torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target))
        return torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)

    def _reward_joint_pos_track(self):
        pos_track_error = torch.sum(torch.square(self.dof_pos-self.default_dof_pos), dim=1)
        return torch.exp(-pos_track_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_base_pos_shift(self):
        return torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=-1)
    
    def _reward_base_orientation_track(self):
        target_orientation = torch.zeros(self.projected_gravity.shape, device=self.device)
        target_orientation[:, 2] = -1
        orientation_track_error = torch.norm(target_orientation - self.projected_gravity, dim=-1)
        #print(torch.exp(-orientation_track_error/self.cfg.rewards.tracking_sigma))
        return torch.exp(-orientation_track_error/self.cfg.rewards.tracking_sigma)
        