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
from .hybriped_rough_config_wheeled import WheeledHybripedRoughCfg

class Wheeled_hybriped(LeggedRobot):
    cfg : WheeledHybripedRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, teleop=False):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, teleop)
    
    def _init_buffers(self):
        super()._init_buffers()
        
        self.p_gain_wheel = self.cfg.control.stiffness_wheel
        self.d_gain_wheel = self.cfg.control.damping_wheel
        self.wheel_idxs = []
        self.dof_idxs = []

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if "wheel" in name:
                self.wheel_idxs.append(i)
                self.p_gains[i] = self.p_gain_wheel
                self.d_gains[i] = self.d_gain_wheel
            else:
                self.dof_idxs.append(i)

        # # Acquire the GPU tensor for DOF states (which includes both positions and velocities)
        # dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # # Wrap the tensor using PyTorch for GPU manipulation
        # dof_state_tensor = gymtorch.wrap_tensor(dof_state_tensor)

        # for i in range(self.num_envs):
        #     for idx in self.wheel_idxs:
        #         dof_state_tensor[idx, 1] = 0.0
            
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state_tensor))

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        torques = torch.zeros(actions_scaled.shape, device=self.device)
        torques[:, self.dof_idxs] = self.p_gains[self.dof_idxs]*(actions_scaled[:, self.dof_idxs] + self.default_dof_pos.squeeze()[self.dof_idxs] - self.dof_pos[:, self.dof_idxs]) - self.d_gains[self.dof_idxs]*self.dof_vel[:, self.dof_idxs]
        torques[:, self.wheel_idxs] = self.p_gains[self.wheel_idxs]*(actions_scaled[:, self.wheel_idxs] - self.dof_vel[:, self.wheel_idxs]) - self.d_gains[self.wheel_idxs]*(self.dof_vel[:, self.wheel_idxs] - self.last_dof_vel[:, self.wheel_idxs])/self.sim_params.dt
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
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
            #torques_wheels = self._compute_torques_wheels()
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            #self.torques[:, self.wheel_idxs] = 0
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            #self.gym.set_dof_actuation_force_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.torques), gymtorch.unwrap_tensor(torch.tensor(self.dof_idxs)), len(self.dof_idxs))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        
