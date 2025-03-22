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
from legged_gym.utils.terrain_eval import Terrain_Eval

from .eval_robot_config import EvalRobotCfg
class EvalRobot(LeggedRobot):
    cfg : EvalRobotCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, teleop=False):
        self.info = []
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, teleop)
        print(self.info)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain_Eval(self.cfg.terrain, self.num_envs, self.info)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins.view(-1, 3)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        #self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf = torch.logical_or(self.projected_gravity[:, 2] > 0, self.episode_length_buf > self.max_episode_length)
        distance_xy = torch.abs(self.root_states[:, :2] - self.env_origins[:, :2])
        self.terrain_solved = torch.logical_or(distance_xy[:, 0]>self.terrain.env_length / 2, distance_xy[:, 1]>self.terrain.env_length / 2) 
        self.reset_buf |= self.terrain_solved
        for i in range(self.terrain_solved.shape[0]):
            if self.info[i]["total"] < self.cfg.logger.number_evaluations:
                self.info[i]["vel_x"][-1].append(self.base_lin_vel[i, 0].item())
                self.info[i]["vel_y"][-1].append(self.base_lin_vel[i, 1].item())
                self.info[i]["torques"][-1].append(self.clipped[i].cpu().numpy().tolist())
                self.info[i]["successful"] += int(self.terrain_solved[i])
                self.info[i]["total"] += int(self.reset_buf[i])
                if int(self.reset_buf[i]) and self.info[i]["total"] < self.cfg.logger.number_evaluations:
                    self.info[i]["vel_x"].append([]), self.info[i]["vel_y"].append([])
                    self.info[i]["torques"].append([])

    def _resample_commands(self, env_ids, teleop=False):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.cfg.logger.vel_x:
            self.commands[env_ids, 0] = self.cfg.logger.linear_vel*torch.ones(len(env_ids), 1, device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(-self.cfg.logger.perpendicular_vel_max, self.cfg.logger.perpendicular_vel_max, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 0] = torch_rand_float(-self.cfg.logger.perpendicular_vel_max, self.cfg.logger.perpendicular_vel_max, (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = self.cfg.logger.linear_vel*torch.ones(len(env_ids), 1, device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch.zeros(len(env_ids), 1, device=self.device).squeeze(1)
        