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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.hybriped.eval_robot_config import EvalRobotCfg
from .terrain import Terrain
class Terrain_Eval(Terrain):
    def __init__(self, cfg: EvalRobotCfg.terrain, num_robots, info) -> None:

        self.cfg = cfg
        self.info = info
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        self.step_width = cfg.step_width
        self.num_rectangles = cfg.num_rectangles
        self.rectangle_min_size = cfg.rectangle_min_size 
        self.rectangle_max_size = cfg.rectangle_max_size 

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        
        #self.create_sloped_terrain(slope=0.83)
        self.create_random_terrain(terrain_type=self.cfg.terrain_type, up=self.cfg.terrain_direction_up)
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    def create_random_terrain(self, terrain_type="slope", up=1):
        slope = None
        height = None

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
                if terrain_type=="slope":
                    slope = np.random.uniform(0, self.cfg.slope_treshold)
                    terrain_utils.pyramid_sloped_terrain(terrain, slope=(2*(int)(up>0)-1)*slope, platform_size=3.)
                elif terrain_type=="stairs":
                    height = np.random.uniform(self.cfg.min_height, self.cfg.max_height)
                    terrain_utils.pyramid_stairs_terrain(terrain, step_width=self.step_width, step_height=(2*(int)(up>0)-1)*height, platform_size=3.)
                elif terrain_type=="discrete":
                    height = np.random.uniform(self.cfg.min_height, self.cfg.max_height)
                    terrain_utils.discrete_obstacles_terrain(terrain, height, self.rectangle_min_size, self.rectangle_max_size, self.num_rectangles, platform_size=3.)
                elif terrain_type=="plane":
                    slope = 0
                    terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)

                self.add_terrain_to_map(terrain, j, i)
                self.info.append(dict())
                self.info[-1]["row_col"] = (i, j)
                self.info[-1]["type"] = terrain_type
                self.info[-1]["height"] = height
                self.info[-1]["slope"] = slope
                self.info[-1]["commands"] = []
                self.info[-1]["direction"] = "up" if up else "down"
                self.info[-1]["successful"] = 0; self.info[-1]["total"] = 0 
                self.info[-1]["vel_x"] = [[]]; self.info[-1]["vel_y"] = [[]]; self.info[-1]["vel_w"] = [[]]
                self.info[-1]["torques"] = [[]]
        return terrain

    def create_sloped_terrain(self, slope):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
                terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                self.add_terrain_to_map(terrain, i, j)
                self.info.append(dict())
                self.info[-1]["row_col"] = (i, j)
                self.info[-1]["type"] = "slope"
                self.info[-1]["height"] = None
                self.info[-1]["slope"] = slope
                self.info[-1]["direction"] = "up"
                self.info[-1]["successful"] = 0; self.info[-1]["total"] = 0 
                self.info[-1]["vel_x"] = [[]]; self.info[-1]["vel_y"] = [[]]
        return terrain

    def create_stairs_terrain(self, step_height, step_width=0.31):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
                terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=3.)
                self.add_terrain_to_map(terrain, i, j)
                self.info.append(dict())
                self.info[-1]["row_col"] = (i, j)
                self.info[-1]["type"] = "stairs"
                self.info[-1]["height"] = step_height
                self.info[-1]["slope"] = None
                self.info[-1]["direction"] = "up"
                self.info[-1]["successful"] = 0; self.info[-1]["total"] = 0 
                self.info[-1]["vel_x"] = [[]]; self.info[-1]["vel_y"] = [[]]
        return terrain

    def create_discrete_obstacles_terrain(self, height):
        num_rectangles = 20
        rectangle_min_size = 1.
        rectangle_max_size = 2.
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
                terrain_utils.discrete_obstacles_terrain(terrain, height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
                self.add_terrain_to_map(terrain, i, j)
        return terrain
            
