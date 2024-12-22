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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .config import *

class EvalRobotCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_actions = 12
        episode_length_s = 40 

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        num_rows= 1 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)

        terrain_type = "slope"
        terrain_direction_up = False
        
        slope_treshold = 1.732

        min_height = 0.05
        max_height = 0.40

        step_width = 0.31
        num_rectangles = 20
        rectangle_min_size = 1.
        rectangle_max_size = 2.

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = GAIT_3_STABLE

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'j1': 100., 'j2': 100., 'j3': 100.}  # [N*m/rad]
        damping = {'j1': 4., 'j2': 4., 'j3': 4.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = "/home/zetans/Desktop/rl_hybriped/assets/urdf/hybriped/urdf/hybriped_simplified_limits_cylinder_gait3_free.urdf"
        name = "hybriped"
        foot_name = "wheel"
        penalize_contacts_on = ["link2", "link3"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        use_mesh_materials = True

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            pass
    
    class logger:
        linear_vel = 0.75
        perpendicular_vel_max = linear_vel/10
        vel_x = False

        number_evaluations = 100

class EvalRobotCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'hybriped'
        load_run = -1
        max_iterations = 2000
