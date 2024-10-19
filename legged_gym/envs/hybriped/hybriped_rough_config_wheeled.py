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

class WheeledHybripedRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 16

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "l4_j1": 0.0,      # [rad]
            "l4_j2": 0.9,      # [rad]
            "l4_j3": 0.7,      # [rad]
            "l4_wheel": 0.0,      # [rad]

            "l3_j1": 0.0,      # [rad]
            "l3_j2": -0.9,      # [rad]
            "l3_j3": -0.7,      # [rad]
            "l3_wheel": 0.0,      # [rad]

            "l2_j1": 0.0,      # [rad]
            "l2_j2": -0.7,      # [rad]
            "l2_j3": -0.7,      # [rad]
            "l2_wheel": 0.0,      # [rad]

            "l1_j1": 0.0,      # [rad]
            "l1_j2": 0.7,      # [rad]
            "l1_j3": 0.7,      # [rad]
            "l1_wheel": 0.0      # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'j1': 100., 'j2': 100., 'j3': 100.}  # [N*m/rad]
        damping = {'j1': 4., 'j2': 4., 'j3': 4.}     # [N*m*s/rad]

        # PD parameters for wheels
        stiffness_wheel = 100
        damping_wheel = 4

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = "/home/zetans/Desktop/rl_hybriped/assets/urdf/hybriped/urdf/hybriped_simplified_limits_cylinder_wheeled.urdf"
        name = "hybriped"
        foot_name = "wheel"
        penalize_contacts_on = ["link2", "link3"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        use_mesh_materials = False

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            pass

class WheeledHybripedRoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'hybriped'
        load_run = -1
