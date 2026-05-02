import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    # Joint names for debug output
    joint_names = [m.joint(i).name for i in range(m.njnt) if m.joint(i).type == 3]  # type 3 = hinge/slide joints

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                control_step = counter // control_decimation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                root_pos = d.qpos[:3]
                root_vel = d.qvel[:3]

                qj_scaled = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega_scaled = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega_scaled
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj_scaled
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj_scaled
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

                # Print debug info every 50 control steps (1 second)
                if control_step % 50 == 0:
                    print(f"\n{'='*70}")
                    print(f"CONTROL STEP {control_step} | Sim Time: {count:.2f}s")
                    print(f"{'='*70}")
                    print(f"Root Pos: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")
                    print(f"Root Vel: [{root_vel[0]:.3f}, {root_vel[1]:.3f}, {root_vel[2]:.3f}]")
                    print(f"Gravity (body): [{gravity_orientation[0]:.4f}, {gravity_orientation[1]:.4f}, {gravity_orientation[2]:.4f}]")
                    print(f"Ang Vel (body): [{omega[0]:.4f}, {omega[1]:.4f}, {omega[2]:.4f}]")
                    print(f"CMD: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]")
                    print(f"Phase: sin={sin_phase:.4f}, cos={cos_phase:.4f}")
                    
                    print("\nJoint Positions (raw):")
                    for i in range(num_actions):
                        print(f"  {i:2d}: {joint_names[i]:30s} pos={qj[i]:8.4f}  "
                              f"default={default_angles[i]:6.2f}  "
                              f"err={qj[i]-default_angles[i]:8.4f}  "
                              f"vel={dqj[i]:8.4f}")
                    
                    print("\nObservations (scaled):")
                    print(f"  omega_scaled:     [{obs[0]:.4f}, {obs[1]:.4f}, {obs[2]:.4f}]")
                    print(f"  gravity_orient:   [{obs[3]:.4f}, {obs[4]:.4f}, {obs[5]:.4f}]")
                    print(f"  cmd_scaled:       [{obs[6]:.4f}, {obs[7]:.4f}, {obs[8]:.4f}]")
                    print(f"  qj_scaled:        {obs[9:9+num_actions]}")
                    print(f"  dqj_scaled:       {obs[9+num_actions:9+2*num_actions]}")
                    print(f"  action_prev:      {obs[9+2*num_actions:9+3*num_actions]}")
                    print(f"  phase:            [{obs[9+3*num_actions]:.4f}, {obs[9+3*num_actions+1]:.4f}]")
                    
                    print(f"\nPolicy Output (action): {action}")
                    print(f"Target DOF Pos: {target_dof_pos}")
                    print(f"Torques: {tau}")

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
