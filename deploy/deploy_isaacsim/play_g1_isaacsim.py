import argparse
import os
import sys
import signal
import numpy as np
import torch


def get_gpu_memory_info():
    """Return (allocated_mb, reserved_mb, total_mb) for current GPU."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return allocated, reserved, total
    return 0.0, 0.0, 0.0


def check_vram_and_exit_if_needed(threshold_ratio=0.92):
    """Exit process gracefully if VRAM usage exceeds threshold."""
    allocated, reserved, total = get_gpu_memory_info()
    usage_ratio = reserved / total if total > 0 else 0
    if usage_ratio > threshold_ratio:
        print(f"\n[VRAM GUARD] CRITICAL: VRAM {reserved:.0f}/{total:.0f} MB "
              f"({usage_ratio*100:.1f}%) exceeds threshold {threshold_ratio*100:.1f}%")
        print("[VRAM GUARD] Emergency shutdown to prevent system freeze...")
        sys.exit(1)
    return usage_ratio


# ------------------------------------------------------------------
# 1. Environment Setup
# ------------------------------------------------------------------
isaaclab_src = os.path.join(
    os.path.dirname(sys.executable), "..", "lib", "python3.10", "site-packages",
    "isaaclab", "source", "isaaclab"
)
if os.path.isdir(isaaclab_src) and isaaclab_src not in sys.path:
    sys.path.insert(0, isaaclab_src)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Robot with motion.pt policy in Isaac Sim")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
URDF_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/robots/g1_description/g1_12dof.urdf")
POLICY_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/pre_train/g1/motion.pt")

# ------------------------------------------------------------------
# 2. URDF -> USD Conversion
# ------------------------------------------------------------------
usd_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/robots/g1_description/usd")
os.makedirs(usd_dir, exist_ok=True)
usd_path = os.path.join(usd_dir, "g1_12dof.usd")

if not os.path.exists(usd_path):
    urdf_converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=URDF_PATH,
            usd_dir=usd_dir,
            usd_file_name="g1_12dof.usd",
            force_usd_conversion=True,
            fix_base=False,
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=200.0,
                    damping=5.0,
                ),
            ),
        )
    )
    usd_path = urdf_converter.usd_path
    print(f"URDF converted to USD: {usd_path}")

# ------------------------------------------------------------------
# 3. Simulation Config
# ------------------------------------------------------------------
sim_cfg = SimulationCfg(
    dt=0.005,
    render_interval=4,
    device="cuda:0",
    use_fabric=True,
    enable_scene_query_support=False,
    gravity=(0.0, 0.0, -9.81),
    render=sim_utils.RenderCfg(
        antialiasing_mode="DLSS",
        dlss_mode=0,
        enable_shadows=True,
        enable_ambient_occlusion=False,
        enable_reflections=False,
        enable_global_illumination=False,
        enable_translucency=False,
        samples_per_pixel=1,
    ),
)
sim_ctx = SimulationContext(sim_cfg)

# Ground plane at z=0
ground_cfg = sim_utils.GroundPlaneCfg(
    size=(100.0, 100.0),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
)
ground_cfg.func("/World/ground", ground_cfg)

# Light
light_cfg = sim_utils.DomeLightCfg(
    intensity=3000.0,
    color=(0.75, 0.75, 0.75),
)
light_cfg.func("/World/light", light_cfg)

# ------------------------------------------------------------------
# 4. Robot Config
# ------------------------------------------------------------------
INITIAL_HEIGHT = 0.8

robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, INITIAL_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
        },
    ),
    actuators={
        "hip_pitch": ImplicitActuatorCfg(joint_names_expr=["hip_pitch"], stiffness=100.0, damping=2.0),
        "hip_roll": ImplicitActuatorCfg(joint_names_expr=["hip_roll"], stiffness=100.0, damping=2.0),
        "hip_yaw": ImplicitActuatorCfg(joint_names_expr=["hip_yaw"], stiffness=100.0, damping=2.0),
        "knee": ImplicitActuatorCfg(joint_names_expr=["knee"], stiffness=150.0, damping=4.0),
        "ankle_pitch": ImplicitActuatorCfg(joint_names_expr=["ankle_pitch"], stiffness=40.0, damping=2.0),
        "ankle_roll": ImplicitActuatorCfg(joint_names_expr=["ankle_roll"], stiffness=40.0, damping=2.0),
    },
)
robot = Articulation(robot_cfg)

sim_ctx.reset()
robot.reset()
robot.update(sim_cfg.dt)

# Isaac Sim joint order (left/right alternating):
# ['left_hip_pitch', 'right_hip_pitch', 'left_hip_roll', 'right_hip_roll',
#  'left_hip_yaw', 'right_hip_yaw', 'left_knee', 'right_knee',
#  'left_ankle_pitch', 'right_ankle_pitch', 'left_ankle_roll', 'right_ankle_roll']
# Training/MuJoCo joint order (all left then all right):
# ['left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
#  'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll']
isaac_to_train = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
train_to_isaac = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]

default_angles = torch.tensor(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    device=robot.device, dtype=torch.float32,
)
default_angles_isaac = default_angles[train_to_isaac]

# ------------------------------------------------------------------
# 5. Initialization & Stabilization
# ------------------------------------------------------------------
print("=" * 60)
print("INITIALIZATION")
print("=" * 60)

# Print initial state for debugging
print(f"Initial root pos: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"Initial joint pos (before set): {robot.data.joint_pos[0].cpu().numpy()}")

robot._data.joint_pos[0] = default_angles_isaac.clone()
robot._data.joint_vel[0] = torch.zeros(12, device=robot.device)
robot.write_data_to_sim()

print(f"After direct set: {robot.data.joint_pos[0].cpu().numpy()}")

# Step once to apply
sim_ctx.step(render=True)
robot.update(sim_cfg.dt)

print(f"After step: {robot.data.joint_pos[0].cpu().numpy()}")
print(f"After step root vel: {robot.data.root_vel_w[0].cpu().numpy()}")

# CRITICAL: sim_ctx.step() overwrites joint positions on first step!
# Re-set immediately and freeze root to stabilize
robot._data.joint_pos[0] = default_angles_isaac.clone()
robot._data.joint_vel[0] = torch.zeros(12, device=robot.device)
robot.write_data_to_sim()

# Stabilize: hold default position for 2 seconds (400 steps @ 0.005s)
print("Stabilizing robot (holding default pose for 2 seconds)...")
for step in range(400):
    root_pose = torch.tensor([[0.0, 0.0, INITIAL_HEIGHT, 1.0, 0.0, 0.0, 0.0]], device=robot.device)
    robot.write_root_pose_to_sim(root_pose)
    robot.set_joint_position_target(default_angles_isaac.unsqueeze(0))
    robot.write_data_to_sim()
    
    sim_ctx.step(render=True)
    robot.update(sim_cfg.dt)
    
    if step == 200:
        print(f"  Mid-stabilization (1s): Root pos = {robot.data.root_pos_w[0].cpu().numpy()}")

print(f"Root pos after stabilization: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"Joint pos after stabilization: {robot.data.joint_pos[0].cpu().numpy()}")
print(f"Projected gravity: {robot.data.projected_gravity_b[0].cpu().numpy()}")

# Reset root state to upright before starting policy
print("Resetting root state to upright...")
root_state = torch.tensor([[0.0, 0.0, INITIAL_HEIGHT, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=robot.device)
robot.write_root_com_state_to_sim(root_state)
robot.write_data_to_sim()
sim_ctx.step(render=True)
robot.update(sim_cfg.dt)
print(f"Root pos after reset: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"Projected gravity after reset: {robot.data.projected_gravity_b[0].cpu().numpy()}")

# ------------------------------------------------------------------
# 6. Load Policy
# ------------------------------------------------------------------
print("Loading policy...")
policy = torch.jit.load(POLICY_PATH, map_location=robot.device)
policy.eval()

# Observation parameters (must match training)
obs_scales_ang_vel = 0.25
obs_scales_dof_pos = 1.0
obs_scales_dof_vel = 0.05
cmd_scale = torch.tensor([2.0, 2.0, 0.25], device=robot.device, dtype=torch.float32)
action_scale = 0.25

cmd = torch.tensor([[0.5, 0.0, 0.0]], device=robot.device, dtype=torch.float32)

num_actions = 12
num_obs = 47
obs_buf = torch.zeros(1, num_obs, device=robot.device, dtype=torch.float32)
actions = torch.zeros(1, num_actions, device=robot.device, dtype=torch.float32)

step_count = 0
decimation = 4

print("=" * 60)
print("STARTING POLICY CONTROL")
print("=" * 60)

# ------------------------------------------------------------------
# 7. Main Loop
# ------------------------------------------------------------------
while True:
    for _ in range(decimation):
        sim_ctx.step(render=True)

    robot.update(sim_cfg.dt * decimation)
    step_count += 1

    root_ang_vel_b = robot.data.root_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b
    dof_pos = robot.data.joint_pos
    dof_vel = robot.data.joint_vel

    # Phase signal (periodic gait)
    period = 0.8
    sim_time = step_count * sim_cfg.dt * decimation
    phase = (sim_time % period) / period
    sin_phase = torch.tensor(
        [[np.sin(2 * np.pi * phase)]], device=robot.device, dtype=torch.float32
    )
    cos_phase = torch.tensor(
        [[np.cos(2 * np.pi * phase)]], device=robot.device, dtype=torch.float32
    )

    # Remap from Isaac Sim order to training order
    dof_pos_train = dof_pos[:, isaac_to_train]
    dof_vel_train = dof_vel[:, isaac_to_train]

    # Construct observation (must match training exactly)
    obs_buf[:, 0:3] = root_ang_vel_b * obs_scales_ang_vel
    obs_buf[:, 3:6] = projected_gravity
    obs_buf[:, 6:9] = cmd * cmd_scale
    obs_buf[:, 9 : 9 + num_actions] = (dof_pos_train - default_angles) * obs_scales_dof_pos
    obs_buf[:, 9 + num_actions : 9 + 2 * num_actions] = dof_vel_train * obs_scales_dof_vel
    obs_buf[:, 9 + 2 * num_actions : 9 + 3 * num_actions] = actions
    obs_buf[:, 9 + 3 * num_actions : 9 + 3 * num_actions + 1] = sin_phase
    obs_buf[:, 9 + 3 * num_actions + 1 : 9 + 3 * num_actions + 2] = cos_phase

    with torch.no_grad():
        actions = policy(obs_buf)

    target_dof_pos = actions * action_scale + default_angles
    target_dof_pos_isaac = target_dof_pos[:, train_to_isaac]
    robot.set_joint_position_target(target_dof_pos_isaac)
    robot.write_data_to_sim()

    if step_count % 50 == 0:
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        usage = check_vram_and_exit_if_needed(threshold_ratio=0.90)
        print(f"Step {step_count} | Pos: [{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}] | VRAM: {usage*100:.1f}%")
        print(f"  obs[0:3] (ang_vel): {obs_buf[0, 0:3].cpu().numpy()}")
        print(f"  obs[3:6] (gravity): {obs_buf[0, 3:6].cpu().numpy()}")
        print(f"  obs[6:9] (cmd): {obs_buf[0, 6:9].cpu().numpy()}")
        print(f"  obs[9:21] (dof_pos): {obs_buf[0, 9:21].cpu().numpy()}")
        print(f"  obs[21:33] (dof_vel): {obs_buf[0, 21:33].cpu().numpy()}")
        print(f"  obs[33:45] (actions): {obs_buf[0, 33:45].cpu().numpy()}")
        print(f"  obs[45:47] (phase): {obs_buf[0, 45:47].cpu().numpy()}")

app_launcher.app.close()
