"""
Deploy G1 12-DoF walking policy in Isaac Sim 4.5 (Isaac Lab).

Design goal: behave identically to deploy_mujoco/deploy_mujoco.py.

Strategy:
  - Use IdealPDActuator: Isaac Lab computes tau = kp*(q* - q) + kd*(0 - dq)
    in Python every time write_data_to_sim() is called, and applies it
    through PhysX's effort interface. This is mathematically identical to
    MuJoCo's pd_control()+d.ctrl[:]=tau (and to the real robot's motor
    cmd's q,kp,kd interface).
  - Every <decimation> sim steps (50 Hz) the policy refreshes the
    position target via set_joint_position_target(); write_data_to_sim()
    is still called every sim step (200 Hz) so PD recomputes torque
    from the freshly-updated joint state every physics tick.
  - Joint-order mapping between Isaac (BFS / left-right alternating) and
    training (left leg first, then right) is computed at runtime from
    Articulation.joint_names so it cannot silently desync from the asset.

Why NOT ImplicitActuator + manual set_joint_effort_target():
  PhysX's implicit drive does the PD inside the rigid-body solver step,
  which makes its effective dynamics differ from MuJoCo's explicit PD
  even with matched gains. And once you also try to set joint armature
  or friction through the actuator, the implicit path silently swallows
  the manually-set effort target -> robot free-falls. IdealPDActuator
  avoids both pitfalls.
"""

import argparse
import os
import sys
import numpy as np
import torch


# ---------------------------------------------------------------------------
# VRAM safety net (RTX 4070 8 GB is tight for Isaac Sim 4.5)
# ---------------------------------------------------------------------------
def _gpu_usage_ratio():
    if not torch.cuda.is_available():
        return 0.0
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    return reserved / total if total > 0 else 0.0


def _check_vram(threshold=0.92):
    ratio = _gpu_usage_ratio()
    if ratio > threshold:
        print(
            f"\n[VRAM GUARD] {ratio*100:.1f}% > {threshold*100:.1f}% -> abort to avoid freeze"
        )
        sys.exit(1)
    return ratio


# ---------------------------------------------------------------------------
# CLI + Isaac Lab AppLauncher (must run before importing isaaclab.sim etc.)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="G1 12-DoF policy in Isaac Sim 4.5")
parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                    help="velocity command [vx, vy, yaw_rate], default = forward 0.5 m/s")
parser.add_argument("--sim_dt", type=float, default=0.005,
                    help="physics dt (s). Use 0.002 with --decimation 10 to mimic MuJoCo")
parser.add_argument("--decimation", type=int, default=4,
                    help="number of sim steps per policy step (50 Hz when sim_dt=0.005)")
parser.add_argument("--stabilize_seconds", type=float, default=1.0,
                    help="seconds of PD-controlled standing before policy takes over")
parser.add_argument("--rebuild_usd", action="store_true",
                    help="force re-conversion of URDF -> USD (use after changing drive cfg)")
parser.add_argument("--ground_friction", type=float, default=1.0,
                    help="ground static/dynamic friction coefficient")
parser.add_argument("--joint_armature", type=float, default=0.0,
                    help="rotor inertia added to each joint axis (MJCF uses 0.01). "
                         "Default 0 -- enable cautiously: with kp/kd=100/2 and tiny link inertia, "
                         "armature changes the PD response time-constant.")
parser.add_argument("--joint_friction", type=float, default=0.0,
                    help="Coulomb dry friction torque per joint (MJCF uses 0.1). Default 0.")
parser.add_argument("--thick_feet", action="store_true",
                    help="enlarge the URDF's 4 corner foot spheres from 5 mm (matches MJCF) "
                         "to 12 mm so PhysX's default 2 cm contact_offset doesn't make them "
                         "buzz against the ground. Forces a USD rebuild.")
parser.add_argument("--foot_radius", type=float, default=0.012,
                    help="radius (m) of foot corner spheres when --thick_feet is used")
parser.add_argument("--print_obs_every", type=int, default=0,
                    help="if >0, dump the full 47-d observation every N policy steps "
                         "(useful for cross-checking against deploy_mujoco.py)")

# Make sure Isaac Lab source is importable (kept compatible with the user's env).
_isaaclab_src_candidates = [
    os.path.join(
        os.path.dirname(sys.executable), "..", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages", "isaaclab", "source", "isaaclab",
    ),
    os.path.join(
        os.path.dirname(sys.executable), "..", "lib", "python3.10", "site-packages",
        "isaaclab", "source", "isaaclab",
    ),
]
for _p in _isaaclab_src_candidates:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

# Now safe to import the rest of Isaac Lab.
from isaaclab.sim import SimulationContext, SimulationCfg                                 # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg                       # noqa: E402
from isaaclab.assets import ArticulationCfg, Articulation                                 # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg                                         # noqa: E402
import isaaclab.sim as sim_utils                                                          # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEGGED_GYM_ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
URDF_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/robots/g1_description/g1_12dof.urdf")
POLICY_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/pre_train/g1/motion.pt")

usd_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/robots/g1_description/usd")
os.makedirs(usd_dir, exist_ok=True)
usd_path = os.path.join(usd_dir, "g1_12dof.usd")


# ---------------------------------------------------------------------------
# Optional URDF rewrite: enlarge the 4 corner foot spheres.
#
# The MJCF and URDF both use 4 spheres of radius 0.005 m as the foot contact.
# In MuJoCo this works perfectly (point contact). In PhysX, the default
# contact_offset is 0.02 m, so a 5 mm sphere produces a tall, narrow "shell"
# of detection-but-no-penetration that is numerically twitchy and tends to
# flicker during the swing-to-stance transition - the textbook reason a
# walking policy starts strong and then nose-dives forward.
# ---------------------------------------------------------------------------
def make_thick_feet_urdf(src_urdf_path: str, sphere_radius: float) -> str:
    """Return path to a URDF where ankle_roll spheres are enlarged to `sphere_radius` m."""
    import re
    with open(src_urdf_path, "r", encoding="utf-8") as f:
        text = f.read()

    foot_corners = [
        (-0.05,  0.025, -0.03),
        (-0.05, -0.025, -0.03),
        ( 0.12,  0.030, -0.03),
        ( 0.12, -0.030, -0.03),
    ]
    sphere_blob = "".join(
        f'    <collision>\n'
        f'      <origin xyz="{x} {y} {z}" rpy="0 0 0"/>\n'
        f'      <geometry><sphere radius="{sphere_radius}"/></geometry>\n'
        f'    </collision>\n'
        for (x, y, z) in foot_corners
    )

    def _patch(link_name: str, src: str) -> str:
        # Replace ALL <collision>...</collision> entries inside the named link.
        link_pat = re.compile(
            rf'(<link name="{link_name}">)(.*?)(</link>)', re.DOTALL,
        )

        def _swap(m):
            head, body, tail = m.group(1), m.group(2), m.group(3)
            body = re.sub(r'<collision>.*?</collision>\s*', '', body, flags=re.DOTALL)
            return head + body + sphere_blob + tail
        return link_pat.sub(_swap, src)

    text = _patch("left_ankle_roll_link", text)
    text = _patch("right_ankle_roll_link", text)

    out_path = os.path.join(os.path.dirname(src_urdf_path), "g1_12dof_thickfeet.urdf")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


# ---------------------------------------------------------------------------
# URDF -> USD
#
# NOTE: stiffness = damping = 0 here so the USD's joint drive does NOT add
# any extra PD on top of our explicit Python PD. Pure torque control only.
# ---------------------------------------------------------------------------
if args.thick_feet:
    URDF_PATH = make_thick_feet_urdf(URDF_PATH, args.foot_radius)
    usd_path = os.path.join(usd_dir, "g1_12dof_thickfeet.usd")
    usd_file_name = "g1_12dof_thickfeet.usd"
    print(f"[URDF] using thick-feet URDF (radius {args.foot_radius*1000:.1f} mm): {URDF_PATH}")
else:
    usd_file_name = "g1_12dof.usd"

need_convert = args.rebuild_usd or args.thick_feet or not os.path.exists(usd_path)
if need_convert:
    print(f"[URDF->USD] converting {URDF_PATH}")
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=URDF_PATH,
            usd_dir=usd_dir,
            usd_file_name=usd_file_name,
            force_usd_conversion=True,
            fix_base=False,
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
        )
    )
    usd_path = converter.usd_path
    print(f"[URDF->USD] done -> {usd_path}")
else:
    print(f"[URDF->USD] reusing {usd_path}  (pass --rebuild_usd to redo)")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SIM_DT = float(args.sim_dt)
DECIMATION = int(args.decimation)
POLICY_DT = SIM_DT * DECIMATION  # = 0.02 s = 50 Hz when defaults are used

sim_cfg = SimulationCfg(
    dt=SIM_DT,
    render_interval=DECIMATION,
    device="cuda:0",
    use_fabric=True,
    enable_scene_query_support=False,
    gravity=(0.0, 0.0, -9.81),
    render=sim_utils.RenderCfg(
        antialiasing_mode="DLSS",
        dlss_mode=0,
        enable_shadows=False,
        enable_ambient_occlusion=False,
        enable_reflections=False,
        enable_global_illumination=False,
        enable_translucency=False,
        samples_per_pixel=1,
    ),
)
sim_ctx = SimulationContext(sim_cfg)

ground_cfg = sim_utils.GroundPlaneCfg(
    size=(100.0, 100.0),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=float(args.ground_friction),
        dynamic_friction=float(args.ground_friction),
        restitution=0.0,
    ),
)
ground_cfg.func("/World/ground", ground_cfg)

light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
light_cfg.func("/World/light", light_cfg)


# ---------------------------------------------------------------------------
# Robot
#
# Uses IdealPDActuator: Isaac Lab computes tau = kp*(q* - q) + kd*(0 - dq)
# in Python every write_data_to_sim() and applies it via PhysX's effort
# interface. Mathematically identical to MuJoCo's pd_control() and to the
# real robot's motor cmd <q, kp, kd>.
# ---------------------------------------------------------------------------
INITIAL_HEIGHT = 0.8


def _make_actuators(armature=None, friction=None):
    """One IdealPDActuator group per joint family. Per-joint kp/kd/effort_limit
    match deploy_mujoco/configs/g1.yaml and the URDF's <limit effort=...>."""
    spec = {
        "hip_pitch":   dict(pattern=".*hip_pitch_joint",   kp=100.0, kd=2.0, eff= 88.0),
        "hip_roll":    dict(pattern=".*hip_roll_joint",    kp=100.0, kd=2.0, eff=139.0),
        "hip_yaw":     dict(pattern=".*hip_yaw_joint",     kp=100.0, kd=2.0, eff= 88.0),
        "knee":        dict(pattern=".*knee_joint",        kp=150.0, kd=4.0, eff=139.0),
        "ankle_pitch": dict(pattern=".*ankle_pitch_joint", kp= 40.0, kd=2.0, eff= 50.0),
        "ankle_roll":  dict(pattern=".*ankle_roll_joint",  kp= 40.0, kd=2.0, eff= 50.0),
    }
    out = {}
    for name, s in spec.items():
        kwargs = dict(
            joint_names_expr=[s["pattern"]],
            stiffness=s["kp"],
            damping=s["kd"],
            effort_limit_sim=s["eff"],
        )
        if armature is not None:
            kwargs["armature"] = armature
        if friction is not None:
            kwargs["friction"] = friction
        out[name] = IdealPDActuatorCfg(**kwargs)
    return out

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
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, INITIAL_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_hip_pitch_joint":  -0.1,
            "left_hip_roll_joint":    0.0,
            "left_hip_yaw_joint":     0.0,
            "left_knee_joint":        0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint":  0.0,
            "right_hip_pitch_joint":  -0.1,
            "right_hip_roll_joint":    0.0,
            "right_hip_yaw_joint":     0.0,
            "right_knee_joint":        0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint":  0.0,
        },
    ),
    actuators=_make_actuators(
        armature=(float(args.joint_armature) if args.joint_armature > 0 else None),
        friction=(float(args.joint_friction) if args.joint_friction > 0 else None),
    ),
)
robot = Articulation(robot_cfg)

sim_ctx.reset()
robot.reset()
robot.update(SIM_DT)
device = robot.device


# ---------------------------------------------------------------------------
# Joint-order mapping  -- computed at runtime from joint_names so it cannot
# silently desync from the loaded asset (the previous bug).
# ---------------------------------------------------------------------------
TRAIN_JOINT_NAMES = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
isaac_joint_names = list(robot.joint_names)
print("[joint mapping] Isaac Sim joint order:")
for i, n in enumerate(isaac_joint_names):
    print(f"  isaac[{i:2d}] = {n}")

missing = [n for n in TRAIN_JOINT_NAMES if n not in isaac_joint_names]
if missing:
    raise RuntimeError(f"URDF/USD missing expected joints: {missing}")

# isaac_idx_for_train[t] = index in isaac order of the joint that has training-order index t.
# Use:  data_train  = data_isaac[:, isaac_idx_for_train]
isaac_idx_for_train = [isaac_joint_names.index(n) for n in TRAIN_JOINT_NAMES]
# Inverse mapping:  data_isaac = data_train[:, train_idx_for_isaac]
train_idx_for_isaac = [0] * 12
for t, i in enumerate(isaac_idx_for_train):
    train_idx_for_isaac[i] = t
print(f"[joint mapping] isaac_idx_for_train = {isaac_idx_for_train}")
print(f"[joint mapping] train_idx_for_isaac = {train_idx_for_isaac}")

isaac_idx_for_train_t = torch.tensor(isaac_idx_for_train, device=device, dtype=torch.long)
train_idx_for_isaac_t = torch.tensor(train_idx_for_isaac, device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Constants (same numbers as configs/g1.yaml and deploy_real)
# ---------------------------------------------------------------------------
default_angles_train = torch.tensor(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    device=device, dtype=torch.float32,
)
# Pre-permute to isaac order: every quantity we read from / write to PhysX
# is in isaac order; only obs construction switches to train order.
default_angles_isaac = default_angles_train[train_idx_for_isaac_t]

ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = torch.tensor([2.0, 2.0, 0.25], device=device, dtype=torch.float32)
CMD = torch.tensor([list(args.cmd)], device=device, dtype=torch.float32)
PHASE_PERIOD = 0.8
NUM_OBS = 47
NUM_ACT = 12


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
print("=" * 60)
print("INITIALIZATION")
print("=" * 60)

robot.write_joint_state_to_sim(
    position=default_angles_isaac.unsqueeze(0),
    velocity=torch.zeros((1, 12), device=device),
)
root_pose = torch.tensor(
    [[0.0, 0.0, INITIAL_HEIGHT, 1.0, 0.0, 0.0, 0.0]],
    device=device, dtype=torch.float32,
)
root_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
robot.write_root_pose_to_sim(root_pose)
robot.write_root_velocity_to_sim(root_vel)
robot.write_data_to_sim()
robot.update(SIM_DT)
print(f"[init] root pos = {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"[init] joint pos (isaac order) = {robot.data.joint_pos[0].cpu().numpy()}")
print(f"[init] projected gravity (body) = {robot.data.projected_gravity_b[0].cpu().numpy()}  "
      f"(should be ~[0, 0, -1])")


# ---------------------------------------------------------------------------
# Stabilization: hold default pose with PD (no policy yet)
#
# This phase uses the same code path the policy will use, so by the time
# the policy starts the controller dynamics are already settled.
# ---------------------------------------------------------------------------
stabilize_steps = max(0, int(round(args.stabilize_seconds / SIM_DT)))
print(f"[stabilize] {stabilize_steps} sim steps ({args.stabilize_seconds:.2f} s) PD-holding default pose…")
target_pos_isaac = default_angles_isaac.unsqueeze(0).clone()
robot.set_joint_position_target(target_pos_isaac)
for step in range(stabilize_steps):
    # IdealPDActuator recomputes tau from CURRENT joint state on every
    # write_data_to_sim() call, so a fresh PD output is applied each sim step
    # even though we only set the position target once.
    robot.write_data_to_sim()
    sim_ctx.step(render=(step % DECIMATION == 0))
    robot.update(SIM_DT)

    if robot.data.root_pos_w[0, 2] < 0.4:
        print(f"[stabilize][warn] robot at z={robot.data.root_pos_w[0,2].item():.2f} "
              f"after {(step+1)*SIM_DT:.2f}s -- continuing anyway")

print(f"[stabilize] done. root z = {robot.data.root_pos_w[0,2].item():.3f}")
print(f"[stabilize] joint pos err (max) = "
      f"{float((robot.data.joint_pos[0] - default_angles_isaac).abs().max()):.3f} rad")
print(f"[stabilize] projected gravity (body) = {robot.data.projected_gravity_b[0].cpu().numpy()}")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
print(f"[policy] loading {POLICY_PATH}")
policy = torch.jit.load(POLICY_PATH, map_location=device)
policy.eval()

obs_buf = torch.zeros((1, NUM_OBS), device=device, dtype=torch.float32)
last_action_train = torch.zeros((1, NUM_ACT), device=device, dtype=torch.float32)


def build_obs(policy_step_idx):
    """Mirrors the obs construction in deploy_mujoco.py exactly."""
    base_ang_vel_b = robot.data.root_ang_vel_b              # body-frame, matches MuJoCo qvel[3:6]
    projected_gravity = robot.data.projected_gravity_b       # body-frame, matches get_gravity_orientation(quat)
    dof_pos_isaac = robot.data.joint_pos
    dof_vel_isaac = robot.data.joint_vel

    dof_pos_train = dof_pos_isaac[:, isaac_idx_for_train_t]
    dof_vel_train = dof_vel_isaac[:, isaac_idx_for_train_t]

    sim_time = policy_step_idx * POLICY_DT
    phase = (sim_time % PHASE_PERIOD) / PHASE_PERIOD
    sin_phase = float(np.sin(2.0 * np.pi * phase))
    cos_phase = float(np.cos(2.0 * np.pi * phase))

    obs_buf[:, 0:3]   = base_ang_vel_b * ANG_VEL_SCALE
    obs_buf[:, 3:6]   = projected_gravity
    obs_buf[:, 6:9]   = CMD * CMD_SCALE
    obs_buf[:, 9:21]  = (dof_pos_train - default_angles_train) * DOF_POS_SCALE
    obs_buf[:, 21:33] = dof_vel_train * DOF_VEL_SCALE
    obs_buf[:, 33:45] = last_action_train
    obs_buf[:, 45]    = sin_phase
    obs_buf[:, 46]    = cos_phase
    return phase


# ---------------------------------------------------------------------------
# Main control loop
#
#   for each sim step (200 Hz):
#       if (step % decimation == 0):           # 50 Hz
#           obs <- read state (with isaac->train remap)
#           action_train <- policy(obs)
#           target_pos_train = default + action_train * 0.25
#           target_pos_isaac = train->isaac remap
#           set_joint_position_target(target_pos_isaac)
#       write_data_to_sim()           # IdealPD recomputes tau from current state
#       sim_ctx.step(); robot.update(SIM_DT)
#
# This is the same PD pipeline as MuJoCo and the real robot, routed
# through Isaac Lab's IdealPDActuator instead of an explicit Python call.
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"STARTING POLICY CONTROL  cmd = {args.cmd}  policy_dt = {POLICY_DT*1000:.1f} ms")
print("=" * 60)

sim_step_count = 0
policy_step_count = 0

try:
    while app_launcher.app.is_running():
        # ---- 50 Hz: refresh policy target ----
        if sim_step_count % DECIMATION == 0:
            phase = build_obs(policy_step_count)
            with torch.no_grad():
                action_train = policy(obs_buf)
            last_action_train = action_train.detach().clone()

            target_pos_train = action_train * ACTION_SCALE + default_angles_train
            target_pos_isaac = target_pos_train[:, train_idx_for_isaac_t]
            robot.set_joint_position_target(target_pos_isaac)

            policy_step_count += 1
            if policy_step_count % 50 == 0:
                root_pos = robot.data.root_pos_w[0].cpu().numpy()
                root_lin_b = robot.data.root_lin_vel_b[0].cpu().numpy()
                pos_err_max = float((robot.data.joint_pos[0] - target_pos_isaac[0]).abs().max())
                vram = _check_vram(0.90)
                print(
                    f"[step {policy_step_count:5d}] "
                    f"pos=({root_pos[0]:+.2f},{root_pos[1]:+.2f},{root_pos[2]:+.2f}) "
                    f"vel_b=({root_lin_b[0]:+.2f},{root_lin_b[1]:+.2f},{root_lin_b[2]:+.2f}) "
                    f"phase={phase:.2f} q_err_max={pos_err_max:.3f} "
                    f"VRAM={vram*100:.0f}%"
                )

            if args.print_obs_every and policy_step_count % args.print_obs_every == 0:
                obs_np = obs_buf[0].cpu().numpy()
                np.set_printoptions(precision=3, suppress=True, linewidth=200)
                print(f"  obs[ang_vel*0.25] = {obs_np[0:3]}")
                print(f"  obs[gravity_b   ] = {obs_np[3:6]}")
                print(f"  obs[cmd*scale   ] = {obs_np[6:9]}")
                print(f"  obs[(q-q*)*1.0  ] = {obs_np[9:21]}   (training order)")
                print(f"  obs[dq*0.05     ] = {obs_np[21:33]}  (training order)")
                print(f"  obs[prev_action ] = {obs_np[33:45]}  (training order)")
                print(f"  obs[sin,cos     ] = {obs_np[45:47]}")

        # ---- 200 Hz: IdealPD recomputes tau internally from current joint state ----
        robot.write_data_to_sim()

        sim_ctx.step(render=(sim_step_count % DECIMATION == 0))
        robot.update(SIM_DT)
        sim_step_count += 1

        # Crash detector: avoid the "lying-on-the-ground but still spinning" failure mode.
        if robot.data.root_pos_w[0, 2] < 0.25:
            print(f"[main] robot fell (z={robot.data.root_pos_w[0,2].item():.2f}). Exiting.")
            break

except KeyboardInterrupt:
    print("\n[main] interrupted by user")
finally:
    print("[main] closing app")
    app_launcher.app.close()
