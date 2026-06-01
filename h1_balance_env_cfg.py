"""H1 standing-balance environment."""

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class H1BalanceSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Track contact on every body — termination reads from torso_link
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class H1BalanceActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class H1BalanceObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Tilt / orientation signal (vestibular-like)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Rotational velocity
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        # Proprioception
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # Previous action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True  # Apply noise to observations
            self.concatenate_terms = (
                True  # Concatenate all terms into a single observation vector
            )

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events (resets / randomisation)
# ---------------------------------------------------------------------------


@configclass
class H1BalanceEventCfg:
    # Reset the root pose with a slight yaw perturbation; start from rest
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset joints to their default positions (no randomisation yet)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class H1BalanceRewardsCfg:
    # -1 when the episode terminates due to a fall
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)

    # Penalize ankle deviation from neutral — encourages active ankle corrections for balance
    ankle_correction = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle"]),
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class H1BalanceTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Episode ends when the torso contacts the ground
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
            "threshold": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------


@configclass
class H1BalanceEnvCfg(ManagerBasedRLEnvCfg):
    scene: H1BalanceSceneCfg = H1BalanceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: H1BalanceObservationsCfg = H1BalanceObservationsCfg()
    actions: H1BalanceActionsCfg = H1BalanceActionsCfg()
    rewards: H1BalanceRewardsCfg = H1BalanceRewardsCfg()
    terminations: H1BalanceTerminationsCfg = H1BalanceTerminationsCfg()
    events: H1BalanceEventCfg = H1BalanceEventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.decimation * self.sim.dt


@configclass
class H1BalanceEnvCfg_PLAY(H1BalanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
