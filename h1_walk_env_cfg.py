"""H1 flat-ground walking environment.

Scaffolding only — the observation, reward and termination configs are
intentionally empty placeholders to be filled in per-experiment.
"""

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

# Biped-specific locomotion terms that the core mdp module does not carry:
#   feet_air_time_positive_biped, feet_slide, track_lin_vel_xy_yaw_frame_exp,
#   track_ang_vel_z_world_exp, stand_still_joint_deviation_l1
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp  # noqa: F401

from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip
import walk_curriculums as custom_curriculums  # noqa: F401  isort: skip
import walk_observations as custom_observations  # noqa: F401  isort: skip
import walk_rewards as custom_rewards  # noqa: F401  isort: skip

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class H1WalkSceneCfg(InteractiveSceneCfg):
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

    # Track contact on every body. Unlike the balance task, track_air_time is on
    # so gait-timing terms such as feet_air_time have data to read.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@configclass
class H1WalkCommandsCfg:
    # Forward-only velocity command: no lateral motion, no turning yet.
    # Widen these ranges once straight-line walking holds.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # new command twice per 20s episode
        rel_standing_envs=0.0,  # fraction of envs commanded to stand still
        heading_command=False,  # sample ang_vel_z directly instead of a heading
        debug_vis=True,  # green arrow = commanded, blue arrow = actual
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class H1WalkActionsCfg:
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
class H1WalkObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ------------------ Observation terms go here ----------------------
        #
        # NOTE: the policy group is empty, so the env builds an observation
        # vector of width 0 and training will fail until a term is added.
        #
        # Shape of a term:
        #   base_lin_vel = ObsTerm(
        #       func=mdp.base_lin_vel,
        #       noise=Unoise(n_min=-0.1, n_max=0.1),
        #   )
        #   velocity_commands = ObsTerm(
        #       func=mdp.generated_commands,
        #       params={"command_name": "base_velocity"},
        #   )

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
class H1WalkEventCfg:
    # Reset the root pose with a random yaw; start from rest
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

    # -------------------- Event Scrapyard ----------------------------------

    # Periodic shove to test gait robustness — enable once a gait exists.
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class H1WalkRewardsCfg:
    # ---------------------- Reward terms go here ---------------------------
    #
    # NOTE: empty, so every step returns a reward of 0 until terms are added.
    #
    # Shape of a term:
    #   track_lin_vel_xy = RewTerm(
    #       func=loco_mdp.track_lin_vel_xy_yaw_frame_exp,
    #       weight=1.0,
    #       params={"command_name": "base_velocity", "std": 0.5},
    #   )
    pass


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


# @configclass
# class H1WalkCurriculumCfg:
#     command_ranges = CurrTerm(
#         func=custom_curriculums.<term_name>,
#         params={},
#     )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class H1WalkTerminationsCfg:
    # -------------------- Termination terms go here ------------------------
    #
    # NOTE: empty, so episodes never end and no reset ever fires. At minimum
    # this needs time_out, plus a fall condition.
    #
    # Shape of a term:
    #   time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #   torso_contact = DoneTerm(
    #       func=mdp.illegal_contact,
    #       params={
    #           "sensor_cfg": SceneEntityCfg(
    #               "contact_forces", body_names=".*torso_link"
    #           ),
    #           "threshold": 1.0,
    #       },
    #   )
    pass


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------


@configclass
class H1WalkEnvCfg(ManagerBasedRLEnvCfg):
    scene: H1WalkSceneCfg = H1WalkSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: H1WalkObservationsCfg = H1WalkObservationsCfg()
    actions: H1WalkActionsCfg = H1WalkActionsCfg()
    commands: H1WalkCommandsCfg = H1WalkCommandsCfg()
    rewards: H1WalkRewardsCfg = H1WalkRewardsCfg()
    terminations: H1WalkTerminationsCfg = H1WalkTerminationsCfg()
    events: H1WalkEventCfg = H1WalkEventCfg()
    # curriculum: H1WalkCurriculumCfg = H1WalkCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4  # policy acts every 4 physics steps (50hz) (200hz / 4)
        self.episode_length_s = 20.0  # 20-second episodes (1000 steps (20s x 50hz))
        self.sim.dt = (
            0.005  # delta time. 200hz physics (1 physics step every 0.005s (1/0.005))
        )
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            # Air-time integrates per physics step, so the contact sensor ticks
            # at sim.dt rather than at the policy period.
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class H1WalkEnvCfg_PLAY(H1WalkEnvCfg):
    # Config for play: viewing checkpoints with Isaac Sim
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = True
        # Fixed forward command so every env on screen is doing the same thing
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class H1WalkEnvCfg_FOCUS_PLAY(H1WalkEnvCfg):
    # Config for focus play: 1 env with live joint torque and reward accumulation visualization panel
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = True
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
