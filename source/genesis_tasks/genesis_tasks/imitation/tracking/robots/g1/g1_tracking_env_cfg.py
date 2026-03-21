"""Robot-specific tracking env configs (e.g., G1 BeyondMimic)."""

from genesislab.utils.configclass import configclass

from genesis_assets.robots.g1.official import G1_FULL_ACT_CFG

from genesis_tasks.imitation.tracking.tracking_env_cfg import TrackingEnvCfg


# Match the IsaacLab reference LOW_FREQ_SCALE used for low-frequency variants.
LOW_FREQ_SCALE: int = 4


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    """G1 BeyondMimic whole-body tracking on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        # Attach G1 BeyondMimic robot asset with a single \"full\" actuator group.
        self.scene.robots["robot"] = G1_FULL_ACT_CFG

        # Joint position action scaling: align with velocity G1 config.
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.actuator_name = "full"

        # Motion command anchor and body list: mirror IsaacLab G1 tracking config.
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    """G1 tracking without state-estimation-related observations (policy sees less)."""

    def __post_init__(self):
        super().__post_init__()
        # Drop motion anchor position and base linear velocity from policy obs,
        # matching the IsaacLab Wo-State-Estimation variant.
        if hasattr(self.observations, "policy"):
            self.observations.policy.motion_anchor_pos_b = None
            self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    """G1 tracking at lower control frequency (higher decimation)."""

    def __post_init__(self):
        super().__post_init__()
        # Reduce control frequency and scale the action-rate penalty accordingly.
        if getattr(self, "decimation", None) is not None:
            self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        if getattr(self, "rewards", None) is not None and hasattr(self.rewards, "action_rate_l2"):
            self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE

