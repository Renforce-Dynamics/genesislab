from __future__ import annotations

from genesislab.utils.configclass import configclass

from ..velocity_env_cfg import HumanVelocityEnvCfg

try:
    # Optional: this asset does not exist in the current repo by default.
    from robotlib.soccerLab.mosc9 import MOSC9_CFG  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MOSC9_CFG = None


@configclass
class MOSCFlatEnvCfg(HumanVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        if MOSC9_CFG is not None:
            self.scene.robots["robot"] = MOSC9_CFG

        # Keep source-task semantics where MOS9 removes arm/waist deviation style terms.
        if hasattr(self.rewards, "joint_deviation_arms"):
            self.rewards.joint_deviation_arms = None
        if hasattr(self.rewards, "joint_deviation_waists"):
            self.rewards.joint_deviation_waists = None
