"""Contact sensor implementations for GenesisLab.

These sensors read real contact forces from the Genesis engine using the
`get_links_net_contact_force()` API and provide the same interface as
IsaacLab's contact sensors for MDP terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from genesislab.utils.configclass import configclass


@configclass
class ContactSensorCfg:
    """Configuration for a contact sensor.

    Attributes:
        name: Logical name of the sensor. If None, the key in SceneCfg.sensors
            is used.
        entity_name: Name of the articulated entity to which this sensor is
            conceptually attached (typically ``"robot"``).
        history_length: Number of past steps of contact forces to keep.
        track_air_time: Whether to track a simple notion of link "air time".
            Currently this is a no-op placeholder kept for API compatibility.
    """

    name: str | None = None
    entity_name: str = "robot"
    history_length: int = 3
    track_air_time: bool = True


@dataclass
class _ContactSensorData:
    """Data buffers exposed by :class:`ContactSensor`.

    For now, these are zero tensors with the right shapes. The fields mirror
    those used in the MDP reward/termination functions:

    - ``net_forces_w_history``: Tensor of shape
      ``(history_length, num_envs, num_channels, 3)``.
    - ``last_air_time``: Tensor of shape ``(num_envs, num_channels)``.
    """

    net_forces_w_history: torch.Tensor
    last_air_time: torch.Tensor


class ContactSensor:
    """Contact sensor for GenesisLab that reads real contact forces from Genesis engine.

    The sensor reads contact forces from the Genesis entity using
    `get_links_net_contact_force()` and maintains a history buffer compatible
    with IsaacLab's contact sensor interface.
    """

    def __init__(
        self,
        cfg: ContactSensorCfg,
        num_envs: int,
        device: str = "cuda",
        entity: Any = None,
    ):
        """Initialize the contact sensor.

        Args:
            cfg: Sensor configuration.
            num_envs: Number of parallel environments.
            device: Device for tensors.
            entity: Genesis entity object to read contact forces from. If None,
                will be set later via `set_entity()`.
        """
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device
        self._entity = entity

        # Get number of links/channels from entity if available
        # Otherwise, we'll infer it on first update
        if entity is not None and hasattr(entity, "n_links"):
            num_channels = entity.n_links
        else:
            # Default to 1 channel, will be updated on first update
            num_channels = 1

        history_len = max(int(cfg.history_length), 1)

        net_forces = torch.zeros(
            history_len,
            self.num_envs,
            num_channels,
            3,
            device=device,
            dtype=torch.float32,
        )
        last_air_time = torch.zeros(
            self.num_envs,
            num_channels,
            device=device,
            dtype=torch.float32,
        )
        self.data = _ContactSensorData(
            net_forces_w_history=net_forces,
            last_air_time=last_air_time,
        )
        # Store previous contact state for first-contact detection
        self._prev_air_time = last_air_time.clone()

    def set_entity(self, entity: Any) -> None:
        """Set the Genesis entity to read contact forces from.

        Args:
            entity: Genesis entity object.
        """
        self._entity = entity
        # Resize buffers if entity has different number of links
        if entity is not None and hasattr(entity, "n_links"):
            num_channels = entity.n_links
            if num_channels != self.data.net_forces_w_history.shape[2]:
                # Resize buffers to match number of links
                history_len = self.data.net_forces_w_history.shape[0]
                self.data.net_forces_w_history = torch.zeros(
                    history_len,
                    self.num_envs,
                    num_channels,
                    3,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.data.last_air_time = torch.zeros(
                    self.num_envs,
                    num_channels,
                    device=self.device,
                    dtype=torch.float32,
                )
                # Initialize previous air time
                self._prev_air_time = self.data.last_air_time.clone()

    def update(self, dt: float) -> None:
        """Update the sensor data for a simulation step.

        Args:
            dt: Physics time-step in seconds.
        """
        # Roll history along the time dimension
        self.data.net_forces_w_history = torch.roll(
            self.data.net_forces_w_history, shifts=-1, dims=0
        )

        # Get real contact forces from Genesis entity
        if self._entity is not None and hasattr(self._entity, "get_links_net_contact_force"):
            try:
                # Get contact forces: shape (num_envs, num_links, 3)
                contact_forces = self._entity.get_links_net_contact_force()
                
                # Ensure device and dtype match
                contact_forces = contact_forces.to(device=self.device, dtype=torch.float32)
                
                # Reshape to match expected format: (num_envs, num_links, 3) -> (num_envs, num_channels, 3)
                # If num_channels doesn't match, resize buffers
                num_links = contact_forces.shape[1]
                if num_links != self.data.net_forces_w_history.shape[2]:
                    # Resize buffers
                    history_len = self.data.net_forces_w_history.shape[0]
                    old_forces = self.data.net_forces_w_history.clone()
                    old_air_time = self.data.last_air_time.clone()
                    
                    self.data.net_forces_w_history = torch.zeros(
                        history_len,
                        self.num_envs,
                        num_links,
                        3,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    self.data.last_air_time = torch.zeros(
                        self.num_envs,
                        num_links,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    
                    # Copy old data if possible
                    if old_forces.shape[2] <= num_links:
                        self.data.net_forces_w_history[:, :, :old_forces.shape[2], :] = old_forces
                        self.data.last_air_time[:, :old_air_time.shape[1]] = old_air_time
                
                # Store latest contact forces in history (last time step)
                self.data.net_forces_w_history[-1, ...] = contact_forces
                
                # Update air time: reset to 0 if contact force magnitude > threshold, otherwise accumulate
                force_mag = torch.norm(contact_forces, dim=-1)  # (num_envs, num_links)
                contact_threshold = 1.0  # Threshold for considering a contact as "active"
                is_contact = force_mag > contact_threshold
                
                # Save previous air time for first-contact detection
                self._prev_air_time = self.data.last_air_time.clone()
                
                # Reset air time for links in contact, accumulate for others
                self.data.last_air_time = torch.where(
                    is_contact,
                    torch.zeros_like(self.data.last_air_time),
                    self.data.last_air_time + dt,
                )
            except Exception as e:
                # Fallback to zeros if API call fails
                self._prev_air_time = self.data.last_air_time.clone()
                self.data.net_forces_w_history[-1, ...] = 0.0
                self.data.last_air_time += dt
        else:
            # No entity available, keep zeros
            self._prev_air_time = self.data.last_air_time.clone()
            self.data.net_forces_w_history[-1, ...] = 0.0
            self.data.last_air_time += dt

    # IsaacLab-style helper used in feet_air_time.
    def compute_first_contact(self, step_dt: float) -> torch.Tensor:
        """Compute whether a first contact was detected at the current step.

        A "first contact" is detected when a link transitions from no contact
        (air_time > threshold) to contact (air_time == 0).

        Args:
            step_dt: Environment step time (used as threshold for detecting "was in air").

        Returns:
            A boolean tensor of shape ``(num_envs, num_channels)`` indicating
            whether a first contact was detected at the current step.
        """
        # Check if previous air_time was > threshold (was in air) and current air_time is 0 (now in contact)
        was_in_air = self._prev_air_time > step_dt
        is_now_in_contact = self.data.last_air_time < step_dt  # Air time reset to near-zero means contact
        
        # First contact: was in air AND now in contact
        first_contact = was_in_air & is_now_in_contact
        
        return first_contact

