from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List, Dict, Any, Self

from .degree_of_freedom import DegreeOfFreedom
from .world_entity import WorldEntity
from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
)


@dataclass(eq=False)
class Actuator(WorldEntity, SubclassJSONSerializer):
    """
    Represents an actuator in the world model.
    """

    _dofs: List[DegreeOfFreedom] = field(default_factory=list, init=False, repr=False)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["name"] = self.name.to_json()
        result["dofs"] = [dof.to_json() for dof in self._dofs]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Actuator:
        actuator = cls(
            name=WorldEntity.name.from_json(data["name"]),
        )
        dofs_data = data.get("dofs", [])
        assert (
            len(dofs_data) > 0
        ), "An actuator must have at least one degree of freedom."
        for dof_data in dofs_data:
            dof = DegreeOfFreedom.from_json(dof_data)
            actuator.add_dof(dof)
        return actuator

    @property
    def dofs(self) -> List[DegreeOfFreedom]:
        """
        Returns a copy of the list of degrees of freedom associated with this actuator.
        """
        return self._dofs[:]

    def add_dof(self, dof: DegreeOfFreedom) -> None:
        """
        Adds a degree of freedom to this actuator.

        :param dof: The degree of freedom to add.
        """
        self._dofs.append(dof)


import mujoco


@dataclass(eq=False)
class MujocoActuator(Actuator):
    """
    Represents a MuJoCo-specific actuator in the world model.
    For more information, see: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general
    """

    activation_limited: mujoco.mjtLimited = mujoco.mjtLimited.mjLIMITED_AUTO
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the internal state (activation) associated with this actuator is automatically clamped to actrange at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, activation clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, activation clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if activation_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE. 
    """

    activation_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    Range for clamping the activation state. The first value must be no greater than the second value.
    """

    ctrl_limited: mujoco.mjtLimited = mujoco.mjtLimited.mjLIMITED_AUTO
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the control input to this actuator is automatically clamped to ctrl_range at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, control input clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, control clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if ctrl_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE.
    """

    ctrl_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    The range of the control input.
    """

    force_limited: mujoco.mjtLimited = mujoco.mjtLimited.mjLIMITED_AUTO
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the force output of this actuator is automatically clamped to force_range at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, force clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, force clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if force_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE.
    """

    force_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    Range for clamping the force output. The first value must be no greater than the second value.
    """

    bias_parameters: List[float] = field(default_factory=lambda: [0.0] * 10)
    """
    Bias parameters. The affine bias type uses three parameters.
    """

    bias_type: mujoco.mjtBias = mujoco.mjtBias.mjBIAS_NONE
    """
    The keywords have the following meaning:
    mujoco.mjtBias.mjBIAS_NONE:     bias_term = 0
    mujoco.mjtBias.mjBIAS_AFFINE:   bias_term = biasprm[0] + biasprm[1]*length + biasprm[2]*velocity
    mujoco.mjtBias.mjBIAS_MUSCLE:   bias_term = mju_muscleBias(…)
    mujoco.mjtBias.mjBIAS_USER:     bias_term = mjcb_act_bias(…)
    """

    dynamics_parameters: List[float] = field(default_factory=lambda: [1.0] + [0.0] * 9)
    """
    Activation dynamics parameters.
    """

    dynamics_type: mujoco.mjtDyn = mujoco.mjtDyn.mjDYN_NONE
    """
    Activation dynamics type for the actuator.
    The keywords have the following meaning:
    mujoco.mjtDyn.mjDYN_NONE:           No internal state
    mujoco.mjtDyn.mjDYN_INTEGRATOR:     act_dot = ctrl
    mujoco.mjtDyn.mjDYN_FILTER:         act_dot = (ctrl - act) / dynprm[0]
    mujoco.mjtDyn.mjDYN_FILTEREXACT:    Like filter but with exact integration
    mujoco.mjtDyn.mjDYN_MUSCLE:         act_dot = mju_muscleDynamics(…)
    mujoco.mjtDyn.mjDYN_USER:           act_dot = mjcb_act_dyn(…)
    """

    gain_parameters: List[float] = field(default_factory=lambda: [0.0] * 10)
    """
    Gain parameters.
    """

    gain_type: mujoco.mjtGain = mujoco.mjtGain.mjGAIN_FIXED
    """
    The gain and bias together determine the output of the force generation mechanism, which is currently assumed to be affine.
    The keywords have the following meaning:
    mujoco.mjtGain.mjGAIN_FIXED:    gain_term = gainprm[0]
    mujoco.mjtGain.mjGAIN_AFFINE:   gain_term = gain_prm[0] + gain_prm[1]*length + gain_prm[2]*velocity
    mujoco.mjtGain.mjGAIN_MUSCLE:   gain_term = mju_muscleGain(…)
    mujoco.mjtGain.mjGAIN_USER:     gain_term = mjcb_act_gain(…)
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["activation_limited"] = self.activation_limited.value
        result["activation_range"] = self.activation_range
        result["ctrl_limited"] = self.ctrl_limited.value
        result["ctrl_range"] = self.ctrl_range
        result["force_limited"] = self.force_limited.value
        result["force_range"] = self.force_range
        result["bias_parameters"] = self.bias_parameters
        result["bias_type"] = self.bias_type.value
        result["dynamics_parameters"] = self.dynamics_parameters
        result["dynamics_type"] = self.dynamics_type.value
        result["gain_parameters"] = self.gain_parameters
        result["gain_type"] = self.gain_type.value
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        actuator = super()._from_json(data, **kwargs)
        actuator.activation_limited = mujoco.mjtLimited(data["activation_limited"])
        actuator.activation_range = data["activation_range"]
        actuator.ctrl_limited = mujoco.mjtLimited(data["ctrl_limited"])
        actuator.ctrl_range = data["ctrl_range"]
        actuator.force_limited = mujoco.mjtLimited(data["force_limited"])
        actuator.force_range = data["force_range"]
        actuator.bias_parameters = data["bias_parameters"]
        actuator.bias_type = mujoco.mjtBias(data["bias_type"])
        actuator.dynamics_parameters = data["dynamics_parameters"]
        actuator.dynamics_type = mujoco.mjtDyn(data["dynamics_type"])
        actuator.gain_parameters = data["gain_parameters"]
        actuator.gain_type = mujoco.mjtGain(data["gain_type"])
        return actuator
