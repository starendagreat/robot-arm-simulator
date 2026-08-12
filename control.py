from math import *

dt = 0.01

MAX_TORQUE = 20.0
GRAVITY = 9.8            # must match the gravity used in physics.update_arm

# ─────────────────────────────────────────────
#  BASE CLASS — do not edit this
# ─────────────────────────────────────────────

class ArmController:
    """
    Base class for all arm controllers.
    Subclass this and override update() to implement your control logic.
    """
    def __init__(self, arm_state):
        self.arm_state = arm_state
        self.enabled = False
        self.t = 0.0              # time since controller was enabled

    def enable(self):
        self.enabled = True
        self.t = 0.0
        self.on_enable()

    def disable(self):
        self.enabled = False
        self.on_disable()

    def step(self):
        """Called every simulation frame. Do not override this."""
        if not self.enabled:
            return
        torque = self.update(self.arm_state, self.t)
        self._apply_torque(torque)
        self.t += dt

    def _apply_torque(self, torque):
        """Converts torque to angular velocity change and applies it."""
        torque = max(-MAX_TORQUE, min(MAX_TORQUE, torque))
        L = self.arm_state["length"]
        mass = self.arm_state["mass"]
        I = (1 / 3) * mass * L ** 2          # moment of inertia
        angular_acc = torque / I
        self.arm_state["angular_velocity"] += angular_acc * dt

    # ── Override these in your subclass ──────

    def on_enable(self):
        """Called once when the controller is enabled. Use for setup/reset."""
        pass

    def on_disable(self):
        """Called once when the controller is disabled."""
        pass

    def update(self, arm_state, t):
        """
        Called every frame. Return a torque (float) to apply to the arm.

        arm_state keys:
            "angle"            — current angle in radians (0 = horizontal right)
            "angular_velocity" — current angular velocity in rad/s
            "length"           — arm length
            "mass"             — arm mass
            "anchor"           — anchor position (VPython vector)

        t — time in seconds since this controller was enabled
        """
        raise NotImplementedError("override update() in your subclass")


# ─────────────────────────────────────────────
#  CONTROLLER MANAGER — do not edit this
# ─────────────────────────────────────────────

class ControllerManager:
    """
    Holds a single active controller at a time.
    Swap controllers at runtime via set_controller().
    """
    def __init__(self):
        self._active = None

    def set_controller(self, controller):
        if self._active is not None:
            self._active.disable()
        self._active = controller
        self._active.enable()

    def clear(self):
        if self._active is not None:
            self._active.disable()
        self._active = None

    def step(self):
        if self._active is not None:
            self._active.step()

    @property
    def active(self):
        return self._active


# ─────────────────────────────────────────────
#  YOUR CONTROLLERS — add new ones below here
# ─────────────────────────────────────────────

class DoNothingController(ArmController):
    """Default placeholder — applies no torque."""
    def update(self, arm_state, t):
        return 0.0


class ConstantTorqueController(ArmController):
    """Example: applies a fixed torque for a set duration, then stops."""
    def __init__(self, arm_state, torque=5.0, duration=2.0):
        super().__init__(arm_state)
        self.torque = torque
        self.duration = duration

    def update(self, arm_state, t):
        if t < self.duration:
            return self.torque
        return 0.0


# ── Add your PID controller here when ready ──

class PIDController(ArmController):
    def __init__(self, arm_state, target_angle, kp, ki, kd, gravity_ff=False):
        super().__init__(arm_state)
        self.target_angle = target_angle
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # When True, add a feed-forward torque that cancels the arm's own weight,
        # so the PID only has to correct residuals instead of holding up gravity.
        self.gravity_ff = gravity_ff

        # internal PID state
        self.totalerror = 0.0
        self.prev_error = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
    
    def on_enable(self):
        # reset PID state every time the controller is enabled
        self.totalerror = 0.0
        self.prev_error = self.target_angle - self.arm_state["angle"]
    
    def on_disable(self):
        # nothing to clean up, but you could log final error here
        pass

    def update(self, arm_state, t):
        error = self.target_angle - arm_state["angle"]
        
        p = self.kp * error

        self.totalerror += error * dt
        i = self.ki * self.totalerror

        derivative = (error - self.prev_error) / dt
        d = self.kd * derivative
        self.prev_error = error

        # Gravity feed-forward: the weight torque physics applies is
        # -m·g·(L/2)·cos(angle), so add its positive to cancel it exactly.
        ff = 0.0
        if self.gravity_ff:
            L = arm_state["length"]
            ff = arm_state["mass"] * GRAVITY * (L / 2) * cos(arm_state["angle"])

        self.last_p = p
        self.last_i = i
        self.last_d = d
        return p + i + d + ff

class CircularController(ArmController):
    def __init__(self, arm_state, target_angle, kp, ki, kd, circular=False):
        super().__init__(arm_state)
        self.target_angle = target_angle
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.circular = circular

        self.totalerror = 0.0
        self.prev_error = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def _error(self, current):
        raw = self.target_angle - current
        if self.circular:
            return (raw + pi) % (2 * pi) - pi
        return raw

    def on_enable(self):
        self.totalerror = 0.0
        self.prev_error = self._error(self.arm_state["angle"])

    def on_disable(self):
        pass

    def update(self, arm_state, t):
        error = self._error(arm_state["angle"])

        p = self.kp * error

        self.totalerror += error * dt
        i = self.ki * self.totalerror

        derivative = (error - self.prev_error) / dt
        d = self.kd * derivative
        self.prev_error = error
        self.last_p = p
        self.last_i = i
        self.last_d = d
        return p + i + d