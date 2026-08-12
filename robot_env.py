"""
Gymnasium environment wrapping the robot-arm simulator for reinforcement learning.

First version — catching, single arm:
  * Observation : every box's position + velocity, the arm's three joint angles, and the
                  claw angle.
  * Action      : a world-space target position for the arm (x, y, z) — a 3-vector in
                  [-1, 1]. The CLAW is AUTOMATIC (closes on its own when a box comes within
                  catchable_dist of the grasp), so the model only positions the arm.
  * Reward      : +1 ONCE, the step a box is caught (grips) — plus a slight dense shaping term
                  each step for the claw getting closer to the nearest FREE box (scaled by the
                  change in distance). Catching a box does NOT end the episode: the sim's own
                  grip logic (grip_angle + _carry_box_in_claw) already carries a held box
                  rigidly with the claw, so the arm keeps being driven by the model and must
                  carry the box to deliver_pos. A further +1 ONCE fires the step the box is
                  DELIVERED there (released), with its own dense shaping term for the grab
                  point reaching a NEW-BEST (closest-ever-this-episode) distance to deliver_pos
                  while holding the box — one-sided (never penalises drifting away) but bounded
                  to the running best so oscillating near the target can't repeatedly re-pay
                  for the same ground. Delivering IS the terminal goal — it ends the episode; a
                  miss (or a catch that never gets delivered) ends only by truncation
                  (max_steps). (Time and fling terms exist but default to off.)

`simulation.py` is a monolithic VPython script, so we load it HEADLESS: a stub `vpython`
module (real vector math, no rendering/GUI) is installed before the script is exec'd,
exactly like the project's headless test harness. Only one arm is active (the others are
commented out in simulation.py's `robots` list).

Requires: gymnasium, numpy   ->   pip install gymnasium
Quick check:                 ->   python robot_env.py
"""
import math
import os
import sys
import types

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_SIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation.py")


# ─────────────────────────────────────────────────────────────────────────────
#  Headless load of the simulator
# ─────────────────────────────────────────────────────────────────────────────
def _install_headless_vpython():
    """Put a non-rendering stub `vpython` in sys.modules so simulation.py loads without a
    browser/GUI. Real vector math (vector, mag, dot, cross, ...) is kept; every visual and
    widget factory returns an inert object that just remembers its kwargs, so attributes
    like .pos, .size, .axis still read back as the values they were created with."""
    import vpython as _vp
    fake = types.ModuleType("vpython")
    for name in ["vector", "mag", "mag2", "dot", "cross", "norm", "hat", "proj", "comp",
                 "diff_angle", "rotate", "sqrt", "pi", "radians", "degrees", "cos", "sin",
                 "tan", "exp", "acos", "asin", "atan", "atan2"]:
        if hasattr(_vp, name):
            setattr(fake, name, getattr(_vp, name))

    class _Dummy:
        def __init__(self, *a, **k):
            object.__setattr__(self, "_d", dict(k))

        def __getattr__(self, n):
            d = object.__getattribute__(self, "_d")
            return d[n] if n in d else _Dummy()

        def __setattr__(self, n, v):
            object.__getattribute__(self, "_d")[n] = v

        def __call__(self, *a, **k):
            return _Dummy(**k)

    def _factory(*a, **k):
        return _Dummy(**k)

    for name in ["box", "sphere", "cylinder", "curve", "arrow", "label", "canvas",
                 "vertex", "quad", "triangle", "text", "compound", "ring", "cone",
                 "pyramid", "ellipsoid", "helix", "points", "extrusion", "graph",
                 "gcurve", "gdots", "gvbars", "button", "winput", "wtext", "checkbox",
                 "radio", "slider", "menu", "rate", "sleep"]:
        setattr(fake, name, _factory)

    colors = _Dummy()
    colors.gray = lambda x=0: _Dummy()
    for cn in ["red", "green", "blue", "yellow", "orange", "cyan", "magenta",
               "white", "black", "purple"]:
        setattr(colors, cn, _Dummy())
    fake.color = colors
    fake.scene = _Dummy()
    if not hasattr(fake, "pi"):
        fake.pi = 3.141592653589793
    sys.modules["vpython"] = fake


def _load_simulation(headless=True):
    """Exec simulation.py up to (but not including) its live `while True:` render loop,
    returning the module namespace. All the module-level setup runs (robots, balls,
    tracker, sensors, ...) and every function (step_robot, command_arm_to, close_claw,
    open_claw, _respawn, spawn_all, ...) becomes available to drive from step()/reset().

    headless=True installs the stub vpython (no window, fast — for training). headless=False
    loads it with the REAL vpython, so simulation.py's `from vpython import *` builds a live,
    rendered scene (a browser window) — used by watch.py to see a policy act."""
    if headless:
        _install_headless_vpython()
    sim_dir = os.path.dirname(_SIM_PATH)
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)                  # so `from physics import *` resolves
    src = open(_SIM_PATH, encoding="utf-8").read()
    src = src[:src.index("\nwhile True:")]           # drop the render loop
    ns = {"__name__": "sim_headless" if headless else "sim_watch"}
    exec(compile(src, _SIM_PATH, "exec"), ns)
    return ns


# ─────────────────────────────────────────────────────────────────────────────
#  The environment
# ─────────────────────────────────────────────────────────────────────────────
class RobotArmCatchEnv(gym.Env):
    """One arm learns to catch a box. See the module docstring for obs/action/reward.

    Parameters
    ----------
    frame_skip  : how many physics frames (dt each) one env step advances — the agent acts
                  at 1/(frame_skip*dt) Hz. Default 10 -> ~10 Hz control at dt=0.01, and long
                  enough that a single "close" action lets the claw fully shut and grip.
    max_steps   : env steps before the episode is truncated (a miss).
    shape_weight: weight on the per-step distance-shaping reward — the reward for the claw
                  getting closer to the target (0 disables it). Higher = a stronger pull
                  toward the box. Default 0.3.
    time_penalty: cost subtracted every step to encourage catching sooner (0 disables it).
                  TEMP: disabled (0.0, was 0.0005).
    fling_penalty: penalty per unit of box-speed knocked AWAY each step, for swatting a box
                  away instead of catching it (0 disables it). TEMP: disabled (0.0, was 0.15).
    fling_away_only: if True, only the impulse that pushes a box AWAY from the claw is
                  penalised (nudging it toward the grasp is free), so the arm isn't scared
                  off the final approach. If False, any imparted speed is penalised (stricter
                  old form). Default True.
    catchable_dist: a box within this distance of the grasp prism counts as "catchable" —
                  the AUTOMATIC claw closes as soon as a free box is this close. Default 0.3.
    start_at_box: if True (and start_pos is None), each episode starts with the claw snapped
                  ONTO the box (open) — the easiest curriculum. Default False.
    start_pos: fixed world point (x, y, z) the claw is snapped to at the start of each
                  episode (open), or None to use start_at_box. TAKES PRECEDENCE over
                  start_at_box. Default (1, 0, 0) — with the box at (3, 0, 0) this leaves a
                  ~2-unit gap for the arm to actually approach (so the distance shaping earns
                  positive reward). The claw is left open, so it still has to close.
    anti_fling: if True, free boxes are pinned in place each frame so the claw can't knock
                  them away — the box stays a fixed, catchable target throughout an episode
                  (a learning aid). A gripped box is exempt. TEMP: re-enabled (True, was
                  False) so the box physically can't be flung, not just penalty-discouraged.
    catch_delay_frames: extra physics frames given to a step the moment the auto-claw STARTS
                  closing (i.e. frame_skip is extended by this many frames for that step
                  only), so the claw has more real time to fully shut on the box before the
                  step ends — without changing how fast the claw physically closes (that's
                  CLAW_MAX_AV in physics.py, untouched). Default 20.
    deliver_pos: fixed world point (x, y, z) the caught box must be carried to and released —
                  matches the full sim's DELIVER_TARGET. Default (3, 0, 0).
    deliver_dist: once caught, the claw's grab point within this distance of deliver_pos counts
                  as "delivered" — the box is released there and the episode ends. Default 0.3.
    deliver_shape_weight: weight on the per-step distance-shaping reward for the grab point
                  reaching a NEW-BEST distance to deliver_pos WHILE HOLDING the box (0 disables
                  it). Unlike shape_weight, drifting away is never penalised (stays unpaid, not
                  negative) — but unlike a plain last-step comparison, it only pays for beating
                  the closest distance reached so far THIS episode, so oscillating near the
                  target can't re-earn the same ground twice. Default 0.3.
    box_spawn_pos: curriculum knob. If set to a world point (x, y, z), every box is pinned to
                  THAT point at the start of each episode (overriding simulation.py's random
                  reachable scatter), giving a fixed, repeatable target — e.g. spawn the box
                  right at deliver_pos so a catch is almost a delivery (stage 1), or at a fixed
                  spot far from it to isolate the carry (stage 2). None (default) keeps the
                  random scatter (the harder, general case).
    box_spawn_region: curriculum knob for PROGRESSIVE randomization. If set to
                  ((cx,cy,cz), (hx,hy,hz)) — a centre and half-extents — each episode's box is
                  sampled uniformly inside that axis-aligned box (rejecting points outside the
                  arm's [1,3]-unit reach shell), instead of at a single fixed point or the full
                  random scatter. Widening the half-extents across stages edges the spawn
                  smoothly from a small patch near the easy zone up to the full distribution
                  (at full width this region equals simulation.py's _scatter_pos). Ignored if
                  box_spawn_pos is set; None (default) with box_spawn_pos None = full scatter.
    grip_offplane_tol: the auto-claw only CLOSES on a box whose off-plane offset (its distance
                  from the grasp PLANE along the claw's `side` axis) is within this — on top of
                  the catchable_dist prism test. The fingers sweep in-plane, so a box further
                  off-plane than ~one box-half-width + finger thickness can't be seated no
                  matter how small the prism distance reads; without this gate the claw fires
                  closes that geometrically can't grip (a big share of the observed "reached
                  the box but no grip" failures). Default 0.12 (just inside the measured ~0.13
                  hard limit for a 0.2 box).
    """
    metadata = {"render_modes": []}

    # World half-range the normalised action target maps onto. The arm reaches ~4 units;
    # command_arm_to clamps anything beyond its own limits, so this only needs to cover
    # the reachable region plus a little slack.
    TARGET_RANGE = 5.0

    def __init__(self, frame_skip=10, max_steps=1000, shape_weight=0.3, time_penalty=0.0,
                 fling_penalty=0.0, fling_away_only=True,
                 catchable_dist=0.3, start_at_box=False, start_pos=(1.0, 0.0, 0.0),
                 anti_fling=True, catch_delay_frames=20,
                 deliver_pos=(3.0, 0.0, 0.0), deliver_dist=0.3, deliver_shape_weight=0.3,
                 box_spawn_pos=None, box_spawn_region=None,
                 grip_offplane_tol=0.12, render=False):
        super().__init__()
        self.render_scene = bool(render)             # True -> real VPython window (watch.py)
        self.sim = _load_simulation(headless=not self.render_scene)
        # When rendering, pace each physics frame to ~real time and flush the scene with
        # vpython.rate(); headless training never rate-limits.
        self._rate = self.sim["rate"] if self.render_scene else None
        self.robot = self.sim["robots"][0]           # the single active arm
        self.balls = self.sim["balls"]
        self.dt = self.sim["dt"]
        self._vector = self.sim["vector"]
        self._mag = self.sim["mag"]
        self._dot = self.sim["dot"]
        self._reach = self.sim["REACH_MAX"]          # commanded targets clamped to this radius
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.shape_weight = float(shape_weight)      # weight on the distance shaping term
        self.time_penalty = float(time_penalty)      # per-step cost -> catch sooner
        self.fling_penalty = float(fling_penalty)    # penalty for knocking a box (imparting speed)
        self.fling_away_only = bool(fling_away_only)  # only penalize knocking it AWAY from the claw
        self.catchable_dist = float(catchable_dist)  # box this close to the grasp -> auto-close
        self.start_at_box = bool(start_at_box)       # start each episode with the claw ON the box
        self.start_pos = tuple(start_pos) if start_pos is not None else None  # fixed claw start
        self.anti_fling = bool(anti_fling)           # pin free boxes so the claw can't knock them
        self.catch_delay_frames = int(catch_delay_frames)  # extra frames given to a close (see step)
        self.deliver_pos = tuple(deliver_pos)         # fixed world point to drop the box at
        self._deliver_pos_v = self._vector(*self.deliver_pos)  # cached vector form
        self.deliver_dist = float(deliver_dist)       # grab point this close to deliver_pos -> delivered
        self.deliver_shape_weight = float(deliver_shape_weight)  # weight on carry-to-target shaping
        self.box_spawn_pos = tuple(box_spawn_pos) if box_spawn_pos is not None else None  # fixed box spawn
        if box_spawn_region is not None:                # (center, half-extents) random spawn box
            c, h = box_spawn_region
            self.box_spawn_region = (tuple(c), tuple(h))
        else:
            self.box_spawn_region = None
        self.grip_offplane_tol = float(grip_offplane_tol)  # max off-plane box offset the auto-claw closes on
        self._pins = []                              # each box's home position (anti-fling)
        self._steps = 0
        self._prev_dist = None                       # claw->box distance last step (shaping)
        self._prev_vel = []                          # free-box velocities last step (fling term)
        self._best_deliver_dist = None                # closest grab->deliver_pos distance REACHED
                                                       # so far this episode, while held (a running
                                                       # best, not last-step — see step() for why)
        self._ever_caught = False                    # latches True the step a catch happens —
                                                       # this is how callers (train.py/watch.py)
                                                       # tell whether an episode included a catch
        self._ever_delivered = False                  # latches True the step the box is delivered —
                                                       # delivery DOES end the episode (terminated)

        n_box = len(self.balls)
        obs_dim = n_box * 6 + 3 + 1                   # boxes(pos+vel) + arm angles + claw
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        # action = [target_x, target_y, target_z] (each in [-1, 1]). The CLAW is automatic —
        # it closes on its own the moment a box is grippable (within catchable_dist of the
        # grasp AND within the off-plane grip limit) — so the model only positions the arm.
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self._claw_shut = False                      # tracks the auto-claw's commanded state

    # ── internals ────────────────────────────────────────────────────────────
    def _advance_frame(self):
        """One physics frame: the live loop's body minus the GUI/auto-catch (the AGENT
        does the catching now). Advances every free box, then steps the arm — which runs
        the claw's grip logic, so a box gets gripped exactly as in the full simulation.
        When rendering (watch.py), rate-limits to ~real time and flushes the scene."""
        ns = self.sim
        if self._rate is not None:
            self._rate(int(round(1.0 / self.dt)))    # pace + flush the render (~real time)
        ns["_col_tick"] += 1
        for obj in self.balls:
            if obj.held_by is None and not obj.delivered:
                obj.pos = obj.pos + obj.velocity * self.dt
        ns["step_robot"](self.robot)
        if self.anti_fling:
            # Pin free (un-gripped) boxes to their home position so the claw can't knock
            # them away — makes the catch easy to discover during learning. A box that gets
            # GRIPPED (held_by set) is exempt; it's carried by the claw as normal.
            #
            # Fix 1: a box the claw is ACTIVELY CLOSING on (claw shutting + box within the
            # grasp prism) is also exempt from the position pin. As the fingers close they push
            # a slightly-off-centre box toward the middle (resolve_finger_ball_collision); re-
            # pinning it home every frame UNDID that push-out, so the grip failed on contact
            # for anything not already near-perfectly centred (measured: ±0.03 pinned vs ±0.33
            # un-pinned in the finger-gap axis). We still zero its velocity, so a failed close
            # can't fling it away — anti_fling's real job (no knock-away during the APPROACH) is
            # kept, we just stop fighting the fingers once they're committed to the grab.
            gd_fn = self.sim["_claw_grab_distance"]
            for obj, home in zip(self.balls, self._pins):
                if obj.held_by is None and not obj.delivered:
                    closing_on_it = (self._claw_shut
                                     and gd_fn(self.robot, obj.pos) <= self.catchable_dist)
                    if closing_on_it:
                        obj.velocity = self._vector(0, 0, 0)     # let the push-out re-centre it
                    else:
                        obj.pos = self._vector(home.x, home.y, home.z)
                        obj.velocity = self._vector(0, 0, 0)

    def _obs(self):
        parts = []
        for b in self.balls:
            p, v = b.pos, b.velocity
            parts += [p.x, p.y, p.z, v.x, v.y, v.z]
        parts += [self.robot["base_state"]["angle"],
                  self.robot["arm_state"]["angle"],
                  self.robot["arm2_state"]["angle"],
                  self.robot["claw_state"]["angle"]]
        return np.asarray(parts, dtype=np.float32)

    def _caught(self):
        """True when the claw has a box firmly gripped (a caught state)."""
        return self.robot["grip_angle"] is not None

    def _claw_box_distance(self):
        """Distance from the claw's grab point (where a held box would sit — the 'claw
        hold') to the nearest FREE box: the box the arm is effectively reaching for.
        Returns None if no free box remains."""
        gp = self.sim["claw_grab_point"](self.robot["arm2_state"],
                                         self.robot["base_state"]["angle"])
        dists = [self._mag(gp - b.pos) for b in self.balls
                 if b.held_by is None and not b.delivered]
        return min(dists) if dists else None

    def _box_velocities(self):
        """A copy of every box's current velocity vector, indexed like self.balls."""
        return [self._vector(b.velocity.x, b.velocity.y, b.velocity.z) for b in self.balls]

    def _place_claw_at(self, pos):
        """Snap the arm so its claw grab point sits at `pos`, instantly (via IK, not by
        driving there — so it doesn't brush/fling the box on the way in). Used to start an
        episode already ON the box: the agent learns the CATCH (the close) first, before the
        harder approach. The claw is left open (from the reset), so it still has to close."""
        base, a1, a2 = self.sim["_target_config"](self.robot, pos)   # IK for the grab point
        self.robot["base_state"]["angle"] = base
        self.robot["arm_state"]["angle"] = a1
        self.robot["arm2_state"]["angle"] = a2
        self.robot["base_state"]["angular_velocity"] = 0.0
        self.robot["arm_state"]["angular_velocity"] = 0.0
        self.robot["arm2_state"]["angular_velocity"] = 0.0
        # the elbow (arm2's anchor) follows arm1's new pose
        self.robot["arm2_state"]["anchor"] = self.sim["arm_tip_3d"](self.robot["arm_state"], base)
        self.sim["command_arm_to"](self.robot, pos)                  # PID holds it here

    def _nearest_grasp_distance(self):
        """Smallest distance from a free box to the claw's grasp PRISM (the graspable volume
        between the fingers): ~0 means a box is sitting where a close would catch it. None if
        no free box remains. (Prism distance only — see _grippable_target_dist for the version
        that also enforces the off-plane grip limit the auto-claw actually keys off.)"""
        gd = self.sim["_claw_grab_distance"]
        ds = [gd(self.robot, b.pos) for b in self.balls
              if b.held_by is None and not b.delivered]
        return min(ds) if ds else None

    def _grippable_target_dist(self):
        """Prism distance to the nearest free box the claw could actually GRIP if it closed
        now: within catchable_dist of the grasp prism AND within grip_offplane_tol of the
        grasp PLANE (the `side`-axis / off-plane offset). None if none qualifies. The extra
        off-plane gate is Fix 2: the fingers sweep in-plane, so a box further off-plane than
        ~a box-half-width + finger thickness can't be seated however small the prism distance
        reads — without this the auto-claw fires closes that geometrically can't grip."""
        gd_fn = self.sim["_claw_grab_distance"]
        off_fn = self.sim["_grab_offsets"]
        best = None
        for b in self.balls:
            if b.held_by is None and not b.delivered:
                d = gd_fn(self.robot, b.pos)
                if d <= self.catchable_dist:
                    _, _, w = off_fn(self.robot, b.pos)          # off-plane offset
                    if abs(w) <= self.grip_offplane_tol and (best is None or d < best):
                        best = d
        return best

    def _auto_claw(self):
        """Automatic claw: close it the moment a free box is actually GRIPPABLE (within
        catchable_dist AND within the off-plane grip limit — see _grippable_target_dist), so
        the arm grips it as in the sim; otherwise keep it open. The model does NOT control the
        claw — it only positions the arm. Toggled on state change so we don't re-issue the same
        command (and reset the claw PID) every frame."""
        if self._caught():
            return
        want_shut = self._grippable_target_dist() is not None
        if want_shut and not self._claw_shut:
            self.sim["close_claw"](self.robot)
            self._claw_shut = True
        elif not want_shut and not self._claw_shut:
            self.sim["open_claw"](self.robot)

    def _sample_region_pos(self):
        """A random reachable box position inside box_spawn_region (a centred axis-aligned
        box), rejecting points outside the arm's [1, 3]-unit reach shell — same shell
        simulation.py's _scatter_pos uses, so at full width the two match. Uses the env's
        seeded RNG (self.np_random) so a given reset seed is reproducible."""
        (cx, cy, cz), (hx, hy, hz) = self.box_spawn_region
        for _ in range(1000):
            x = self.np_random.uniform(cx - hx, cx + hx)
            y = self.np_random.uniform(cy - hy, cy + hy)
            z = self.np_random.uniform(cz - hz, cz + hz)
            if 1.0 <= (x * x + y * y + z * z) ** 0.5 <= 3.0:
                return self._vector(x, y, z)
        return self._vector(cx, cy, cz)               # fallback (region barely overlaps shell)

    # ── gym API ──────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Full reset: return the arm to rest, respawn the boxes, drop stale tracks.
        self.sim["_respawn"](self.sim["spawn_all"])
        if self.box_spawn_pos is not None:            # curriculum: pin the box to a FIXED point
            for b in self.balls:                      # (overrides simulation.py's random scatter)
                if not b.delivered:
                    b.pos = self._vector(*self.box_spawn_pos)
                    b.velocity = self._vector(0, 0, 0)
        elif self.box_spawn_region is not None:       # curriculum: random within a widening box
            for b in self.balls:
                if not b.delivered:
                    b.pos = self._sample_region_pos()
                    b.velocity = self._vector(0, 0, 0)
        if self.start_pos is not None:                # fixed claw start point (takes precedence)
            self._place_claw_at(self._vector(*self.start_pos))
        elif self.start_at_box:                       # curriculum: begin with the claw ON the box
            box = next((b for b in self.balls
                        if b.held_by is None and not b.delivered), None)
            if box is not None:
                self._place_claw_at(box.pos)
        # Record each box's home position for the anti-fling pin (after spawn/placement).
        self._pins = [self._vector(b.pos.x, b.pos.y, b.pos.z) for b in self.balls]
        self._steps = 0
        self._claw_shut = False                       # claw starts open (auto-controlled)
        self._prev_dist = self._claw_box_distance()   # baseline for the distance shaping
        self._prev_vel = self._box_velocities()       # baseline for the fling term
        self._best_deliver_dist = None                 # no carry-to-target baseline until caught
        self._ever_caught = False
        self._ever_delivered = False
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        tx, ty, tz = action[0], action[1], action[2]
        # Drive the arm's grab point toward the chosen world position. The claw is automatic.
        target = self._vector(float(tx) * self.TARGET_RANGE,
                              float(ty) * self.TARGET_RANGE,
                              float(tz) * self.TARGET_RANGE)
        # Limit the model's control to the arm's reach sphere: clamp the commanded target to
        # within self._reach (~4) of the base, so it can never aim the arm outside where it
        # can physically go. Projects an out-of-range target onto the sphere (keeps direction).
        rel = target - self.robot["anchor"]
        m = self._mag(rel)
        if m > self._reach:
            target = self.robot["anchor"] + rel * (self._reach / m)
        # If a box is ALREADY grippable, FREEZE the arm at its current grab point instead of
        # applying the model's new target. Otherwise command_arm_to keeps pulling the arm
        # toward wherever the model points for the whole frame_skip window — dragging it away
        # WHILE the claw is auto-closing on the box, so the fingers can bottom out on empty
        # air a moment after the box was sitting right in the grasp. Holding still lets the
        # close (already triggered by _auto_claw below) actually complete on the box. Uses the
        # same grippable test as the auto-claw (Fix 2), so we don't freeze onto a box the claw
        # will never seat (too far off-plane) — the arm stays free to reposition it instead.
        if self._grippable_target_dist() is not None:
            target = self.sim["claw_grab_point"](self.robot["arm2_state"],
                                                 self.robot["base_state"]["angle"])
        self.sim["command_arm_to"](self.robot, target)

        # Run frame_skip frames as usual — but if the auto-claw STARTS closing partway
        # through (the box just became catchable), extend this step by catch_delay_frames
        # extra frames so the claw has more real TIME to finish shutting on it before the
        # step ends, rather than being cut off mid-close by a short frame_skip. The claw's
        # own closing SPEED (CLAW_MAX_AV) is unchanged — this only lengthens how long we
        # keep advancing frames for it, and only extends once per step.
        total_frames = self.frame_skip
        extended = False
        i = 0
        while i < total_frames:
            was_shut = self._claw_shut
            self._auto_claw()                         # auto-close on a nearby box; else open
            if self._claw_shut and not was_shut and not extended:
                total_frames += self.catch_delay_frames  # just started closing -> give it time
                extended = True
            self._advance_frame()
            i += 1
            if self._caught():
                break                                # box caught mid-skip — stop early

        self._steps += 1
        caught = self._caught()
        # The +1 fires ONCE, on the step the catch happens (the caught->caught transition),
        # not every step afterward — a catch no longer ends the episode (the box is carried
        # by the claw via the sim's own grip logic instead), so paying it every step while
        # held would let the agent farm reward by just sitting there holding the box.
        newly_caught = caught and not self._ever_caught
        if newly_caught:
            self._ever_caught = True
        reward = 1.0 if newly_caught else 0.0
        # Time penalty: a small cost every step so the agent is pushed to catch SOONER
        # (a late catch is worth less than an early one). A no-catch episode accrues the
        # full -time_penalty*max_steps, so catching stays strictly better than giving up.
        reward -= self.time_penalty
        # Dense distance shaping: a slight reward each step the claw's grab point gets
        # CLOSER to the nearest FREE box, and a slight penalty for drifting away — scaled
        # by how much the distance changed. Using the CHANGE in distance (not the raw
        # distance) means the agent can't farm reward by just hovering near a box: only
        # actually closing the gap pays. _claw_box_distance already excludes held/delivered
        # boxes, so this naturally goes quiet (d is None) once the only box is caught.
        d = self._claw_box_distance()
        if d is not None and self._prev_dist is not None:
            reward += self.shape_weight * (self._prev_dist - d)
        self._prev_dist = d
        # Fling penalty: discourage knocking a box away. Scored on the CHANGE in each free
        # box's velocity this step (the impulse), so the knock itself is penalised while a
        # box already drifting (constant velocity) is not charged forever.
        #   fling_away_only (default): penalise ONLY the part of the impulse that pushes the
        #     box AWAY from the claw's grab point. Nudging it toward the grasp (or sideways)
        #     is free, so the arm can commit to the final approach instead of hovering out of
        #     fear of any contact — it's only punished for actually swatting the box away.
        #   else: penalise the full impulse magnitude (any contact), the stricter old form.
        if self.fling_penalty:
            vels = self._box_velocities()
            if self.fling_away_only:
                grab = self.sim["claw_grab_point"](self.robot["arm2_state"],
                                                   self.robot["base_state"]["angle"])
                pen = 0.0
                for i, b in enumerate(self.balls):
                    if b.held_by is None and not b.delivered:
                        rel = b.pos - grab                         # claw -> box direction
                        m = self._mag(rel)
                        if m > 1e-6:
                            away = self._dot(vels[i] - self._prev_vel[i], rel) / m
                            if away > 0.0:                         # impulse pushing it AWAY
                                pen += away
            else:
                pen = sum(self._mag(vels[i] - self._prev_vel[i])
                          for i, b in enumerate(self.balls)
                          if b.held_by is None and not b.delivered)
            reward -= self.fling_penalty * pen
            self._prev_vel = vels
        else:
            self._prev_vel = self._box_velocities()
        # (No claw-discipline penalty: the claw is automatic now, so there's no claw action to
        #  penalise — the only claw-related reward left is the +1 catch.)
        # Delivery: once caught, the box must be carried to deliver_pos and released there.
        # Dense shaping rewards NEW-BEST progress toward deliver_pos while HOLDING the box —
        # only pays when the grab point reaches a distance CLOSER than any point so far this
        # episode (a running best, not just closer than last step), and drifting away is never
        # penalised (stays unpaid, per the earlier "don't penalise" request). This closes the
        # oscillation-farming hole a plain last-step comparison has: nudging closer, retreating
        # (free), then re-covering the SAME ground again no longer pays twice — only genuine
        # progress past the best-ever mark does. That makes the running total for this term
        # telescope exactly to deliver_shape_weight * (dist_at_catch - dist_at_delivery), the
        # same hard cap the two-sided catch shaping gets "for free" from being symmetric.
        # A one-time +1 fires the step the box is dropped off, and delivering IS the terminal
        # goal (unlike catching, which no longer ends the episode), so it terminates the episode.
        delivered_now = False
        if caught:
            grab_pt = self.sim["claw_grab_point"](self.robot["arm2_state"],
                                                   self.robot["base_state"]["angle"])
            dd = self._mag(grab_pt - self._deliver_pos_v)
            if self._best_deliver_dist is None:
                self._best_deliver_dist = dd          # baseline the step it's first caught
            else:
                reward += self.deliver_shape_weight * max(0.0, self._best_deliver_dist - dd)
                self._best_deliver_dist = min(self._best_deliver_dist, dd)
            if dd <= self.deliver_dist:
                delivered_now = True
                reward += 1.0
                held = self.robot["held"]
                self.robot["grip_angle"] = None
                self.robot["grip_local"] = None
                self.robot["held"] = None
                if held is not None:
                    held.held_by = None
                    held.velocity = self._vector(0, 0, 0)
                    held.pos = self._deliver_pos_v
                    held.delivered = True
                self.sim["open_claw"](self.robot)
        else:
            self._best_deliver_dist = None
        if delivered_now:
            self._ever_delivered = True

        terminated = bool(self._ever_delivered)   # delivering the box ends the episode
        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {
            "caught": self._ever_caught, "delivered": self._ever_delivered,
        }


# Register so it can also be created with gym.make("RobotArmCatch-v0").
try:
    gym.register(id="RobotArmCatch-v0", entry_point="robot_env:RobotArmCatchEnv")
except Exception:
    pass   # already registered (re-import)


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test:  python robot_env.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = RobotArmCatchEnv()
    print("observation space:", env.observation_space)
    print("action space     :", env.action_space)
    obs, info = env.reset(seed=0)
    print("reset obs shape  :", obs.shape)

    total_r, n_ep, caught_eps, delivered_eps = 0.0, 0, 0, 0
    obs, _ = env.reset(seed=0)
    for t in range(3000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            n_ep += 1
            caught_eps += int(info["caught"])
            delivered_eps += int(info["delivered"])
            obs, _ = env.reset()
    print(f"\nran {t + 1} random steps over {n_ep} episodes")
    print(f"total reward: {total_r}  |  episodes with a catch: {caught_eps}  |  "
          f"delivered: {delivered_eps}")
    print("env works: reset/step/spaces all functioning.")
