"""
Gymnasium environment for the CLAW-ONLY grasp task (a focused successor to robot_env.py).

Scope
-----
The full catch-then-deliver task (robot_env.RobotArmCatchEnv, now archived) plateaued
because one policy had to learn aiming, grasping AND carrying at once. This env isolates the
one skill worth learning with RL — timing the grasp on a MOVING box — and leaves aiming to
the simulator's own analytic control, matching how it'll plug back into simulation.py:

  * The arm is POSED each episode (via the sim's IK) to reach an intercept point in its reach —
    the SAME angled-finger pose the live sim (simulation.py) settles into when it reaches for a
    box. The agent does NOT move the arm.
  * A single box then drifts in along WORLD −x (exactly like simulation.py's boxes, which spawn
    far out at +x and drift inward) at a curriculum-controlled speed, CROSSING the claw from the
    side — the claw's fingers point ~60deg off the −x drift, so this is a side approach, not a
    straight-down-the-fingers one. This matches how the policy is actually used in the sim (an
    earlier version trained boxes coming straight down the finger axis, which never happens in
    the live sim, so the policy fired spurious closes on the real side approach). The base
    rotation (curriculum `azimuth`) sets the ANGLE the −x drift makes with the claw (0 = head-on
    to the base's facing). The real grip physics are kept intact: a grip forms only when BOTH
    fingers contact the box, and closing takes real time (CLAW_MAX_AV) — so the agent must commit
    EARLY (anticipate) as the box crosses the narrow grasp window.

  * Observation : the box's position RELATIVE to the grasp point (3) + the box's velocity (3)
                  + the claw angle (1) + the base angle (1) = 8 floats. Claw-frame relative
                  position keeps it orientation-invariant; the raw base angle is along for
                  context (and to expose any pose-dependent dynamics).
  * Action      : a single scalar a in [-1, 1]. a <= commit_threshold -> keep the claw open
                  and wait; a > commit_threshold -> COMMIT: close the claw, run the grip to
                  completion, score, and END the episode. One terminal attempt per episode.
  * Reward      : terminal. A COMMITTED close is scored by the grasp result:
                    grabbed     -> +1  +  a bounded one-time CENTERING bonus (how centred the
                                   box sat between the fingers).
                    committed miss -> a continuous curve in the closest grasp-prism distance d:
                                   +near_reward when it just missed (d~0), 0 at the catch-zone
                                   edge (catchable_dist), and NEGATIVE beyond — so committing
                                   while the box is outside the catch zone (too early / too
                                   late) is PENALISED, not merely paid less.
                  A NO-COMMIT ending (the box passes by, or the episode times out) is a flat
                  miss — no distance credit — so passivity can't farm the near_reward by letting
                  the box roll into the open claw and sitting there. Plus a small per-step
                  time_penalty, so the only way to earn positive reward is to commit well.

Curriculum (see train.py STAGES): lead with SPEED (fixed head-on approach, 0.5 -> 1.0 ->
random), then widen the approach ANGLE as a range, and only randomise the spawn position last.

`simulation.py` is loaded HEADLESS exactly as in robot_env.py (a stub `vpython` module,
no rendering), or with real VPython when render=True (watch.py).

Requires: gymnasium, numpy
Quick check:  python claw_env.py
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
#  Headless load of the simulator (identical to robot_env.py)
# ─────────────────────────────────────────────────────────────────────────────
def _install_headless_vpython():
    """Put a non-rendering stub `vpython` in sys.modules so simulation.py loads without a
    browser/GUI. Real vector math is kept; every visual/widget factory returns an inert object
    that just remembers its kwargs, so .pos/.size/.axis read back as created."""
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
    returning the module namespace with every helper (step_robot, command_arm_to, close_claw,
    claw_grab_point, _target_config, ...) ready to drive. headless=True uses the stub vpython
    (fast, no window); headless=False loads real vpython for a rendered window (watch.py)."""
    if headless:
        _install_headless_vpython()
    sim_dir = os.path.dirname(_SIM_PATH)
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    src = open(_SIM_PATH, encoding="utf-8").read()
    src = src[:src.index("\nwhile True:")]           # drop the render loop
    ns = {"__name__": "sim_headless" if headless else "sim_watch"}
    exec(compile(src, _SIM_PATH, "exec"), ns)
    return ns


# ─────────────────────────────────────────────────────────────────────────────
#  The environment
# ─────────────────────────────────────────────────────────────────────────────
class ClawGraspEnv(gym.Env):
    """One arm, pre-aimed, learns to TIME a single grasp on a box moving toward the claw.
    See the module docstring for obs/action/reward.

    Parameters
    ----------
    frame_skip   : physics frames (dt each) advanced per env step while waiting. Default 5 —
                   finer than robot_env's 10 so the commit can be timed to the step at high
                   box speeds (the catch window is only a few steps wide near speed ~2.5).
    max_steps    : env steps before truncation (counts as a miss — the agent never committed).
    speed_range  : (lo, hi); the box's inbound (−x) speed is sampled uniformly here each episode.
    azimuth_range: (lo, hi) RADIANS; the horizontal angle of the INTERCEPT point around the base
                   (which the base rotates to face). Since the box always drifts −x, this sets
                   the angle the drift makes with the claw: (0, 0) = −x drift head-on to the
                   base's +x facing; ±pi/2 = the box crosses fully sideways. Widen across stages.
    elevation_range: (lo, hi) RADIANS; vertical angle of the intercept point. (0, 0) = level.
    lateral_offset_max: max offset of the box's −x path from the grab point, in the CLAW'S OWN
                   finger-gap/off-plane plane (not world y-z — see reset(), this keeps a given
                   offset magnitude equally severe at every azimuth). A box that doesn't cross
                   dead-centre, so CENTERING matters. 0 = always centred. The reliably-catchable
                   envelope is ~0.14; beyond ~0.18 the grip often can't seat, so keep it inside
                   the catch area.
    lateral_offset_min: minimum offset magnitude (default 0). Set > 0 for an "imperfect
                   alignment" stage where the box is ALWAYS non-trivially off-centre (sampled in
                   [min, max]) instead of mostly near-centred (uniform from 0).
    grasp_radius : distance from the base the claw is pre-aimed to (where the box is intercepted).
    spawn_gap    : how far beyond the grasp point (along the approach) the box starts.
    near_reward  : the committed-miss reward at d=0 ("just missed"); the miss curve is
                   near_reward*(1 - d/catchable_dist), crossing 0 at the catch-zone edge
                   (catchable_dist) and going negative for commits outside it (too early/late).
    centering_weight: max centering bonus added to a +1 grab (dead-centre grip). Bounded, one-time.
    centering_tol: offset (units) at which the centering bonus reaches 0 (an edge grip).
    time_penalty : per-step cost so indefinitely waiting isn't free.
    commit_threshold: action value above which the agent COMMITS to the (terminal) close.
    catchable_dist: prism distance under which the box counts as within the grasp (used to
                   detect a "just missed" close and to detect the box leaving without a commit).
    grab_resolve_frames: max frames to run after a commit waiting for the grip to FORM (the
                   close takes ~7 frames; give it a little slack for a box still arriving).
    settle_frames: extra frames run AFTER the grip forms so the claw finishes closing and seats
                   on the box (the grip registers mid-close, ~0.56 rad, before the fully-shut
                   ~0.45), and so a momentary touch that slips is rejected — grabbed is re-checked
                   after settling. Also lets the completed close be seen when rendering.
    """
    metadata = {"render_modes": []}

    def __init__(self, frame_skip=5, max_steps=60,
                 speed_range=(0.5, 0.5), azimuth_range=(0.0, 0.0),
                 elevation_range=(0.0, 0.0), lateral_offset_max=0.0, lateral_offset_min=0.0,
                 grasp_radius=1.8, spawn_gap=0.9,
                 near_reward=0.3, centering_weight=0.5, centering_tol=0.12,
                 time_penalty=0.002, commit_threshold=0.0,
                 catchable_dist=0.3, grab_resolve_frames=14, settle_frames=6, render=False):
        super().__init__()
        self.render_scene = bool(render)
        self.sim = _load_simulation(headless=not self.render_scene)
        self._rate = self.sim["rate"] if self.render_scene else None
        self.robot = self.sim["robots"][0]
        self.balls = self.sim["balls"]
        self.dt = self.sim["dt"]
        self._vector = self.sim["vector"]
        self._mag = self.sim["mag"]

        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.azimuth_range = (float(azimuth_range[0]), float(azimuth_range[1]))
        self.elevation_range = (float(elevation_range[0]), float(elevation_range[1]))
        self.lateral_offset_max = float(lateral_offset_max)
        self.lateral_offset_min = float(lateral_offset_min)
        self.grasp_radius = float(grasp_radius)
        self.spawn_gap = float(spawn_gap)
        self.near_reward = float(near_reward)
        self.centering_weight = float(centering_weight)
        self.centering_tol = float(centering_tol)
        self.time_penalty = float(time_penalty)
        self.commit_threshold = float(commit_threshold)
        self.catchable_dist = float(catchable_dist)
        self.grab_resolve_frames = int(grab_resolve_frames)
        self.settle_frames = int(settle_frames)

        self._anchor = np.array([self.robot["anchor"].x, self.robot["anchor"].y,
                                 self.robot["anchor"].z], dtype=float)
        self.target = None            # the one box the agent must grab (self.balls[k])
        self._grasp_pt_np = None      # the pre-aimed grasp point (np, world) — held all episode
        self._dir = None              # unit inbound direction of the box (np)
        self._steps = 0
        self._committed = False

        # obs = box pos relative to grasp point (3) + box velocity (3) + claw angle + base angle
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        # action = [a]; a <= commit_threshold -> wait (claw open); a > commit_threshold -> commit.
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    # ── geometry ───────────────────────────────────────────────────────────────
    def _outward_radial(self, az, el):
        """Unit outward radial at azimuth az / elevation el — used to pick WHERE around the arm
        to place the interception (grasp) point. Base rotation is about +y and az=0 faces +x
        (matches command_arm_to's convention), so it's (cos el cos az, sin el, cos el sin az).
        The claw's ACTUAL approach axis (which the box travels down) is read separately from the
        achieved arm pose via _claw_axes, since the arm sags a little under gravity."""
        out = np.array([math.cos(el) * math.cos(az), math.sin(el),
                        math.cos(el) * math.sin(az)])
        return out / np.linalg.norm(out)

    def _grasp_point(self):
        """Current world grasp point (where a held box sits) as an np array."""
        gp = self.sim["claw_grab_point"](self.robot["arm2_state"],
                                         self.robot["base_state"]["angle"])
        return np.array([gp.x, gp.y, gp.z], dtype=float)

    def _claw_axes(self):
        """The claw's actual orthonormal frame (np unit vectors): approach axis `ad` (the
        direction the fingers point — a box must travel DOWN this to stay on the gripper
        centreline), the finger-gap axis `up` (the fingers splay in ±up, so an offset here
        decides centering), and the off-plane axis `side`. Read AFTER pre-aiming, so it
        reflects the arm's achieved pose (which sags a little under gravity), not the ideal."""
        tip, ad, up, side = self.sim["claw_frame"](self.robot["arm2_state"],
                                                   self.robot["base_state"]["angle"])
        def _np(v):
            a = np.array([v.x, v.y, v.z], dtype=float)
            n = np.linalg.norm(a)
            return a / n if n > 1e-9 else a
        return _np(ad), _np(up), _np(side)

    def _pre_aim(self, world_pt):
        """Snap the arm (via IK) so the claw's grasp point sits at world_pt and PID-hold it
        there — the analytic aiming the agent doesn't have to learn. Mirrors robot_env's
        _place_claw_at: set joint angles directly (no travel, so nothing brushes the box) then
        command_arm_to to hold."""
        v = self._vector(float(world_pt[0]), float(world_pt[1]), float(world_pt[2]))
        base, a1, a2 = self.sim["_target_config"](self.robot, v)
        self.robot["base_state"]["angle"] = base
        self.robot["arm_state"]["angle"] = a1
        self.robot["arm2_state"]["angle"] = a2
        self.robot["base_state"]["angular_velocity"] = 0.0
        self.robot["arm_state"]["angular_velocity"] = 0.0
        self.robot["arm2_state"]["angular_velocity"] = 0.0
        self.robot["arm2_state"]["anchor"] = self.sim["arm_tip_3d"](self.robot["arm_state"], base)
        self.sim["command_arm_to"](self.robot, v)

    # ── frame stepping ───────────────────────────────────────────────────────────
    def _advance_frame(self):
        """One physics frame: move the free box along its velocity, hold the arm at the grasp
        point, then step the robot (which runs the sim's grip logic). No anti-fling pin — the
        box MUST travel its inbound path. Rate-limits + flushes the scene when rendering."""
        ns = self.sim
        if self._rate is not None:
            self._rate(int(round(1.0 / self.dt)))
        ns["_col_tick"] += 1
        for obj in self.balls:
            if obj.held_by is None and not obj.delivered:
                obj.pos = obj.pos + obj.velocity * self.dt
        # keep the arm pinned at the pre-aimed grasp point (the agent only controls the claw)
        gp = self._vector(float(self._grasp_pt_np[0]), float(self._grasp_pt_np[1]),
                          float(self._grasp_pt_np[2]))
        self.sim["command_arm_to"](self.robot, gp)
        ns["step_robot"](self.robot)

    def _caught(self):
        return self.robot["grip_angle"] is not None

    def _prism_dist(self):
        """Grasp-prism distance to the target box (0 = sitting where a close would grip)."""
        return self.sim["_claw_grab_distance"](self.robot, self.target.pos)

    def _centering(self):
        """Grip centering quality in [0, 1] from the box's lateral+off-plane offset in the claw
        frame: 1 = dead-centre, 0 at centering_tol or worse. Only meaningful once gripped."""
        _, v, w = self.sim["_grab_offsets"](self.robot, self.target.pos)
        off = math.hypot(float(v), float(w))
        return max(0.0, 1.0 - off / self.centering_tol)

    def _obs(self):
        # Box position RELATIVE to the CURRENT claw grab point (+ box velocity, claw angle, base
        # angle) — all world-frame. Built from the live grab point (claw_grab_point) each step,
        # exactly as simulation.py's _rl_wants_close does, so training and deployment see the
        # identical observation.
        b = self.target
        ba = self.robot["base_state"]["angle"]
        gp = self.sim["claw_grab_point"](self.robot["arm2_state"], ba)
        rel = np.array([b.pos.x - gp.x, b.pos.y - gp.y, b.pos.z - gp.z])
        parts = [rel[0], rel[1], rel[2],
                 b.velocity.x, b.velocity.y, b.velocity.z,
                 self.robot["claw_state"]["angle"], ba]
        return np.asarray(parts, dtype=np.float32)

    # ── gym API ──────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim["_respawn"](self.sim["spawn_all"])
        # Use the first box as the target; park every other box far away and mark it delivered
        # so the sim's grip logic and our scoring ignore it entirely (single-box task).
        self.target = self.balls[0]
        for other in self.balls[1:]:
            other.held_by = None
            other.delivered = True
            other.velocity = self._vector(0, 0, 0)
            other.pos = self._vector(100.0, 100.0, 100.0)
        self.target.held_by = None
        self.target.delivered = False

        # Sample the episode's intercept geometry and speed.
        az = self.np_random.uniform(*self.azimuth_range)
        el = self.np_random.uniform(*self.elevation_range)
        speed = self.np_random.uniform(*self.speed_range)
        out = self._outward_radial(az, el)               # intercept-point direction (front hemi)

        # Pose the arm (real IK) to reach the intercept point and hold there — the SAME angled-
        # finger pose the live sim settles into when it reaches for a box. The box then drifts in
        # along WORLD −x, exactly like simulation.py's boxes, and crosses the claw FROM THE SIDE.
        # We deliberately do NOT align the approach with the claw's finger axis (that was the old
        # setup the deployed policy couldn't use — see the module docstring). `az` now sets the
        # base rotation, so it controls the ANGLE the −x drift makes with the claw, 0 = head-on.
        grasp_world = self._anchor + self.grasp_radius * out
        self._pre_aim(grasp_world)
        self._grasp_pt_np = self._grasp_point()          # actual grab point after IK
        self._dir = np.array([-1.0, 0.0, 0.0])           # box travels −x (comes from +x)

        # Lateral offset in the CLAW'S OWN (finger-gap, off-plane) plane — NOT world (y,z). World-
        # frame sampling made a given offset magnitude land far more severely on the claw's tight-
        # tolerance v/w axes at some azimuths than others (measured: ~3x worse effective
        # misalignment at moderate azimuth than at extreme azimuth — a coordinate-frame artifact
        # of the claw rotating with azimuth while a world-fixed offset direction didn't). Sampling
        # in the claw's own frame makes a given offset magnitude equally severe at every azimuth,
        # so difficulty reflects genuine off-centring, not which way the arm happens to be turned.
        perp = np.zeros(3)
        if self.lateral_offset_max > 0.0:
            _, gap_axis, side_axis = self._claw_axes()
            ang = self.np_random.uniform(0.0, 2.0 * math.pi)
            r = self.np_random.uniform(self.lateral_offset_min, self.lateral_offset_max)
            perp = r * (math.cos(ang) * gap_axis + math.sin(ang) * side_axis)

        # Box starts a gap out along +x from the grab point (offset in the claw's v-w plane),
        # drifting −x through it.
        box_start = self._grasp_pt_np - self.spawn_gap * self._dir + perp
        self.target.pos = self._vector(*box_start)
        self.target.velocity = self._vector(*(speed * self._dir))
        self.sim["open_claw"](self.robot)               # start open

        self._steps = 0
        self._committed = False
        return self._obs(), {}

    def _resolve_close(self):
        """Commit: close the claw and run the grip to completion, tracking the CLOSEST the box
        got to the grasp prism. Waits up to grab_resolve_frames for the grip to FORM, then runs
        settle_frames MORE so the claw finishes closing/seating (the grip registers mid-close,
        ~0.56 rad, before the fully-shut ~0.45) and the grip is confirmed to still hold — not a
        momentary touch. Returns (grabbed, min_prism_dist)."""
        self.sim["close_claw"](self.robot)
        min_d = self._prism_dist()
        settle_left = None                               # None until the grip first forms
        f = 0
        while True:
            self._advance_frame()
            min_d = min(min_d, self._prism_dist())
            f += 1
            if settle_left is None:
                if self._caught():
                    settle_left = self.settle_frames     # grip formed -> let it finish seating
                elif f >= self.grab_resolve_frames:
                    break                                # never gripped in time -> give up
            else:
                settle_left -= 1
                if settle_left <= 0:
                    break                                # settled (grip re-checked below)
        return self._caught(), min_d

    def _score_miss(self, d):
        """Committed-miss terminal reward: a continuous curve in the closest prism distance d —
        +near_reward at d=0 ("just missed"), 0 at the graspable-zone edge (catchable_dist), and
        NEGATIVE beyond. So a close committed while the box is OUTSIDE the catch zone (too early
        or too late) is penalised, not merely paid less — the box never spawns far enough (in
        prism terms) for a `1 - d` curve to ever go negative, which left early commits rewarded."""
        return self.near_reward * (1.0 - d / self.catchable_dist)

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        reward = -self.time_penalty                      # small cost for taking a step
        info = {"caught": False, "committed": False, "centering": 0.0, "outcome": "waiting"}

        if a > self.commit_threshold:                    # COMMIT -> terminal grasp attempt
            self._committed = True
            info["committed"] = True
            grabbed, min_d = self._resolve_close()
            if grabbed:
                centering = self._centering()
                reward += 1.0 + self.centering_weight * centering
                info.update(caught=True, centering=centering, outcome="grabbed")
            else:
                reward += self._score_miss(min_d)
                info["outcome"] = "missed_close"
            self._steps += 1
            return self._obs(), reward, True, False, info

        # WAIT: keep the claw open and advance; end if the box passes the grasp point (a miss).
        for _ in range(self.frame_skip):
            self._advance_frame()
        self._steps += 1

        # A no-commit ending (the box goes by, or the episode times out) is a FLAT miss — no
        # distance credit. The distance-graded near_reward is reserved for a genuine COMMITTED
        # close, so the agent can't farm it by letting the box roll into the open claw and
        # sitting there (at low speed the box parks at the grasp point, d~0). Only committing
        # earns positive reward; passivity nets just the accrued time penalty.
        b_pos = np.array([self.target.pos.x, self.target.pos.y, self.target.pos.z])
        to_grasp = self._grasp_pt_np - b_pos
        passed = float(np.dot(to_grasp, self._dir)) < 0.0   # box is now beyond the grasp point
        d = self._prism_dist()
        if passed and d > self.catchable_dist:              # went by without a committed close
            info["outcome"] = "passed_by"
            return self._obs(), reward, True, False, info

        truncated = self._steps >= self.max_steps
        if truncated:
            info["outcome"] = "timeout"
        return self._obs(), reward, False, truncated, info


# Register so it can also be created with gym.make("ClawGrasp-v0").
try:
    gym.register(id="ClawGrasp-v0", entry_point="claw_env:ClawGraspEnv")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test:  python claw_env.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = ClawGraspEnv()
    print("observation space:", env.observation_space)
    print("action space     :", env.action_space)
    obs, info = env.reset(seed=0)
    print("reset obs shape  :", obs.shape, "| obs:", np.round(obs, 3))

    # A hand-coded oracle: wait until the box reaches the grab point (obs is its position
    # RELATIVE to that point, so |rel| small == seated), then commit. Confirms the
    # physics/geometry line up — a well-timed close actually grabs.
    grabbed_eps = n_ep = 0
    returns, centerings = [], []
    for ep in range(30):
        obs, _ = env.reset(seed=ep)
        done, ret = False, 0.0
        while not done:
            rel = obs[:3]                         # box position relative to the grab point
            act = [1.0] if float(np.linalg.norm(rel)) < 0.18 else [-1.0]
            obs, r, term, trunc, info = env.step(act)
            ret += r
            done = term or trunc
        n_ep += 1
        ret_grab = int(info["caught"])
        grabbed_eps += ret_grab
        returns.append(ret)
        if info["caught"]:
            centerings.append(info["centering"])
    print(f"\noracle (commit when box reaches the grab point): grabbed {grabbed_eps}/{n_ep} "
          f"({grabbed_eps / n_ep:.0%})")
    print(f"  mean return {np.mean(returns):+.3f} | mean centering "
          f"{np.mean(centerings) if centerings else 0:.2f}")
    print("env works: reset/step/spaces all functioning.")
