from vpython import *
from math import *
from physics import *
from sys import *
from control import *
from sensor import *
import heapq
import random
import threading
import time

dt = 0.01
initial_av = 0

# Robot dimensions (shared by every arm)
ARM1_LEN = 2
ARM2_LEN = 1.8
ARM_WIDTH = 0.1
FINGER_LEN = 0.4
TOOTH_LEN = 0.15

def init():
    scene = canvas(title='simulation',
                   width=800, height=600,
                   center=vector(0,0,0),
                   background=color.white,
                   align='left')
    scene.select()

def endtask(b):
    print("Closing simulation...")
    b.disabled = True
    raise SystemExit

paused = False
def toggle_pause(b):
    global paused
    paused = not paused
    b.text = 'Resume' if paused else 'Pause'

# ─────────────────────────────────────────────
#  One robot = base + 2 arm segments + claw + ball.
#  Everything a robot needs lives in a single dict, so the simulation scales
#  to any number of arms just by adding entries to `robots`.
# ─────────────────────────────────────────────

def _quarter_dome(center, radius, col, zsign, ysign=1, stacks=4, slices=6):
    """The curved shell of a quarter sphere (one vertical half, one azimuthal half) built
    as a quad mesh — VPython has no quarter-sphere primitive and no CSG. Its flat bottom
    lies at center.y and its flat vertical face at z = center.z; it bulges toward zsign*z
    and toward ysign*y (ysign=+1 = apex up, ysign=-1 = apex down, a mirror image). Purely
    visual (static geometry — no physics). Returns the list of mesh faces (quads + cap
    triangles) it created, so the caller can merge them into a compound."""
    faces = []
    grid = []
    for i in range(stacks + 1):
        theta = (pi / 2) * i / stacks            # 0 = apex, pi/2 = flat rim at y=center.y
        row = []
        for j in range(slices + 1):
            phi = pi * j / slices                # 0..pi -> flat vertical face at both ends
            nrm = vector(sin(theta) * cos(phi), ysign * cos(theta), zsign * sin(theta) * sin(phi))
            row.append(vertex(pos=center + radius * nrm, normal=nrm, color=col))
        grid.append(row)
    for i in range(stacks):
        for j in range(slices):
            faces.append(quad(vs=[grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]]))
    # Cap the two flat faces (triangle fans from the centre) so the dome is solid.
    phis = [pi * j / slices for j in range(slices + 1)]
    #  - flat half-disc in the y = center.y plane (faces away from the dome body)
    nb = vector(0, -ysign, 0)
    cb = vertex(pos=center, normal=nb, color=col)
    rb = [vertex(pos=center + radius * vector(cos(p), 0, zsign * sin(p)), normal=nb, color=col)
          for p in phis]
    for j in range(slices):
        faces.append(triangle(vs=[cb, rb[j], rb[j + 1]]))
    #  - vertical half-disc in the z = center.z plane (faces the central gap)
    nv = vector(0, 0, -zsign)
    cv = vertex(pos=center, normal=nv, color=col)
    rv = [vertex(pos=center + radius * vector(cos(p), ysign * sin(p), 0), normal=nv, color=col)
          for p in phis]
    for j in range(slices):
        faces.append(triangle(vs=[cv, rv[j], rv[j + 1]]))
    return faces

def _make_arm_base(anchor, col):
    """Build the visual arm base centred on `anchor`: a 0.05-high, 0.25-radius disc with two
    radius-0.1 quarter-sphere domes whose flat faces are separated by 0.1 (they bulge outward
    in z). The whole thing is merged into ONE compound and flipped upside down, so the disc
    and domes both point down. Returned as the robot's 'armbase'. Static geometry — nothing
    about the physics/anchor changes."""
    disc = cylinder(pos=anchor - vector(0, 0.025, 0), axis=vector(0, 0.05, 0),
                    radius=0.25, color=col)
    top = anchor + vector(0, 0.025, 0)           # disc top surface = the domes' flat bottom
    parts = [disc]
    parts += _quarter_dome(top + vector(0, 0, 0.05), 0.1, col, +1)   # flat faces 0.1 apart,
    parts += _quarter_dome(top + vector(0, 0, -0.05), 0.1, col, -1)  # domes bulge outward (±z)
    base = compound(parts, origin=anchor)
    base.rotate(angle=pi, axis=vector(1, 0, 0), origin=anchor)       # flip the whole base upside down
    return base

def _make_joint(center, col):
    """The elbow joint (replaces the old radius-0.25 sphere between arm 1 and arm 2): two
    arm-bases stacked back to back, i.e. their disc faces touching. A single 0.1-high,
    0.25-radius disc (the two 0.05 discs merged) carries a pair of radius-0.1 quarter-domes
    on its TOP face bulging up (+y, flat faces 0.1 apart in ±z) and a mirrored pair on its
    BOTTOM face bulging down. Merged into ONE compound centred on `center` so it moves
    rigidly with the elbow. Purely visual — the physics/joint position is unchanged."""
    disc = cylinder(pos=center - vector(0, 0.05, 0), axis=vector(0, 0.1, 0),
                    radius=0.25, color=col)          # two 0.05-high discs, back to back
    top = center + vector(0, 0.05, 0)                # upper disc face -> up-domes sit here
    bot = center - vector(0, 0.05, 0)                # lower disc face -> down-domes sit here
    parts = [disc]
    parts += _quarter_dome(top + vector(0, 0, 0.05), 0.1, col, +1, ysign=+1)
    parts += _quarter_dome(top + vector(0, 0, -0.05), 0.1, col, -1, ysign=+1)
    parts += _quarter_dome(bot + vector(0, 0, 0.05), 0.1, col, +1, ysign=-1)
    parts += _quarter_dome(bot + vector(0, 0, -0.05), 0.1, col, -1, ysign=-1)
    return compound(parts, origin=center)

def _half_cylinder(center, radius, length, col, xsign, slices=8):
    """A half of a cylinder (axis along z, given `length` and radius) whose curved shell
    bulges toward xsign*x and whose FLAT cut face is on the -xsign side — built as a mesh
    (curved shell + two semicircular end caps + the flat rectangular cut face). Purely
    visual. Returns the list of mesh faces so the caller can merge them into a compound."""
    faces = []
    z0, z1 = center.z - length / 2, center.z + length / 2
    a0 = -pi / 2 if xsign > 0 else pi / 2         # sweep the half-circle on the xsign side
    angs = [a0 + pi * j / slices for j in range(slices + 1)]
    def arc(z):                                   # rim vertices with radial-outward normals
        return [vertex(pos=vector(center.x + radius * cos(a), center.y + radius * sin(a), z),
                       normal=vector(cos(a), sin(a), 0), color=col) for a in angs]
    lo, hi = arc(z0), arc(z1)
    for j in range(slices):                       # curved shell
        faces.append(quad(vs=[lo[j], lo[j + 1], hi[j + 1], hi[j]]))
    for z, nz in ((z0, -1), (z1, 1)):             # flat semicircular end caps
        cen = vertex(pos=vector(center.x, center.y, z), normal=vector(0, 0, nz), color=col)
        rim = [vertex(pos=vector(center.x + radius * cos(a), center.y + radius * sin(a), z),
                      normal=vector(0, 0, nz), color=col) for a in angs]
        for j in range(slices):
            faces.append(triangle(vs=[cen, rim[j], rim[j + 1]]))
    # Flat rectangular cut face (the flat end), facing away from the curved side.
    nf = vector(-1 if xsign > 0 else 1, 0, 0)
    y0, y1 = center.y - radius, center.y + radius
    faces.append(quad(vs=[
        vertex(pos=vector(center.x, y0, z0), normal=nf, color=col),
        vertex(pos=vector(center.x, y1, z0), normal=nf, color=col),
        vertex(pos=vector(center.x, y1, z1), normal=nf, color=col),
        vertex(pos=vector(center.x, y0, z1), normal=nf, color=col)]))
    return faces

def _make_arm1(anchor, col):
    """Build the arm-1 visual: a 1.9-long, ARM_WIDTH-wide bar with a radius-0.1 half-
    cylinder (mesh) rounding each end — so it fills the length-2 arm with a 0.05 gap at each
    end. Returned as ONE compound so it moves rigidly with the arm exactly like the old box.
    The old box is length 2; this compound is length 1.9, so the per-frame visual update
    rescales its axis back to 1.9 (see step_robot / init_robot_state). Physics unchanged."""
    # Built centred on the anchor along +x, total x-extent [-0.95, 0.95]; positioned per
    # frame. The bar stops just short (±0.93) so its end hides inside the caps (no z-fight);
    # each cap's flat face is the very tip (±0.95) with the curved shoulder bulging inward.
    parts = [box(pos=anchor, size=vector(1.86, 0.1, ARM_WIDTH), color=col)]
    parts += _half_cylinder(anchor + vector(0.95, 0, 0), 0.1, ARM_WIDTH, col, -1)   # right end
    parts += _half_cylinder(anchor + vector(-0.95, 0, 0), 0.1, ARM_WIDTH, col, +1)  # left end
    return compound(parts, origin=anchor)

def _make_arm2(anchor, col):
    """Build the arm-2 visual: a 1.75-long, ARM_WIDTH-wide bar with a radius-0.1 half-
    cylinder (mesh) rounding the ELBOW end and a flat far (wrist) end, set 0.05 off the
    elbow joint (a standoff). Returned as ONE compound so it moves rigidly with the arm
    exactly like the old box. The old box is length ARM2_LEN=1.8; this compound is length
    1.75, so the per-frame visual update rescales its axis back to 1.75 (see step_robot /
    init_robot_state). Its origin sits at the arm midpoint (ARM2_LEN/2 from the elbow) so
    apply_base_rotation — which places the origin at anchor + (L/2)·dir — still positions it
    correctly even though the shape is asymmetric. Physics unchanged."""
    # Built along +x from the elbow `anchor`. A 0.05 standoff leaves [0, 0.05] empty; the
    # shape then spans [0.05, 1.80] (length 1.75): a rounded elbow cap (flat face at 0.05
    # facing the joint, curved shoulder bulging into the bar) and a flat far end at 1.80.
    # The bar is tucked 0.02 under the cap's flat face so it can't z-fight it.
    bar_lo, bar_hi = 0.07, 1.80
    parts = [box(pos=anchor + vector((bar_lo + bar_hi) / 2, 0, 0),
                 size=vector(bar_hi - bar_lo, 0.1, ARM_WIDTH), color=col)]
    parts += _half_cylinder(anchor + vector(0.05, 0, 0), 0.1, ARM_WIDTH, col, +1)  # elbow cap
    return compound(parts, origin=anchor + vector(ARM2_LEN / 2, 0, 0))

def make_robot(base_offset, color1, color2=None):
    if color2 is None:
        color2 = color1        # same color for both arm segments of this robot
    anchor = base_offset
    armbase = _make_arm_base(anchor, color1)
    arm = _make_arm1(anchor, color1)
    arm_state = {"angle": 0, "angular_velocity": initial_av, "mass": 1,
                 "anchor": anchor, "length": ARM1_LEN, "width": ARM_WIDTH}

    tip = anchor + vector(ARM1_LEN, 0, 0)
    armbase2 = _make_joint(tip, color2)
    arm2 = _make_arm2(tip, color2)
    arm2_state = {"angle": 0, "angular_velocity": initial_av, "mass": 1,
                  "anchor": tip, "length": ARM2_LEN, "width": ARM_WIDTH}

    finger1 = box(pos=tip + vector(ARM2_LEN + FINGER_LEN/2, 0, FINGER_LEN/2),
                  axis=vector(FINGER_LEN, 0, 0), size=vector(FINGER_LEN, 0.08, 0.08), color=color.yellow)
    finger2 = box(pos=tip + vector(ARM2_LEN + FINGER_LEN/2, 0, -FINGER_LEN/2),
                  axis=vector(FINGER_LEN, 0, 0), size=vector(FINGER_LEN, 0.08, 0.08), color=color.yellow)
    tooth1 = box(pos=tip + vector(ARM2_LEN + FINGER_LEN, 0, FINGER_LEN/2),
                 axis=vector(0, -TOOTH_LEN, 0), size=vector(TOOTH_LEN, 0.08, 0.08), color=color.yellow)
    tooth2 = box(pos=tip + vector(ARM2_LEN + FINGER_LEN, 0, -FINGER_LEN/2),
                 axis=vector(0, TOOTH_LEN, 0), size=vector(TOOTH_LEN, 0.08, 0.08), color=color.yellow)

    base_state = {"angle": 0, "angular_velocity": 0, "mass": 3, "length": 3, "width": 0.2}
    claw_state = {"angle": pi/2,    # PID-commanded angle both fingers aim for
                  "angle1": pi/2,   # actual angle of finger 1
                  "angle2": pi/2,   # actual angle of finger 2
                  "angular_velocity": 0, "mass": 0.1, "length": 0.4, "width": 0.08}

    robot = {
        "anchor": anchor,
        "armbase_angle": 0.0,   # base rotation currently applied to the base-visual compound
        "arm": arm, "arm_state": arm_state, "armbase": armbase,
        "arm2": arm2, "arm2_state": arm2_state, "armbase2": armbase2,
        "finger1": finger1, "finger2": finger2, "tooth1": tooth1, "tooth2": tooth2,
        "base_state": base_state, "claw_state": claw_state,
        "ball_grabbed": False, "grip_angle": None, "grip_local": None,
        "held": None,           # the object this arm currently holds, or None
        "empty_closed": 0,      # frames the claw has been shut-and-empty (auto-catch retry)
        "close_lock": 0,        # frames left in the just-closed hold-shut lockout (no reopen)
        "claw_latched_closing": False,  # True while the claw travels shut: blocks any reopen
                                        # until it bottoms out (see close_claw/open_claw)
        "catch_timer": 0,       # frames parked-but-not-centred (auto-catch close timeout)
        "plan_obj": None,       # the tracked object the current wait point was planned for
        "plan_vel": None,       # its velocity when the wait point was last computed
        "plan_camp": None,      # the cached wait point (re-planned only on a velocity change)
        "ignore_ball": False,   # when True, this arm doesn't collide with any object
        "manager": ControllerManager(), "manager2": ControllerManager(),
        "manager_base": ControllerManager(), "manager_claw": ControllerManager(),
        "base_target_deg": 0.0, "arm1_target_deg": 0.0,
        "coord_input": None, "base_label": None,   # set when the UI is built
    }
    init_robot_state(robot)   # put the arm in its rest pose (shared with reset)
    return robot

def _store_box_in_claw(robot, arm2_state, base_angle, ball):
    """Record the box's position in the claw's frame at the moment of grip."""
    tip, ad, up, side = claw_frame(arm2_state, base_angle)
    rel = ball.pos - tip
    robot["grip_local"] = vector(dot(rel, ad), dot(rel, up), dot(rel, side))

def _carry_box_in_claw(robot, arm2_state, base_angle, ball):
    """Rigidly reposition the box to its stored offset in the claw's frame."""
    tip, ad, up, side = claw_frame(arm2_state, base_angle)
    L = robot["grip_local"]
    new_pos = tip + ad * L.x + up * L.y + side * L.z
    ball.velocity = (new_pos - ball.pos) / dt   # so it keeps moving when released
    ball.pos = new_pos

def step_robot(robot):
    arm, arm_state = robot["arm"], robot["arm_state"]
    arm2, arm2_state = robot["arm2"], robot["arm2_state"]
    base_state, claw_state = robot["base_state"], robot["claw_state"]

    if robot["close_lock"] > 0:        # count down the just-closed hold-shut lockout
        robot["close_lock"] -= 1

    # Arm 1
    drag(arm, arm_state)
    robot["manager"].step()
    update_arm(arm, arm_state)

    # Arm 2
    drag(arm2, arm2_state)
    robot["manager2"].step()
    update_arm(arm2, arm2_state)

    # Base rotation (no gravity torque, just drag + PID)
    drag(arm, base_state)
    robot["manager_base"].step()
    update_base(base_state)

    # Spin the base-pedestal visual to match the base heading (about the vertical axis
    # through the anchor). Applied as the delta since last frame so it never scales.
    d_base = base_state["angle"] - robot["armbase_angle"]
    if d_base:
        robot["armbase"].rotate(angle=d_base, axis=vector(0, -1, 0), origin=robot["anchor"])
        robot["armbase_angle"] = base_state["angle"]

    # Override visual positions with 3D base-rotated versions
    apply_base_rotation(arm, arm_state, base_state["angle"])
    arm.axis = arm.axis.hat * 1.9        # arm-1 visual is a 1.9-long compound (0.05 end gaps)
    tip = arm_tip_3d(arm_state, base_state["angle"])
    arm2_state["anchor"] = tip
    robot["armbase2"].pos = tip
    apply_base_rotation(arm2, arm2_state, base_state["angle"])
    arm2.axis = arm2.axis.hat * 1.75     # arm-2 visual is a 1.75-long compound (0.05 standoff)

    prev = claw_state["angle1"]   # both fingers share one closing angle
    robot["manager_claw"].step()
    update_claw(claw_state)
    cmd = claw_state["angle"]     # PID-commanded angle the fingers aim for

    # Release the closing latch once the claw has FINISHED closing (bottomed out) or has
    # gripped something — only then may a reopen (retry / release) happen. While it's still
    # travelling shut the latch stays set, so open_claw can't reverse it mid-close.
    if robot["claw_latched_closing"] and (cmd <= CLAW_SHUT_ANGLE
            or robot["grip_angle"] is not None or robot["ball_grabbed"]):
        robot["claw_latched_closing"] = False

    held = robot["held"]
    if robot["ball_grabbed"] and held is not None:
        # This arm button-holds an object — carry it at the grab point
        held.pos = claw_grab_point(arm2_state, base_state["angle"])
        angle = cmd
    elif robot["grip_angle"] is not None and held is not None:
        # Firmly gripping the object: keep the fingers locked on it and hold it
        # rigidly in the claw's frame, so it follows the arm securely. Release
        # once the claw is commanded open past the grip.
        grip = robot["grip_angle"]
        if cmd > grip + 0.05:
            robot["grip_angle"] = None
            robot["grip_local"] = None
            held.held_by = None
            robot["held"] = None
            angle = cmd
        else:
            angle = grip
            _carry_box_in_claw(robot, arm2_state, base_state["angle"], held)
    elif robot["ignore_ball"]:
        # This arm has just let go and is moving away — don't touch any object at
        # all, so opening/rotating can't drag or fling one.
        angle = cmd
    else:
        # Not holding anything — try to close on the first FREE object (one not
        # already held by another arm). Keep shoving it toward the centre via the
        # collision push-out until BOTH fingers press it; at that moment lock the
        # grip and capture the object's pose so it can be held rigidly from here on.
        angle = cmd
        # Grip whenever both fingers press the box and the claw is NOT opening (cmd <= prev)
        # while it's shut enough for a real grasp (cmd < CATCH_OPEN_ANGLE). This covers both
        # the active close (cmd < prev) AND the bottomed-out shut claw (cmd ≈ prev at the
        # clamp), so a box that seats a moment after the fingers bottom out is still gripped.
        grippable = cmd <= prev and cmd < CATCH_OPEN_ANGLE
        finger_av = (angle - prev) / dt
        for obj in balls:
            if obj.held_by is not None or obj.delivered:
                continue                                   # held by another arm, or already delivered
            if (grippable
                    and finger_contacts_ball(arm2_state, cmd, base_state["angle"], obj, 1)
                    and finger_contacts_ball(arm2_state, cmd, base_state["angle"], obj, -1)):
                robot["grip_angle"] = cmd
                robot["held"] = obj
                obj.held_by = robot
                _store_box_in_claw(robot, arm2_state, base_state["angle"], obj)
                break
            resolve_finger_ball_collision(arm_state, arm2_state, base_state, angle, finger_av, obj, 1)
            resolve_finger_ball_collision(arm_state, arm2_state, base_state, angle, finger_av, obj, -1)

    claw_state["angle1"] = angle
    claw_state["angle2"] = angle
    apply_claw(robot["finger1"], robot["finger2"], robot["tooth1"], robot["tooth2"],
               arm2_state, angle, angle, base_state["angle"])

def init_robot_state(robot):
    """Put a robot into its rest pose — the single source of truth for a robot's
    initial/reset state. Called once when the robot is built and again by reset."""
    anchor = robot["anchor"]
    arm_state, arm2_state = robot["arm_state"], robot["arm2_state"]
    base_state, claw_state = robot["base_state"], robot["claw_state"]

    arm_state["angle"] = 0
    arm_state["angular_velocity"] = initial_av
    arm2_state["angle"] = 0
    arm2_state["angular_velocity"] = initial_av
    arm2_state["anchor"] = anchor + vector(arm_state["length"], 0, 0)

    robot["manager"].set_controller(DoNothingController(arm_state))
    robot["manager2"].set_controller(DoNothingController(arm2_state))
    robot["manager_base"].set_controller(DoNothingController(base_state))
    base_state["angle"] = 0
    base_state["angular_velocity"] = 0
    # Return the base-pedestal visual to its rest heading (undo any applied spin).
    if robot["armbase_angle"]:
        robot["armbase"].rotate(angle=-robot["armbase_angle"], axis=vector(0, -1, 0),
                                origin=robot["anchor"])
        robot["armbase_angle"] = 0.0

    claw_state["angle"] = pi/2
    claw_state["angle1"] = pi/2
    claw_state["angle2"] = pi/2
    claw_state["angular_velocity"] = 0
    robot["manager_claw"].set_controller(DoNothingController(claw_state))

    robot["ball_grabbed"] = False
    robot["grip_angle"] = None
    robot["grip_local"] = None
    robot["held"] = None
    robot["empty_closed"] = 0
    robot["close_lock"] = 0
    robot["claw_latched_closing"] = False
    robot["catch_timer"] = 0
    robot["plan_obj"] = None
    robot["plan_vel"] = None
    robot["plan_camp"] = None
    robot["drive_path"] = None
    robot["ignore_ball"] = False
    robot["base_target_deg"] = 0.0
    robot["arm1_target_deg"] = 0.0

    # Snap the visuals back to the rest pose
    apply_base_rotation(robot["arm"], arm_state, 0)
    robot["arm"].axis = robot["arm"].axis.hat * 1.9   # keep the 1.9-long arm-1 compound
    tip = arm_tip_3d(arm_state, 0)
    arm2_state["anchor"] = tip
    robot["armbase2"].pos = tip
    apply_base_rotation(robot["arm2"], arm2_state, 0)
    robot["arm2"].axis = robot["arm2"].axis.hat * 1.75   # keep the 1.75-long arm-2 compound
    apply_claw(robot["finger1"], robot["finger2"], robot["tooth1"], robot["tooth2"],
               arm2_state, pi/2, pi/2, 0)

# ─────────────────────────────────────────────
#  Controls — these drive EVERY robot in `robots`
# ─────────────────────────────────────────────

def _respawn(spawn_fn):
    """Shared reset: clear all in-flight tasks/grips, respawn the boxes with `spawn_fn`,
    and drop stale tracks (the boxes teleport)."""
    global claw_open, locate_task, pathfind_task
    handoff_tasks.clear()
    locate_task = None
    pathfind_task = None
    for robot in robots:
        init_robot_state(robot)
        if robot["coord_input"] is not None:      # delivery-only arms have no input
            robot["coord_input"].text = '1,0,0'
            robot["base_label"].text = ' 0.000 rad'
    spawn_fn()
    tracker.reset()                               # drop stale tracks (ball teleported)
    claw_open = True
    claw_btn.text = 'Close Claw'
    grab_btn.text = 'Grab'

def reset(b):
    _respawn(spawn_all)                            # scattered boxes + a cluster near the arms

# Starting point for tuning the arm joints. Gravity is now cancelled by the
# controller's feed-forward, so ki no longer has to hold the arm up against
# droop — it's much smaller than before (was 5.5) to avoid overshoot/windup.
def _set_pid(mgr, state, angle, kp=30, ki=0.5, kd=14):
    c = mgr.active
    if isinstance(c, PIDController):
        c.target_angle = angle
    else:
        mgr.set_controller(
            PIDController(state, target_angle=angle, kp=kp, ki=ki, kd=kd, gravity_ff=True))

def set_base_position(robot, x_val, y_val):
    target_angle = translate_degree(x_val, y_val)
    robot["base_target_deg"] = degrees(target_angle)
    c = robot["manager_base"].active
    if isinstance(c, CircularController):
        c.target_angle = target_angle
    else:
        robot["manager_base"].set_controller(CircularController(
            robot["base_state"], target_angle=target_angle,
            kp=20, ki=0.0, kd=27, circular=True))
    return target_angle

def set_arm_angles_from_position(robot, x_val, y_val, z_val):
    arm1_angle, arm2_angle = caculate_target_arm_angles(x_val, y_val, z_val, robot["arm_state"]["length"])
    robot["arm1_target_deg"] = degrees(arm1_angle)
    _set_pid(robot["manager"], robot["arm_state"], arm1_angle)
    _set_pid(robot["manager2"], robot["arm2_state"], arm2_angle)

def _drive_to_shared(robot, x, y, z):
    """Aim `robot` at shared control coords (x, y, z), clamped. The shared frame is
    the first arm's; each arm subtracts its base offset so all reach the same world
    point. Control axes map to world as (x, y, z) -> (worldX, worldZ, worldY)."""
    x = max(-6.0, min(6.0, x))
    y = max(-6.0, min(10.0, y))    # world Z: reach out to the 3rd arm (z=6) and beyond
    z = max(-4.0, min(4.0, z))
    off = robot["anchor"] - robots[0]["anchor"]
    lx = x - off.x
    ly = y - off.z
    lz = z - off.y
    target_angle = set_base_position(robot, lx, ly)
    set_arm_angles_from_position(robot, lx, ly, lz)
    if robot["base_label"] is not None:
        robot["base_label"].text = f' {target_angle:.3f} rad'
    return target_angle

def input_coordinates(w, robot):
    try:
        parts = [float(p.strip()) for p in w.text.split(',')]
        if len(parts) == 3:
            _drive_to_shared(robot, parts[0], parts[1], parts[2])
    except ValueError:
        pass

def toggle_claw(b):
    global claw_open
    if claw_open:
        claw_open = False
        b.text = 'Open Claw'
        target = 0.0
    else:
        claw_open = True
        b.text = 'Close Claw'
        target = pi/2
    for robot in robots:
        robot["manager_claw"].set_controller(
            PIDController(robot["claw_state"], target_angle=target, kp=8, ki=0.2, kd=0.5))

def toggle_grab(b):
    if any(r["ball_grabbed"] for r in robots):
        # Release: every arm lets go of whatever it holds
        for robot in robots:
            robot["ball_grabbed"] = False
            if robot["held"] is not None:
                robot["held"].held_by = None
                robot["held"] = None
            robot["grip_angle"] = None
            robot["grip_local"] = None
        b.text = 'Grab'
    else:
        # The first arm whose claw is near a free object grabs it
        for robot in robots:
            grab_pt = claw_grab_point(robot["arm2_state"], robot["base_state"]["angle"])
            for obj in balls:
                if obj.held_by is None and mag(obj.pos - grab_pt) <= obj.radius + 0.5:
                    robot["ball_grabbed"] = True
                    robot["held"] = obj
                    obj.held_by = robot
                    b.text = 'Release'
                    break
            if robot["ball_grabbed"]:
                break

LOCATE_STANDOFF_ANGLE = pi / 8   # base rotation of the standoff away from the box (22.5°)
locate_task = None               # active "move to ball" sequence, or None

def _rotated_standoff(robot, target, angle):
    """A point at the same reach and height as `target`, but with the base rotated
    by `angle` around the vertical. The arm reaches out to the side of the box,
    then swings its base onto it — avoiding a straight-in move through the box."""
    rel = target - robot["anchor"]
    rh = hypot(rel.x, rel.z)                    # horizontal reach
    theta = atan2(rel.z, rel.x) + angle         # rotate the approach direction
    return robot["anchor"] + vector(rh * cos(theta), rel.y, rh * sin(theta))

GRAB_STANDOFF = 0.1   # pull grab targets this much toward the base so the arm stops
                      # a touch short and the cube seats ~0.3 out (by the teeth)

def _grab_target(robot, point):
    """Pull a grab target toward the arm's base by GRAB_STANDOFF. The arm then
    reaches slightly less far, so when the fingers lock the cube sits out near the
    teeth instead of jammed in the corner at the finger base. Grab moves only —
    delivery still commands the exact target."""
    d = point - robot["anchor"]
    m = mag(d)
    if m <= GRAB_STANDOFF:
        return vector(point.x, point.y, point.z)
    return point - d.hat * GRAB_STANDOFF

def _tracked_estimate(point):
    """Best (position, velocity) estimate from the multi-object tracker for the
    confirmed track nearest `point`. Falls back to the raw single-object
    trilateration + finite-difference velocity when no track is confirmed yet
    (e.g. the first few frames after a spawn/reset)."""
    tracks = tracker.confirmed_tracks()
    if not tracks:
        return vector(*trilateration(sensors, ball)), ball_velocity_est
    best = min(tracks, key=lambda tr: mag(vector(*tr.position) - point))
    return vector(*best.position), vector(*best.velocity)

def _predict_intercept(delay):
    """Solve for where to meet the moving ball. The travel time depends on the
    intercept point, but the intercept point (ball position + vel·t) depends on
    the travel time — so this is a fixed point of  t = travel_time(pos + vel·t).
    We iterate a few times to converge instead of guessing once (which projects
    a far/fast ball wildly off). Returns (robot, aim_point).

    Position and velocity come from the tracker's confirmed track for the object
    (smoother than the raw per-frame trilateration/finite-difference)."""
    ref = vector(*trilateration(sensors, ball))     # which object we're after (the ball)
    pos, vel = _tracked_estimate(ref)               # filtered tracker estimate
    aim = vector(pos.x, pos.y, pos.z)
    robot = min(robots, key=lambda r: mag(aim - r["anchor"]))
    t = delay
    for _ in range(5):                               # converges in a few iterations
        aim = pos + vel * t                          # where the ball will be in t seconds
        robot = min(robots, key=lambda r: mag(aim - r["anchor"]))
        t = estimate_travel_time(robot, aim) + delay # ...and how long that arm takes to get there
    return robot, aim

def move_to_ball(b):
    global locate_task
    # Infer the ball's position, solve the intercept for the arm that will take
    # it, and approach in two steps so the arm doesn't knock the ball.
    robot, pos = _predict_intercept(delay=2.5)
    # First go to a standoff rotated away from the box, then swing onto it
    command_arm_to(robot, _rotated_standoff(robot, pos, LOCATE_STANDOFF_ANGLE))
    locate_task = {"robot": robot, "target": pos, "stage": "standoff"}

def catch_ball(b):
    # Like "Locate Ball" but goes straight to the (lead-predicted) intercept
    # point in one move — no standoff — to catch a moving ball.
    global locate_task
    locate_task = None                              # cancel any two-phase approach
    robot, pos = _predict_intercept(delay=2.5)

    command_arm_to(robot, _grab_target(robot, pos)) # go straight to the intercept point

# ── Ball delivery / hand-off between arms ─────────────────────────────
DELIVER_TARGET = vector(3, 0, 0)   # fixed drop point; boxes are passed along the grid to
                                   # the arm nearest it, which places them here
GRID_DIAG_TOL = 0.5       # a neighbour counts as axis-aligned only if its off-axis
                          # spread is under this fraction of its on-axis distance
                          # (bigger = tolerate a more crooked grid; >1 admits diagonals)
APPROACH_TOL = 0.12       # receiver must get this close to the ball...
APPROACH_DWELL = 30       # ...and hold it this many frames (settled) before closing
APPROACH_TIMEOUT = 500    # safety cap for the giver's stages (not the receiver's grab)
GIVER_OPEN_ANGLE = 1.2    # claw angle at which the giver's fingers are clear of the ball
GIVER_CLEAR_DIST = 1.0    # giver must swing this far from the ball before the receiver picks up
SETTLE_AV = 0.15          # a joint counts as "still" when its angular speed is below this
SETTLE_DWELL = 20         # frames the giver must hold still before it releases the ball
HANDOFF_TIMEOUT = 1200    # hard cap on a whole delivery cycle — abort so arms never freeze
handoff_tasks = []                       # active delivery cycles (run concurrently); each is
                                         # one box's hand-off, on its own disjoint set of arms

def command_arm_to(robot, world_pos):
    """Drive one arm so its claw reaches `world_pos` (world coordinates)."""
    ref = robots[0]["anchor"]
    cx = world_pos.x - ref.x
    cy = world_pos.z - ref.z
    cz = world_pos.y - ref.y
    _drive_to_shared(robot, cx, cy, cz)
    if robot["coord_input"] is not None:
        robot["coord_input"].text = f'{cx:.3f},{cy:.3f},{cz:.3f}'

# ── Arm sweep: the volume the arm passes through on its way to a target ─────────
def _dir_at(angle, base_angle):
    """Unit direction of a link raised by `angle` and rotated about the base by
    `base_angle` (matches physics._arm_dir)."""
    fwd = vector(cos(base_angle), 0, sin(base_angle))
    return fwd * cos(angle) + vector(0, 1, 0) * sin(angle)

def _target_config(robot, world_pos):
    """The (base, arm1, arm2) joint angles this arm would drive to for `world_pos` —
    the same IK as command_arm_to/_drive_to_shared, but computed without moving anything."""
    ref = robots[0]["anchor"]
    cx = max(-6.0, min(6.0, world_pos.x - ref.x))
    cy = max(-6.0, min(10.0, world_pos.z - ref.z))
    cz = max(-4.0, min(4.0, world_pos.y - ref.y))
    off = robot["anchor"] - ref
    lx, ly, lz = cx - off.x, cy - off.z, cz - off.y
    base = translate_degree(lx, ly)
    a1, a2 = caculate_target_arm_angles(lx, ly, lz, robot["arm_state"]["length"])
    return base, a1, a2

def _arm_skeleton(robot, base, a1, a2):
    """World points the arm occupies at joint config (base, a1, a2):
    [anchor, elbow, wrist, claw grab point]."""
    anchor = robot["anchor"]
    d1 = _dir_at(a1, base)
    elbow = anchor + robot["arm_state"]["length"] * d1
    d2 = _dir_at(a2, base)
    wrist = elbow + robot["arm2_state"]["length"] * d2
    grab = wrist + 0.3 * d2                       # claw grab point (see claw_grab_point)
    return [anchor, elbow, wrist, grab]

def arm_sweep_poses(robot, world_pos, steps=12):
    """The arm skeletons sampled along the joint interpolation from the arm's CURRENT
    pose to the pose that reaches `world_pos`. The union of these is the volume the arm
    passes through on the way — use it for collision checks (e.g. arm-vs-arm). Returns a
    list of poses, each [anchor, elbow, wrist, grab] in world coordinates."""
    b0, a10, a20 = (robot["base_state"]["angle"], robot["arm_state"]["angle"],
                    robot["arm2_state"]["angle"])
    b1, a11, a21 = _target_config(robot, world_pos)
    db = (b1 - b0 + pi) % (2 * pi) - pi           # base wraps — take the short way round
    poses = []
    for i in range(steps + 1):
        t = i / steps
        poses.append(_arm_skeleton(robot, b0 + db * t,
                                   a10 + (a11 - a10) * t, a20 + (a21 - a20) * t))
    return poses

def draw_arm_sweep(robot, world_pos, steps=12):
    """Draw the arm's swept path to `world_pos` as INVISIBLE curves — one arm silhouette
    per interpolation sample, so the union traces the volume the arm will occupy. The
    curves are hidden (for collision use / optional debugging); reuse a per-robot pool so
    repeated calls don't leak objects. Returns the poses (see arm_sweep_poses). Call
    show_arm_sweep(robot) to reveal it."""
    poses = arm_sweep_poses(robot, world_pos, steps)
    curves = robot.get("sweep_curves")
    if curves is None or len(curves) != len(poses):
        for c in curves or []:
            c.clear(); c.visible = False
        curves = [curve(color=color.white, radius=0.015, visible=False) for _ in poses]
        robot["sweep_curves"] = curves
    for c, pose in zip(curves, poses):
        c.clear()
        c.append(pose)
        c.visible = False
    return poses

def show_arm_sweep(robot, visible=True):
    """Reveal or hide a robot's drawn sweep (created by draw_arm_sweep)."""
    for c in robot.get("sweep_curves") or []:
        c.visible = visible

SWEEP_ARM_RADIUS = ARM_WIDTH / 2   # arm links AND claw fingers/teeth treated as capsules of
                                   # this radius for sweep collision (the arm is ARM_WIDTH
                                   # thick; the ~0.08 fingers are a hair thinner, so this is
                                   # slightly conservative — fine for a collision bound)
SWEEP_JOINT_RADIUS = 0.25          # base and elbow joint spheres (match the armbase spheres)
SWEEP_MARGIN = 0.05                # extra clearance added to every sweep-collision threshold,
                                   # so the swept arm/claw keeps a little distance from objects
                                   # and other arms instead of only just grazing them

def _up_perp_from(arm_dir):
    """The claw's up-perpendicular (in the arm's vertical plane) from its axis direction:
    perpendicular to arm_dir, toward +y. Its SIGN is irrelevant to the swept claw (both
    fingers, ±up_perp, are included). Falls back to a horizontal perpendicular if the arm
    points straight up or down."""
    up = vector(0, 1, 0) - arm_dir * dot(vector(0, 1, 0), arm_dir)
    if mag(up) < 1e-6:                            # arm vertical — any horizontal perp will do
        up = cross(arm_dir, vector(1, 0, 0))
        if mag(up) < 1e-6:
            up = cross(arm_dir, vector(0, 0, 1))
    return up.hat

def _claw_segments(tip, arm_dir, up_perp, fa):
    """The claw's four capsule segments at finger opening `fa`: each of the two fingers
    (from the wrist `tip` out FINGER_LEN, splayed ±up_perp) and its tooth (hooking
    TOOTH_LEN in from the finger tip). Matches apply_claw / finger_contacts_ball exactly,
    so the swept volume is the REAL claw — including the finger tips that splay well outside
    a bounding sphere when the claw is open. Returns a list of (a, b) vector pairs."""
    ca, sa = cos(fa), sin(fa)
    segs = []
    for sign in (1, -1):
        fd = arm_dir * ca + up_perp * (sign * sa)
        ftip = tip + fd * FINGER_LEN
        td = arm_dir * sa - up_perp * (sign * ca)
        segs.append((tip, ftip))
        segs.append((ftip, ftip + td * TOOTH_LEN))
    return segs

def _seg_point_dist(a, b, p):
    """Shortest distance from point `p` to segment `a`-`b`."""
    ab = b - a
    denom = dot(ab, ab)
    if denom < 1e-12:
        return mag(p - a)
    t = max(0.0, min(1.0, dot(p - a, ab) / denom))
    return mag(p - (a + ab * t))

def _seg_seg_dist(p1, q1, p2, q2):
    """Shortest distance between segments p1-q1 and p2-q2 in 3D (clamped closest
    points; Ericson, Real-Time Collision Detection)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = dot(d1, d1), dot(d2, d2), dot(d2, r)
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return mag(r)
    if a <= EPS:
        s, t = 0.0, max(0.0, min(1.0, f / e))
    else:
        c = dot(d1, r)
        if e <= EPS:
            t, s = 0.0, max(0.0, min(1.0, -c / a))
        else:
            b = dot(d1, d2)
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, max(0.0, min(1.0, (b - c) / a))
    return mag((p1 + d1 * s) - (p2 + d2 * t))

def _skeleton_segments(skel):
    """The three arm links of a skeleton [anchor, elbow, wrist, grab] as (start, end)
    pairs: arm1, arm2, and the claw."""
    return [(skel[0], skel[1]), (skel[1], skel[2]), (skel[2], skel[3])]

def _current_skeleton(robot):
    """The arm's skeleton points at its CURRENT joint angles."""
    return _arm_skeleton(robot, robot["base_state"]["angle"],
                         robot["arm_state"]["angle"], robot["arm2_state"]["angle"])

# Fast raw-float collision primitives used by sweep_overlaps' inner loops. They take
# plain (x, y, z) tuples and return SQUARED distance, so the caller compares against a
# squared threshold — same accept/reject as the vector-based _seg_*_dist, no sqrt and no
# VPython-vector object churn in the hot path.
def _seg_point_d2(ax, ay, az, bx, by, bz, px, py, pz):
    """Squared distance from point (px,py,pz) to segment (a)-(b)."""
    abx, aby, abz = bx - ax, by - ay, bz - az
    den = abx * abx + aby * aby + abz * abz
    if den < 1e-12:
        dx, dy, dz = px - ax, py - ay, pz - az
        return dx * dx + dy * dy + dz * dz
    t = ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / den
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    dx, dy, dz = px - (ax + abx * t), py - (ay + aby * t), pz - (az + abz * t)
    return dx * dx + dy * dy + dz * dz

def _seg_seg_d2(p1, q1, p2, q2):
    """Squared shortest distance between segments p1-q1 and p2-q2 (tuples); same clamped
    closest-point logic as _seg_seg_dist."""
    d1x, d1y, d1z = q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2]
    d2x, d2y, d2z = q2[0] - p2[0], q2[1] - p2[1], q2[2] - p2[2]
    rx, ry, rz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return rx * rx + ry * ry + rz * rz
    if a <= EPS:
        s = 0.0
        t = f / e
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    else:
        c = d1x * rx + d1y * ry + d1z * rz
        if e <= EPS:
            t = 0.0
            s = -c / a
            s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        else:
            b = d1x * d2x + d1y * d2y + d1z * d2z
            denom = a * e - b * b
            s = (b * f - c * e) / denom if denom > EPS else 0.0
            s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = -c / a
                s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
            elif t > 1.0:
                t = 1.0
                s = (b - c) / a
                s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    cx = (p1[0] + d1x * s) - (p2[0] + d2x * t)
    cy = (p1[1] + d1y * s) - (p2[1] + d2y * t)
    cz = (p1[2] + d1z * s) - (p2[2] + d2z * t)
    return cx * cx + cy * cy + cz * cz

_col_tick = 0                                   # bumped each main-loop frame
_col_cache = {"tick": -1, "arm_pts": None, "balls": None}

def _collision_snapshot():
    """Per-frame cached obstacle geometry as raw-float tuples: every arm's current
    skeleton points, its joint/claw spheres, and a snapshot of the live balls. Rebuilt
    once per frame (keyed on _col_tick), so the many sweep checks inside one plan share a
    single computation instead of rebuilding skeletons every call.

    Each arm carries (robot, segment-points, joint-spheres, claw-segments): the
    segment-points give the two thin arm links; `joints` are the fat base and elbow joint
    spheres to path around (each (x, y, z, (SWEEP_ARM_RADIUS + SWEEP_JOINT_RADIUS)^2), the
    squared threshold a swept capsule must clear); and `claw_segs` are the claw's four
    finger/tooth capsule segments at the arm's CURRENT opening — the real claw geometry,
    not a bounding sphere."""
    c = _col_cache
    if c["tick"] == _col_tick:
        return c
    jt = (SWEEP_ARM_RADIUS + SWEEP_JOINT_RADIUS + SWEEP_MARGIN) ** 2
    arm_pts = []
    for r in robots:
        sk = _current_skeleton(r)
        pts = ((sk[0].x, sk[0].y, sk[0].z), (sk[1].x, sk[1].y, sk[1].z),
               (sk[2].x, sk[2].y, sk[2].z), (sk[3].x, sk[3].y, sk[3].z))
        joints = ((pts[0][0], pts[0][1], pts[0][2], jt),   # base joint
                  (pts[1][0], pts[1][1], pts[1][2], jt))    # elbow joint
        arm_dir = (sk[3] - sk[2]).hat                       # claw axis = (grab - wrist)
        cs = _claw_segments(sk[2], arm_dir, _up_perp_from(arm_dir), r["claw_state"]["angle"])
        claw_segs = tuple(((a.x, a.y, a.z), (b.x, b.y, b.z)) for a, b in cs)
        arm_pts.append((r, pts, joints, claw_segs))
    balls_snap = [(b.pos.x, b.pos.y, b.pos.z, b.radius, b)
                  for b in balls if not b.delivered and getattr(b, "visible", True)]
    c.update(tick=_col_tick, arm_pts=arm_pts, balls=balls_snap)
    return c

def sweep_overlaps(robot, world_pos=None, poses=None,
                   check_objects=True, check_arms=True, ignore=()):
    """True if `robot`'s sweep collides with any object or any OTHER arm.

    The sweep is the arm's path to `world_pos` (or a precomputed `poses` list from
    arm_sweep_poses). Every arm link AND every claw finger/tooth is treated as a capsule
    of radius SWEEP_ARM_RADIUS — so the swept volume is the arm's real silhouette, claw
    included (the splayed fingers, not a bounding sphere that misses their tips). Checks,
    for every sampled pose along the path:
      - objects: any live ball whose surface a swept link or claw segment passes within;
      - arms: any OTHER arm's CURRENT volume a swept segment comes within range of — its
        two arm links, its four claw finger/tooth segments, and its base and elbow joint
        spheres (so the path routes around claws and joints, not just the arm bodies).
    `ignore` is an iterable of balls to skip (e.g. the ball this arm is going to grab,
    which the sweep necessarily ends on). Balls held by this robot are skipped too.
    Returns a single True/False."""
    if poses is None:
        poses = arm_sweep_poses(robot, world_pos)
    snap = _collision_snapshot()
    ra = SWEEP_ARM_RADIUS
    arm_thresh2 = (2 * ra + SWEEP_MARGIN) ** 2   # capsule (arm link / claw segment) vs capsule

    # Broad phase: every swept point lies within REACH_MAX of the anchor; the claw fingers
    # add up to FINGER_LEN of reach beyond the wrist. Cull any obstacle past that (plus its
    # own extent) once here instead of testing it against every swept segment.
    anx, any_, anz = robot["anchor"].x, robot["anchor"].y, robot["anchor"].z
    cull2 = (REACH_MAX + FINGER_LEN + 0.5 + SWEEP_MARGIN) ** 2

    other_segs = []                              # arm links + claw finger/tooth segments
    other_spheres = []                           # base + elbow joint spheres
    if check_arms:
        for r, pts, joints, claw_segs in snap["arm_pts"]:
            if r is robot:
                continue
            for c, d in ((pts[0], pts[1]), (pts[1], pts[2])):    # the two arm links
                if _seg_point_d2(c[0], c[1], c[2], d[0], d[1], d[2], anx, any_, anz) <= cull2:
                    other_segs.append((c, d))
            for c, d in claw_segs:                               # the claw fingers/teeth
                if _seg_point_d2(c[0], c[1], c[2], d[0], d[1], d[2], anx, any_, anz) <= cull2:
                    other_segs.append((c, d))
            for x, y, z, thr2 in joints:
                if (x - anx) ** 2 + (y - any_) ** 2 + (z - anz) ** 2 <= cull2:
                    other_spheres.append((x, y, z, thr2))

    check_balls = []                             # (x, y, z, (arm_radius + ball_radius)^2)
    if check_objects:
        for x, y, z, rad, b in snap["balls"]:
            if b in ignore or b.held_by is robot:
                continue
            if (x - anx) ** 2 + (y - any_) ** 2 + (z - anz) ** 2 > cull2:
                continue                         # too far from this arm to ever collide
            thr = ra + rad + SWEEP_MARGIN
            check_balls.append((x, y, z, thr * thr))

    fa = robot["claw_state"]["angle"]            # the claw's real shape at its current opening
    for pose in poses:
        p0 = (pose[0].x, pose[0].y, pose[0].z)
        p1 = (pose[1].x, pose[1].y, pose[1].z)
        p2 = (pose[2].x, pose[2].y, pose[2].z)
        arm_dir = (pose[3] - pose[2]).hat        # claw axis = (grab - wrist)
        moving = [(p0, p1), (p1, p2)]            # the two arm links...
        for a, b in _claw_segments(pose[2], arm_dir, _up_perp_from(arm_dir), fa):
            moving.append(((a.x, a.y, a.z), (b.x, b.y, b.z)))   # ...plus the claw fingers/teeth
        for a, b in moving:
            ax, ay, az = a
            bx, by, bz = b
            for x, y, z, thr2 in check_balls:
                if _seg_point_d2(ax, ay, az, bx, by, bz, x, y, z) <= thr2:
                    return True
            for x, y, z, thr2 in other_spheres:  # base/elbow joints
                if _seg_point_d2(ax, ay, az, bx, by, bz, x, y, z) <= thr2:
                    return True
            for c, d in other_segs:
                if _seg_seg_d2(a, b, c, d) <= arm_thresh2:
                    return True
    return False

# ── Sweep-based pathfinding (A* over a Cartesian grab-point grid) ───────────────
ASTAR_STEP = 0.3            # Cartesian grid spacing (world units) for A* grab-point pathing
ASTAR_EDGE_STEPS = 3        # sweep sub-samples validating each short grid edge (no tunnelling)
ASTAR_NODE_BUDGET = 4000    # cap on nodes A* expands before giving up (-> caller holds & retries)
ASTAR_HEURISTIC_WEIGHT = 2.0 # f = g + W*h (W=1 -> plain A*). W>1 biases the search greedily
                             # toward the goal instead of expanding a fat frontier -> far fewer
                             # nodes explored (the freeze's main cost), at the price of a
                             # possibly-suboptimal path — cleaned up after by the string-pull.
ASTAR_TIME_BUDGET = 0.040   # wall-clock cap (seconds) on a single plan_path search. Set above the
                             # measured cost of NORMAL solvable detours with check_arms=True and 6
                             # arms present (profiled up to ~30ms), so legitimate replans aren't
                             # aborted — it only cuts off genuinely pathological searches (dense/
                             # maze-like obstacle fields) that would otherwise run far longer and
                             # stall the main loop. With up to 6 arms possibly replanning in ONE
                             # frame, the node budget alone doesn't bound wall-clock time per call,
                             # and since VPython dispatches button clicks at the same yield points
                             # the loop is busy blocking, a long stall can delay/coalesce UI clicks
                             # (e.g. the RL Grasp toggle "not registering"). On expiry: same
                             # fallback as the node budget (None -> caller holds & retries).

def _current_config(robot):
    """The arm's current joint angles (base, arm1, arm2)."""
    return (robot["base_state"]["angle"], robot["arm_state"]["angle"],
            robot["arm2_state"]["angle"])

def _sweep_poses_between(robot, cfg_a, cfg_b, steps=12):
    """Arm skeletons sampled along the joint interpolation between two configs (base
    taken the short way round) — the sweep from a HYPOTHETICAL pose, for planning."""
    b0, a10, a20 = cfg_a
    b1, a11, a21 = cfg_b
    db = (b1 - b0 + pi) % (2 * pi) - pi
    return [_arm_skeleton(robot, b0 + db * t, a10 + (a11 - a10) * t, a20 + (a21 - a20) * t)
            for t in (i / steps for i in range(steps + 1))]

def _clear_between(robot, cfg_a, cfg_b, ignore, steps, check_objects, check_arms):
    """True if the sweep from config cfg_a to cfg_b hits nothing that's being checked."""
    poses = _sweep_poses_between(robot, cfg_a, cfg_b, steps)
    return not sweep_overlaps(robot, poses=poses, ignore=ignore,
                              check_objects=check_objects, check_arms=check_arms)

def _astar_smooth(robot, pts, cfgs, ignore, check_objects, check_arms):
    """String-pulling: greedily drop intermediate waypoints a straight sweep can skip, turning
    the grid's staircase path into a few clean hops. Keeps the first (current pose) and last
    (exact target). Operates on the parallel (world point, config) lists — each shortcut is
    collision-checked between the two endpoints' CONFIGS at the full sweep resolution."""
    n = len(cfgs)
    if n <= 2:
        return list(pts)
    out = [pts[0]]
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if _clear_between(robot, cfgs[i], cfgs[j], ignore, 12, check_objects, check_arms):
                break
            j -= 1
        out.append(pts[j])
        i = j
    return out

def plan_path(robot, target, ignore=(), steps=12, check_objects=True, check_arms=True):
    """Plan a collision-free path for `robot`'s claw from its CURRENT pose to `target` (a world
    position). Returns a list of world-point waypoints ending at `target`, or None if none found.

    A* over a Cartesian grid of REACHABLE grab-point positions around the arm (spacing
    ASTAR_STEP). Each grid point maps to the arm's single-valued IK config (_target_config), so
    every node is a pose the drive can actually reproduce — and each returned waypoint round-trips
    faithfully through command_arm_to. A grid edge (adjacent points) is accepted only when the
    swept motion between their configs is collision-free (_clear_between, lazily). The admissible
    straight-line-distance heuristic makes this A* (not plain Dijkstra), so it expands few nodes;
    the result is then string-pulled to drop redundant hops. `ignore` is balls to skip (pass the
    target ball so the path may end on it); `check_objects`/`check_arms` pick what to avoid.
    None -> the caller (drive_avoiding) holds position and retries, exactly as before."""
    t0 = time.perf_counter()
    start_cfg = _current_config(robot)
    goal = vector(target.x, target.y, target.z)
    goal_cfg = _target_config(robot, goal)

    # Straight shot first — the common case, and cheapest.
    if _clear_between(robot, start_cfg, goal_cfg, ignore, steps, check_objects, check_arms):
        return [goal]

    anchor = robot["anchor"]
    st = ASTAR_STEP

    def to_world(node):
        return anchor + vector(node[0] * st, node[1] * st, node[2] * st)

    def reachable_node(node):
        return REACH_MIN <= mag(to_world(node) - anchor) <= REACH_MAX

    def snap_reachable(pt):
        node = (int(round((pt.x - anchor.x) / st)),
                int(round((pt.y - anchor.y) / st)),
                int(round((pt.z - anchor.z) / st)))
        for _ in range(4):                           # snapping can land just outside the shell —
            if reachable_node(node):                 # step back toward the anchor until reachable
                break
            node = tuple(i - (1 if i > 0 else (-1 if i < 0 else 0)) for i in node)
        return node

    start = snap_reachable(_arm_skeleton(robot, *start_cfg)[3])
    goaln = snap_reachable(goal)

    cfg_cache = {}
    def cfg_of(node):
        c = cfg_cache.get(node)
        if c is None:
            c = _target_config(robot, to_world(node))
            cfg_cache[node] = c
        return c

    def heur(node):
        return mag(to_world(node) - goal)            # straight-line distance -> admissible

    _NEI = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    openh = [(ASTAR_HEURISTIC_WEIGHT * heur(start), 0.0, start)]
    gsc = {start: 0.0}
    parent = {}
    closed = set()
    found = False
    while openh:
        _, gc, node = heapq.heappop(openh)
        if node in closed:
            continue
        closed.add(node)
        if node == goaln:
            found = True
            break
        if len(closed) > ASTAR_NODE_BUDGET or (time.perf_counter() - t0) > ASTAR_TIME_BUDGET:
            return None                              # too large/slow -> give up (caller holds/retries)
        ca = cfg_of(node)
        for dx, dy, dz in _NEI:
            nb = (node[0] + dx, node[1] + dy, node[2] + dz)
            if nb in closed or not reachable_node(nb):
                continue
            if not _clear_between(robot, ca, cfg_of(nb), ignore, ASTAR_EDGE_STEPS,
                                  check_objects, check_arms):
                continue
            g2 = gc + st
            if g2 < gsc.get(nb, float("inf")):
                gsc[nb] = g2
                parent[nb] = node
                heapq.heappush(openh, (g2 + ASTAR_HEURISTIC_WEIGHT * heur(nb), g2, nb))
    if not found:
        return None

    chain = [goaln]
    while chain[-1] in parent:
        chain.append(parent[chain[-1]])
    chain.reverse()                                  # start-side ... goal-side grid nodes

    # Parallel world-point / config lists WITH the exact endpoints, then string-pull.
    pts = [_arm_skeleton(robot, *start_cfg)[3]] + [to_world(n) for n in chain] + [goal]
    cfgs = [start_cfg] + [cfg_of(n) for n in chain] + [goal_cfg]
    smoothed = _astar_smooth(robot, pts, cfgs, ignore, check_objects, check_arms)
    return smoothed[1:]                              # drop the current pose; ends exactly on goal

pathfind_task = None      # active pathfinding route being followed, or None
PATHFIND_WP_TOL = 0.3     # how close the claw must get to an intermediate waypoint...
PATHFIND_END_TOL = 0.15   # ...and to the final target, before advancing/finishing
DRIVE_REPLAN_DIST = 0.3   # re-plan a detour if its goal has drifted more than this
PATHFIND_RETRY_FRAMES = 20 # while waiting (no route found), retry the planner this often —
                           # an obstacle may have moved and opened a detour since

def _route_clear(robot, path, i, ignore, check_objects, check_arms):
    """True if the sweep from the arm's CURRENT pose through the remaining waypoints
    (path[i:]) is still collision-free — re-validates a cached detour against obstacles
    that may have moved into it since it was planned."""
    cfg = _current_config(robot)
    for wp in path[i:]:
        nxt = _target_config(robot, wp)
        if not _clear_between(robot, cfg, nxt, ignore, 12, check_objects, check_arms):
            return False
        cfg = nxt
    return True

def _hold_position(robot):
    """Freeze the arm at its CURRENT joint angles. Used when no collision-free route to the
    goal exists this frame, so the arm WAITS in place instead of charging straight through
    the obstacle. It keeps retrying the planner and takes any clear shot the moment one
    opens, so this is a pause, not a permanent stop."""
    _set_pid(robot["manager"], robot["arm_state"], robot["arm_state"]["angle"])
    _set_pid(robot["manager2"], robot["arm2_state"], robot["arm2_state"]["angle"])
    cb = robot["manager_base"].active
    if isinstance(cb, CircularController):
        cb.target_angle = robot["base_state"]["angle"]
    else:
        robot["manager_base"].set_controller(CircularController(
            robot["base_state"], target_angle=robot["base_state"]["angle"],
            kp=20, ki=0.0, kd=27, circular=True))

def drive_avoiding(robot, goal, ignore=(), check_objects=False, check_arms=True):
    """Drive `robot`'s claw toward `goal` while routing around obstacles automatically.
    Goes STRAIGHT when the sweep to `goal` is clear (the common case) and drops any detour;
    when the way is blocked it plans a detour, follows it waypoint by waypoint, and re-plans
    if the goal drifts, a fresh straight shot opens up, OR an obstacle moves into the
    remaining detour (the route is re-validated every frame). If NO collision-free route is
    found it HOLDS POSITION and keeps retrying — it never drives straight through obstacles.
    Defaults to avoiding other ARMS only — boxes aren't dodged, since the arm is generally
    trying to reach one.

    The per-arm detour state lives in robot['drive_path']; command_arm_to matches how each
    waypoint was planned, so the arm follows the collision-checked sweep."""
    plan = robot.get("drive_path")
    # Clear straight shot? -> go direct and drop any detour/decision (the common case). This
    # runs every frame, so the arm resumes the instant a straight path opens up.
    if not sweep_overlaps(robot, world_pos=goal, ignore=ignore,
                          check_objects=check_objects, check_arms=check_arms):
        if plan is not None:
            robot["drive_path"] = None
        command_arm_to(robot, goal)
        return
    # Following a detour: if a (moving) obstacle has since blocked the remaining route,
    # discard the stale plan so we re-plan below against the obstacles' current positions.
    if (plan is not None and plan["path"]
            and not _route_clear(robot, plan["path"], plan["i"],
                                 ignore, check_objects, check_arms)):
        plan = None
        robot["drive_path"] = None
    # Blocked. (Re)plan when we have no decision for ~this goal, the goal drifted, or a
    # cached "no route" has waited long enough to retry (an obstacle may have cleared).
    if (plan is None or mag(plan["goal"] - goal) > DRIVE_REPLAN_DIST
            or (not plan["path"] and plan.get("retry", 0) <= 0)):
        path = plan_path(robot, goal, ignore=ignore,
                         check_objects=check_objects, check_arms=check_arms)
        robot["drive_path"] = {"goal": vector(goal.x, goal.y, goal.z), "path": path,
                               "i": 0, "retry": PATHFIND_RETRY_FRAMES}
        if path:
            command_arm_to(robot, path[0])           # follow the collision-free route
        else:
            _hold_position(robot)                    # no route -> WAIT, don't charge through
        return
    path = plan["path"]
    if not path:                                     # still no route for this goal — wait
        plan["retry"] = plan.get("retry", 0) - 1     # count down to the next planning retry
        _hold_position(robot)
        return
    i = plan["i"]                                    # follow the cached detour
    if i < len(path) - 1 and _grab_point_near(robot, path[i], tol=PATHFIND_WP_TOL):
        plan["i"] = i + 1
        command_arm_to(robot, path[i + 1])

def start_pathfind(robot, target_ball):
    """Plan a collision-free route for `robot` to `target_ball` — avoiding the other
    balls and arms — and begin following it. The target ball itself is passed in `ignore`
    so the route is allowed to end on it. Returns True if a route was found."""
    global pathfind_task
    path = plan_path(robot, target_ball.pos, ignore=(target_ball,))
    if not path:
        pathfind_task = None
        return False
    command_arm_to(robot, path[0])               # head for the first waypoint
    pathfind_task = {"robot": robot, "ball": target_ball, "path": path, "i": 0}
    return True

def pathfind_step():
    """Advance the active pathfinding route each frame: drive to the current waypoint,
    step to the next once the claw reaches it, and finish at the final one (the target).
    Each waypoint is commanded exactly as it was planned (command_arm_to), so the arm
    follows the collision-checked sweep."""
    global pathfind_task
    if pathfind_task is None:
        return
    robot, path, i = pathfind_task["robot"], pathfind_task["path"], pathfind_task["i"]
    final = (i == len(path) - 1)
    tol = PATHFIND_END_TOL if final else PATHFIND_WP_TOL
    if _grab_point_near(robot, path[i], tol=tol):
        if final:
            pathfind_task = None                 # reached the target
            return
        pathfind_task["i"] = i + 1
        command_arm_to(robot, path[i + 1])       # on to the next waypoint

def pathfind_ball(b):
    """Manual 'Pathfind' button: plan and follow a collision-free route for the nearest
    arm to a target ball (a stationary cluster box), weaving around the other balls."""
    target = cluster_balls[0]
    robot = min(robots, key=lambda r: mag(target.pos - r["anchor"]))
    start_pathfind(robot, target)

def close_claw(robot):
    robot["close_lock"] = CLOSE_LOCK_FRAMES   # hold shut briefly so a just-closed claw can't
                                              # immediately reopen and fling the arriving box
    # Latch the claw closed while it travels shut: until it fully bottoms out (or grips a
    # box), open_claw is a HARD no-op, so nothing can reverse it mid-close. Don't latch a
    # claw that already holds a box (a delivery giver re-gripping for the pass) — it must
    # stay able to open to release.
    if robot["grip_angle"] is None and not robot["ball_grabbed"]:
        robot["claw_latched_closing"] = True
    robot["manager_claw"].set_controller(
        PIDController(robot["claw_state"], target_angle=0.0, kp=8, ki=0.2, kd=0.5))

def open_claw(robot):
    if robot.get("claw_latched_closing"):
        return                                # still mid-close — refuse to reopen until it
                                              # has bottomed out (latch cleared in step_robot)
    robot["manager_claw"].set_controller(
        PIDController(robot["claw_state"], target_angle=pi/2, kp=8, ki=0.2, kd=0.5))

def _grab_point_near(robot, world_pos, tol=0.5):
    gp = claw_grab_point(robot["arm2_state"], robot["base_state"]["angle"])
    return mag(gp - world_pos) < tol

def _ball_holder():
    """Index of the arm currently holding the ball (grabbed or gripped), or None."""
    for i, r in enumerate(robots):
        if r["ball_grabbed"] or r["grip_angle"] is not None:
            return i
    return None

def _arm_still(robot):
    """True when all of the arm's joints are nearly stopped."""
    return (abs(robot["base_state"]["angular_velocity"]) < SETTLE_AV
            and abs(robot["arm_state"]["angular_velocity"]) < SETTLE_AV
            and abs(robot["arm2_state"]["angular_velocity"]) < SETTLE_AV)

def _handoff_point(giver_idx, receiver_idx):
    """A spot both arms can reach: the midpoint between their bases, offset forward
    by 2 in world x. For arm1<->arm2 this is (2, 0, 1.5)."""
    a = robots[giver_idx]["anchor"]
    b = robots[receiver_idx]["anchor"]
    return (a + b) / 2 + vector(2, 0, 0)

REST_OFFSET = vector(2, 0, 0)   # an arm's rest pose, relative to its own base

def _rest_pos(robot):
    """The arm's rest position — straight out in front of its base by REST_OFFSET."""
    return robot["anchor"] + REST_OFFSET

def _center_arm():
    """The delivery sink: the arm nearest the drop point DELIVER_TARGET. Boxes are passed
    along the grid hop by hop to this arm, which then places them at the drop point. Ties
    break by index so it is a single, stable target every box is routed to."""
    return min(range(len(robots)),
               key=lambda i: (mag(robots[i]["anchor"] - DELIVER_TARGET), i))

def _grid_neighbors(i):
    """Arm indices directly adjacent to arm i on the grid — the nearest arm in each
    axis-aligned direction (±x, ±y, ±z), i.e. up/down, left/right, forward/back.
    A candidate counts only if the step to it is predominantly along one axis (not a
    diagonal), so the routing never passes across a diagonal gap."""
    ai = robots[i]["anchor"]
    best = {}   # (axis, sign) -> (on_axis_distance, arm_index)
    for j, r in enumerate(robots):
        if j == i:
            continue
        d = r["anchor"] - ai
        comps = (("x", d.x), ("y", d.y), ("z", d.z))
        axis, val = max(comps, key=lambda c: abs(c[1]))
        if abs(val) < 1e-6:
            continue
        off = sqrt(sum(c * c for name, c in comps if name != axis))
        if off > abs(val) * GRID_DIAG_TOL:      # too diagonal — not a grid neighbour
            continue
        key = (axis, 1 if val > 0 else -1)
        if key not in best or abs(val) < best[key][0]:
            best[key] = (abs(val), j)
    return [j for _, j in best.values()]

def _hops_to_center():
    """BFS hop-distance from every arm to the centre arm over the grid-adjacency
    graph. Boxes are routed down these hop counts, so they always step to a directly
    adjacent arm that is one step closer to the centre — works for any grid shape,
    including even grids where distance-to-centroid alone would stall."""
    graph = {i: _grid_neighbors(i) for i in range(len(robots))}
    center = _center_arm()
    dist = {center: 0}
    frontier = [center]
    while frontier:
        nxt = []
        for u in frontier:
            for v in graph[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist

def _next_hop(holder, hops=None):
    """The adjacent arm to pass to: the grid neighbour of `holder` that is one hop
    closer to the centre arm. None if `holder` is the centre arm (place the box) or
    can't reach the centre over grid edges (fallback: place it here)."""
    if hops is None:
        hops = _hops_to_center()
    h = hops.get(holder)
    if not h:                                    # centre arm (0) or unreachable (None)
        return None
    best = None
    for v in _grid_neighbors(holder):
        hv = hops.get(v)
        if hv is not None and hv < h and (best is None or hv < best[0]):
            best = (hv, v)
    return best[1] if best else None

def _deliver_target():
    """The fixed drop point where the sink arm places delivered boxes."""
    return DELIVER_TARGET

def _is_holding(robot):
    """True if this arm currently holds an object (button-grabbed or gripped)."""
    return robot["ball_grabbed"] or robot["grip_angle"] is not None

def _start_delivery(holder):
    # One cycle: the holding arm either hands its box to the adjacent arm one grid hop
    # closer to the centre, or (if it IS the centre arm) carries it to the centre
    # arm's drop spot and places it. Repeated cycles walk a box in to the centre. The
    # box being delivered is stored so the whole sequence follows the right object.
    # Multiple cycles run at once as long as they use disjoint arms — so a box advances
    # whenever ITS next arm is free, even while other arms are busy delivering.
    if holder is None:
        return
    box = robots[holder]["held"]
    if box is None:
        return                                   # holder isn't actually carrying a box
    busy = _delivery_busy_arms()
    if holder in busy:
        return                                   # this arm is already mid-hand-off
    receiver = _next_hop(holder)
    if receiver is not None and receiver in busy:
        return                                   # its next arm is tied up in another cycle
    box.reserved = True                          # hide it from tracking during the pass
    robots[holder]["ignore_ball"] = False        # only this cycle's arms — leave others' alone
    robots[holder]["drive_path"] = None          # start this cycle's routing fresh (drop any
                                                 # stale detour left from catching)
    if receiver is None:
        # Centre arm (or unroutable) — deliver straight to the centre drop spot
        target = _deliver_target()
        command_arm_to(robots[holder], target)
        handoff_tasks.append({"stage": "final", "arm": holder, "box": box,
                              "elapsed": 0, "target": target})
    else:                                        # hand off toward the centre
        robots[receiver]["ignore_ball"] = False
        robots[receiver]["drive_path"] = None
        hp = _handoff_point(holder, receiver)
        command_arm_to(robots[holder], hp)
        close_claw(robots[holder])               # hold it firmly for the pass
        standby = hp + 0.7 * (robots[receiver]["anchor"] - hp).hat
        command_arm_to(robots[receiver], standby) # receiver waits close by, ready
        open_claw(robots[receiver])
        handoff_tasks.append({"stage": "to_handoff", "from": holder, "to": receiver,
                              "hp": hp, "settle": 0, "elapsed": 0, "box": box})

def deliver_ball(b):
    # Manual "Deliver Ball" button: one cycle for the first arm holding a box.
    _start_delivery(_ball_holder())

def estimate_travel_time(robot, target_pos):
    """Rough estimate (seconds) of how long `robot` takes to reach the world
    position `target_pos`. Each joint is modelled as torque-limited (bang-bang):
    a move of Δθ with peak acceleration α = MAX_TORQUE / I takes
    t = 2·sqrt(Δθ · I / MAX_TORQUE). The base, arm1 and arm2 move in parallel,
    so the arm's travel time is the slowest joint. (Idealised — it ignores drag
    and PID overshoot, so the real motion is a bit slower.)"""
    anchor = robot["anchor"]
    lx = target_pos.x - anchor.x      # target in this arm's local control frame
    ly = target_pos.z - anchor.z      # (x, y, z) = (worldX, worldZ, worldY)
    lz = target_pos.y - anchor.y

    target_base = translate_degree(lx, ly)
    target_a1, target_a2 = caculate_target_arm_angles(lx, ly, lz, robot["arm_state"]["length"])

    def joint_time(target_angle, current_angle, state, circular=False):
        d = target_angle - current_angle
        if circular:
            d = (d + pi) % (2 * pi) - pi   # shortest way round
        d = abs(d)
        I = (1 / 3) * state["mass"] * state["length"] ** 2
        return 2 * sqrt(d * I / MAX_TORQUE)

    return max(
        joint_time(target_base, robot["base_state"]["angle"], robot["base_state"], circular=True),
        joint_time(target_a1, robot["arm_state"]["angle"], robot["arm_state"]),
        joint_time(target_a2, robot["arm2_state"]["angle"], robot["arm2_state"]),
    )

# ── Arm ↔ object assignment ───────────────────────────────────────────
MOVING_SPEED_TOL = 0.1 # speed above which an object counts as "moving" (below = stationary)

def _obj_pos(o):
    """World position (vector) of a tracked object — a KalmanTrack, an (x, y, z)
    tuple, or a vector."""
    if hasattr(o, "position"):
        return vector(*o.position)          # KalmanTrack
    if isinstance(o, vector):
        return o
    return vector(*o)

def _obj_speed(o):
    """Speed (velocity magnitude) of a tracked object; 0 if it exposes no velocity."""
    v = getattr(o, "velocity", None)
    return mag(vector(*v)) if v is not None else 0.0

def _optimal_catch_assignment(tracks, free_arms):
    """Match free arms to reachable objects so as many boxes as possible are caught at
    the SAME TIME — a maximum-cardinality matching, not just each arm grabbing its own
    nearest (which dropped a catchable box whenever two arms wanted the same one).

    Each arm catches at most one object and each object goes to at most one arm; an arm
    is only offered objects whose wait point it can actually reach. Among all matchings
    that catch the most boxes, ties break in order by: catching more MOVING boxes (they
    pass by and get away — stationary ones can wait), then keeping arms on their current
    target (stability, so the plan doesn't thrash frame to frame), then shortest total
    travel. Returns {arm_idx: obj_idx}.

    Exhaustive over the free arms (each picks a reachable object or nothing) — tiny for
    a handful of arms; a cardinality bound prunes hopeless branches so it stays cheap
    even as the arm count grows."""
    n_obj = len(tracks)
    # Reachable (arm -> objects) edges, plus per-pair metrics for the tie-breakers.
    reach, dist, keep = {}, {}, {}
    for ai in free_arms:
        anchor = robots[ai]["anchor"]
        edges = []
        for oi in range(n_obj):
            pos = _obj_pos(tracks[oi])
            vel = vector(*tracks[oi].velocity)
            if _reachable(robots[ai], _intercept_camp(robots[ai], pos, vel)):
                edges.append(oi)
                dist[(ai, oi)] = mag(pos - anchor)
                keep[(ai, oi)] = robots[ai]["plan_obj"] is tracks[oi]
        reach[ai] = edges
    moving = [_obj_speed(tracks[oi]) > MOVING_SPEED_TOL for oi in range(n_obj)]

    best = [None]                                    # (key, mapping); higher key = better
    n_free = len(free_arms)
    def rec(i, used, chosen, cnt, mov, kp, dsum):
        # Prune: if matching every remaining arm still can't beat the best count, stop.
        if best[0] is not None and cnt + (n_free - i) < best[0][0][0]:
            return
        if i == n_free:
            key = (cnt, mov, kp, -dsum)              # maximize count, moving, kept; min dist
            if best[0] is None or key > best[0][0]:
                best[0] = (key, dict(chosen))
            return
        ai = free_arms[i]
        rec(i + 1, used, chosen, cnt, mov, kp, dsum)   # this arm catches nothing
        for oi in reach[ai]:                           # ...or a reachable, unclaimed object
            if oi in used:
                continue
            chosen[ai] = oi; used.add(oi)
            rec(i + 1, used, chosen, cnt + 1, mov + moving[oi],
                kp + keep[(ai, oi)], dsum + dist[(ai, oi)])
            used.discard(oi); del chosen[ai]
    rec(0, set(), {}, 0, 0, 0, 0.0)
    return best[0][1] if best[0] else {}

# ── Auto-catch: drive assigned arms to an intercept and close on arrival ───────
CATCH_CLOSE_DIST = 0.08  # object must be this close to the claw's centre line to close
GRAB_LATERAL_TOL = 0.05  # a STATIONARY object closes only when its offset from the claw
                         # axis (dead between the fingers) is under this — much tighter
                         # than merely being anywhere inside the grab prism, so a close
                         # can't clip one finger and shove the object out (a misgrab)
GRAB_DEPTH_LO = 0.10     # ...and it sits within this depth band along the claw axis
GRAB_DEPTH_HI = 0.50     # (finger base out to the teeth), not jammed in the corner
CENTER_GAIN = 0.5        # per-frame re-centring feedback: shift a catch's wait point by this
                         # fraction of the object's lateral offset so the claw axis servos onto
                         # it, cancelling the steady IK aiming error (kept <1 so the second-order
                         # arm converges without oscillating). Used for both the stationary servo
                         # and the moving-box servo below.
CENTER_DEADBAND = 0.02   # the moving-box servo stops nudging once the box's path is within this
                         # of the claw centre-line, so the arm SETTLES (the RL close needs
                         # _arm_still) — well inside the ~0.05 "centred" catch band.
CATCH_CLOSED_ANGLE = 0.6 # claw counts as "shut" below this angle
CLAW_SHUT_ANGLE = 0.5    # claw counts as FULLY closed (bottomed out) at/below this — just
                         # above the 0.45 physical minimum. The failed-catch recovery only
                         # arms once the claw has finished closing, so it can never reverse a
                         # claw that is still mid-close.
CATCH_OPEN_ANGLE = 1.2   # claw counts as "open" (ready to close) above this
EMPTY_CLOSE_FRAMES = 12  # frames a claw may stay shut-and-empty before the catch is
                         # judged failed and the claw reopens to retry
CLOSE_LOCK_FRAMES = 25   # once the claw is commanded shut it's LOCKED closed for this many
                         # frames (~0.25s): the empty-claw recovery can't reopen it in that
                         # window, so a box arriving a hair late still gets gripped instead
                         # of the claw closing and immediately reopening and flinging it
CATCH_CLOSE_TIMEOUT = 200 # frames the arm may sit parked (settled, open) without the
                          # object centred before it closes anyway and hopes to grab (~2s)
# The lead time for the close is no longer a fixed constant: _claw_close_time(robot)
# computes it per arm from the claw's CURRENT opening (open->contact sweep at
# CLAW_MAX_AV), so a wide-open claw leads more than a nearly-shut one. The time-to-go
# gate fires when the box will reach the grab point within that many seconds, so the
# fingers finish closing just as it arrives. Because the lead is measured along the
# box's OWN path (t_go), it can be the honest close time without over-leading an angled
# head-on box; the closest-approach prism test is what prevents a premature fire.
CLOSE_LEAD_FACTOR = 0.70 # scales the estimated close time used as the lead. <1 shrinks the
                         # lead so the claw fires when the box is nearer — i.e. closes a
                         # little later (the box seats deeper before the fingers meet it).
CATCH_STANDOFF_DIST = 2.5 # comfortable reach the arm waits at for a passing object
CATCH_NEAR_DIST = 0.5    # the object counts as "arrived" at the claw within this range
                         # (only then does the parked close-timeout start counting)
VEL_CHANGE_TOL = 0.3    # only re-plan an arm's wait point when its object's velocity
                        # changes by more than this (else keep the arm parked, ignoring
                        # the tracker's per-frame velocity jitter)
# Reach envelope: the claw's grab point can reach objects in a spherical shell
# about the base. Objects outside it are freed (left for later) rather than chased.
REACH_MAX = ARM1_LEN + ARM2_LEN + 0.2   # ~4.0: full extension plus a little grab reach
REACH_MIN = abs(ARM1_LEN - ARM2_LEN)    # 0.2: the arm can't fold tighter than this
_catching_arms = set()   # arms auto-catch is actively driving this frame (delivery
                         # must not hijack them as hand-off receivers)
CATCH_PROTECT_TIME = 5.0 # an arm whose assigned box will reach the claw within this many
                         # seconds is off-limits to deliveries — even if the box is still
                         # just out of reach this instant, so the arm is free the moment
                         # it arrives instead of being tied up in a hand-off
_catch_protected = set() # arms with an imminent assigned catch (ETA < CATCH_PROTECT_TIME)

def _ray_sphere_earliest(p0, v, c, r):
    """Earliest t >= 0 where the ray p0 + v·t meets the sphere of radius r about c,
    or None if it never does."""
    d = p0 - c
    a = dot(v, v)
    b = 2 * dot(v, d)
    cc = dot(d, d) - r * r
    disc = b * b - 4 * a * cc
    if disc < 0 or a < 1e-12:
        return None
    s = sqrt(disc)
    for t in sorted(((-b - s) / (2 * a), (-b + s) / (2 * a))):
        if t >= 0:
            return t
    return None

def _intercept_camp(robot, pos, vel):
    """A STABLE point on the object's forward path for the arm to wait at, so it
    parks and lets the object come to it rather than chasing it every frame.

    The point depends only on the object's straight-line path and the arm's base
    (not the arm's momentary pose), so it holds still while the arm drives to it:
      - stationary object -> its own position;
      - approaching object -> where its path first reaches a comfortable catch
        distance from the base (or the path's closest approach if it stays closer);
      - receding / never-in-reach object -> a far-downstream point, so the caller's
        reachability check frees the arm instead of chasing a box it can't catch."""
    anchor = robot["anchor"]
    v2 = dot(vel, vel)
    if v2 < 1e-9:                                    # stationary — its own spot
        return pos
    t_star = dot(anchor - pos, vel) / v2             # time of closest approach
    if t_star <= 0:                                  # closest approach already past — receding
        return pos + vel * 1000.0                    # far downstream -> out of reach -> freed
    q = pos + vel * t_star                           # closest-approach point (stable)
    # If the path comes closer than the standoff, wait where it *first* reaches the
    # standoff (on the incoming side) instead of letting the arm sit right at the base.
    if mag(q - anchor) < CATCH_STANDOFF_DIST and mag(pos - anchor) > CATCH_STANDOFF_DIST:
        t_hit = _ray_sphere_earliest(pos, vel, anchor, CATCH_STANDOFF_DIST)
        if t_hit is not None:
            return pos + vel * t_hit
    return q

CLAW_RECT_LEN = FINGER_LEN           # grab-prism depth along the claw axis (finger reach)
CLAW_RECT_HALFWIDTH = FINGER_LEN / 4 # half the finger gap — reaches each finger's quarter point
CLAW_RECT_HALFDEPTH = 0.02           # half the off-plane thickness (0.04 total)

def _claw_grab_distance(robot, point):
    """Distance from `point` to the claw's grab PRISM — a rectangular box fixed to
    the claw between the fingers. It spans the claw axis (arm direction) out to the
    finger reach, the finger-gap direction out to each finger's quarter point, and a
    thin ±HALFDEPTH off the grasp plane. A point inside the prism reads ~0; the metric
    is how far it sits outside along any of the three axes, so the close test keys off
    the box being inside the grasp volume rather than exactly on the centre line."""
    tip, arm_dir, up_perp, side = claw_frame(robot["arm2_state"], robot["base_state"]["angle"])
    rel = point - tip
    u = dot(rel, arm_dir)          # along the claw axis (depth into the claw)
    v = dot(rel, up_perp)          # across the finger gap
    w = dot(rel, side)             # out of the grasp plane
    du = max(0.0, -u, u - CLAW_RECT_LEN)             # outside the [0, LEN] depth span
    dv = max(0.0, abs(v) - CLAW_RECT_HALFWIDTH)      # outside the ±HALFWIDTH gap span
    dw = max(0.0, abs(w) - CLAW_RECT_HALFDEPTH)      # outside the ±HALFDEPTH off-plane slab
    return sqrt(du * du + dv * dv + dw * dw)

def _grab_offsets(robot, point):
    """`point` resolved in the claw frame relative to the finger tip: (u, v, w) =
    (depth along the claw axis, and the two offsets perpendicular to it). A clean grab
    needs small |v| and |w| — the object sitting on the claw axis, centred between the
    fingers rather than against one — at a sensible depth u (not jammed at the base)."""
    tip, arm_dir, up_perp, side = claw_frame(robot["arm2_state"], robot["base_state"]["angle"])
    rel = point - tip
    return dot(rel, arm_dir), dot(rel, up_perp), dot(rel, side)

def _claw_close_time(robot):
    """Estimate the seconds this arm's claw needs to shut onto an object from its CURRENT
    opening, so the time-to-go gate leads by the right amount instead of a fixed guess.
    The close is velocity-limited: the finger's torque-to-inertia ratio is so high that
    the speed pins at the CLAW_MAX_AV cap almost immediately, so the time is essentially
    the angle left to travel (down to the 'shut' angle where the fingers meet a centred
    box) divided by that capped speed — minus the 2%/frame drag update_claw applies. A
    fully-open claw (angle ≈ π/2) therefore leads far more than a barely-open one, which
    a single constant couldn't capture."""
    travel = max(0.0, robot["claw_state"]["angle"] - CATCH_CLOSED_ANGLE)
    return travel / (CLAW_MAX_AV * 0.98) * CLOSE_LEAD_FACTOR

def _reachable(robot, point):
    """True if `point` is within the arm's reach envelope (a spherical shell about
    the base), so the claw can actually get to it."""
    d = mag(point - robot["anchor"])
    return REACH_MIN <= d <= REACH_MAX

def _delivery_busy_arms():
    """Arm indices currently tied up in ANY delivery cycle (they must not be pulled
    away by auto-catch, nor double-booked by another hand-off). The giver is only busy
    until it has dropped the box and swung clear — once the receiver is picking up
    (approach/grabbing), the giver is free again and only the receiver stays busy."""
    busy = set()
    for task in handoff_tasks:
        if task["stage"] == "final":
            busy.add(task["arm"])
        elif task["stage"] in ("approach", "grabbing"):
            busy.add(task["to"])                     # giver has handed off — it's free
        else:
            busy.add(task["from"]); busy.add(task["to"])
    return busy

def _free_task_arms(task):
    """Clear the `ignore_ball` flag AND any obstacle-avoidance detour on just the arms this
    delivery cycle owns, so ending one cycle can't re-enable box collisions on an arm still
    mid-hand-off in another, nor leave a stale detour behind for its next move."""
    for key in ("from", "to", "arm"):
        if key in task:
            robots[task[key]]["ignore_ball"] = False
            robots[task[key]]["drive_path"] = None

def _progress_delivery(task):
    """Advance one delivery cycle by a single frame. Returns True when the cycle is
    finished (the caller then drops it from handoff_tasks). Only ever touches the arms
    this task owns, so concurrent cycles on other arms are unaffected."""
    bx = task.get("box")                                 # the box this cycle delivers
    stage = task["stage"]
    if stage == "final":
        # Nearest arm carries the box to the target, settles, then places it — routing
        # around the other boxes and arms on the way (the box it carries is ignored).
        arm = robots[task["arm"]]
        target = task["target"]
        drive_avoiding(arm, target, ignore=(bx,) if bx is not None else (), check_objects=True)
        task["elapsed"] += 1
        arrived = _grab_point_near(arm, target, tol=0.2) and _arm_still(arm)
        if arrived or task["elapsed"] >= APPROACH_TIMEOUT:
            arm["ball_grabbed"] = False                  # place it and let go
            arm["grip_angle"] = None
            arm["grip_local"] = None
            arm["held"] = None
            if bx is not None:
                bx.held_by = None
                bx.velocity = vector(0, 0, 0)
                n = sum(1 for x in balls if x.delivered)
                bx.pos = target + vector(0, 0.25 * n, 0)   # stack neatly at the drop spot
                bx.delivered = True
                bx.reserved = False
            open_claw(arm)
            command_arm_to(arm, _rest_pos(arm))          # withdraw to the rest spot
            _free_task_arms(task)
            return True                                  # placed — cycle done
    else:
        src = robots[task["from"]]                       # giver
        rcv = robots[task["to"]]                         # receiver
        hp = task["hp"]
        if stage == "to_handoff":
            # Giver carries the box to the handoff spot — routing around the other boxes
            # and arms (ignoring the box it carries) — and waits until it is nearly still
            # before releasing, so the box barely moves when dropped.
            drive_avoiding(src, hp, ignore=(bx,) if bx is not None else (), check_objects=True)
            task["elapsed"] += 1
            if _grab_point_near(src, hp) and _arm_still(src):
                task["settle"] += 1
            else:
                task["settle"] = 0
            if task["settle"] >= SETTLE_DWELL or task["elapsed"] >= APPROACH_TIMEOUT:
                src["ball_grabbed"] = False              # let go...
                src["grip_angle"] = None
                src["grip_local"] = None
                if bx is not None:
                    bx.held_by = None                    # ...it's free for the receiver
                    bx.velocity = vector(0, 0, 0)        # ...and stays put (no gravity)
                src["held"] = None
                src["ignore_ball"] = True                 # ...and stop touching the box
                open_claw(src)                           # ...open the fingers off the box
                grab_btn.text = 'Grab'
                task["stage"] = "dropping"
        elif stage == "dropping":
            # Wait for the giver's claw to open clear of the box, then withdraw
            # the arm to its rest pose to swing clear of the box.
            if src["claw_state"]["angle1"] >= GIVER_OPEN_ANGLE:
                command_arm_to(src, _rest_pos(src))      # retreat to the rest spot
                task.update(stage="retract", elapsed=0)
        elif stage == "retract":
            # Wait until the giver has swung well clear before the receiver picks up
            task["elapsed"] += 1
            if (not _grab_point_near(src, bx.pos, tol=GIVER_CLEAR_DIST)
                    or task["elapsed"] >= APPROACH_TIMEOUT):
                src["ignore_ball"] = False       # giver is clear — free it to catch again
                open_claw(rcv)
                task.update(stage="approach", dwell=0, elapsed=0)
        elif stage == "approach":
            # The receiver drives onto the free box, routing around the other boxes and
            # arms (ignoring the box it's picking up). Only close once genuinely close
            # AND settled there.
            predicted = vector(*trilateration(sensors, bx))
            drive_avoiding(rcv, predicted, ignore=(bx,) if bx is not None else (), check_objects=True)
            if _grab_point_near(rcv, predicted, tol=APPROACH_TOL) and _arm_still(rcv):
                task["dwell"] += 1
            else:
                task["dwell"] = 0
            if task["dwell"] >= APPROACH_DWELL:
                close_claw(rcv)                          # receiver closes onto the free box
                task["stage"] = "grabbing"
        elif stage == "grabbing":
            # Receiver grabs with the normal free-box physics (it sets grip_angle
            # itself). Confirm via the sensors, then end the cycle — the next
            # cycle continues from the new holder.
            predicted = vector(*trilateration(sensors, bx))
            if rcv["grip_angle"] is not None and _grab_point_near(rcv, predicted, tol=0.5):
                if bx is not None:
                    bx.reserved = False
                _free_task_arms(task)
                return True                              # one cycle done
    return False

def auto_catch_step():
    """Every frame while Auto Catch is on: assign arms to tracked objects, drive
    each to an intercept point on its object's path, and close the claw the moment
    the object crosses the claw's centre line."""
    _catching_arms.clear()
    _catch_protected.clear()
    tracks = tracker.confirmed_tracks()
    if not tracks:
        return
    busy = _delivery_busy_arms()
    def _free(ai):
        """An arm that can take on a new catch: not tied up delivering, not holding."""
        r = robots[ai]
        return ai not in busy and r["grip_angle"] is None and not r["ball_grabbed"]

    # Match the free arms to reachable objects to maximize how many boxes are caught at
    # once (moving boxes first, then keep current targets, then shortest travel), rather
    # than each arm greedily taking only its nearest — which dropped a catchable box
    # whenever two arms contended for the same one.
    free_arms = [ai for ai in range(len(robots)) if _free(ai)]
    arm_to_obj = _optimal_catch_assignment(tracks, free_arms)
    for ai, oi in arm_to_obj.items():
        robot = robots[ai]
        if ai in busy:
            robot["catch_timer"] = 0
            continue                                 # tied up delivering — leave it
        if robot["grip_angle"] is not None or robot["ball_grabbed"]:
            robot["catch_timer"] = 0
            continue                                 # already holding something
        obj = tracks[oi]
        pos = _obj_pos(obj)
        vel = vector(*obj.velocity)
        grab_pt = claw_grab_point(robot["arm2_state"], robot["base_state"]["angle"])
        arrived = mag(pos - grab_pt) < CATCH_NEAR_DIST        # object is right at the claw now
        # Re-plan the wait point when a new object is assigned, the object's velocity
        # changes (so the arm re-aims to a maneuver immediately — even right at the
        # claw, which is the key to reacting quickly), or it passes its wait point.
        # The 'passed' trigger is suppressed once the object has ARRIVED, so a box
        # crossing the grasp isn't yanked to a far "receding" point mid-grab. Between
        # triggers the plan holds, so the arm doesn't twitch on per-frame velocity
        # jitter.
        passed = (not arrived) and dot(robot["anchor"] - pos, vel) <= 0
        if (robot["plan_obj"] is not obj or robot["plan_camp"] is None
                or robot["plan_vel"] is None
                or mag(vel - robot["plan_vel"]) > VEL_CHANGE_TOL or passed):
            robot["plan_obj"] = obj
            robot["plan_vel"] = vel
            robot["plan_camp"] = _intercept_camp(robot, pos, vel)
        camp = robot["plan_camp"]
        # Protect this arm from deliveries if its box will reach the wait point soon.
        # camp lies on the box's forward path, so the ETA is the along-path travel time
        # from the box to camp. Done BEFORE the reachability check so a box arriving
        # within CATCH_PROTECT_TIME still guards the arm while it's a moment out of
        # reach; a receding box (camp far downstream) has a huge ETA and doesn't guard.
        v2c = dot(vel, vel)
        eta = dot(camp - pos, vel) / v2c if v2c > 1e-9 else float("inf")
        if 0.0 <= eta < CATCH_PROTECT_TIME:
            _catch_protected.add(ai)
        # Reachability: free the arm if the wait point is out of reach. A box crossing
        # the grasp keeps its (reachable) wait point frozen, so it isn't freed; a box
        # that has maneuvered away re-plans to a far point and is freed here.
        if not _reachable(robot, camp):
            robot["catch_timer"] = 0
            continue                                 # out of reach — leave it for later
        _catching_arms.add(ai)                       # committed to a catch this frame
        claw_angle = robot["claw_state"]["angle"]

        # Recovery: a grip only locks *while the claw is still closing*; once the
        # claw bottoms out empty it can never grab. So if this (non-holding) arm has
        # sat with a FULLY-shut claw and nothing gripped for a while, the catch failed
        # (missed, wrong depth, or the box got knocked away) — reopen and retry. A
        # successful grip flips the arm to 'holding' above, so this only counts real
        # failures.
        #
        # The timer only accrues once the claw has finished closing (reached CLAW_SHUT_ANGLE,
        # its physical bottom), NEVER while it is still travelling shut — so the recovery can
        # never reverse a claw that is still mid-close. A claw still on its way down resets
        # the counter, so the close always completes before any reopen is even considered.
        if claw_angle <= CLAW_SHUT_ANGLE:
            robot["empty_closed"] += 1
        else:
            robot["empty_closed"] = 0
        if robot["empty_closed"] > EMPTY_CLOSE_FRAMES and robot["close_lock"] <= 0:
            # Past the just-closed lockout, and empty for a while. Don't reopen while a free
            # box is actually in the fingers — the outward-swinging fingers would fling it
            # out. Wait instead; the bottomed-out grip in step_robot locks onto it as it
            # seats. Only reopen a genuinely empty claw.
            a2s, ba = robot["arm2_state"], robot["base_state"]["angle"]
            box_in_claw = any(
                (not b.delivered) and b.held_by is None
                and (finger_contacts_ball(a2s, claw_angle, ba, b, 1)
                     or finger_contacts_ball(a2s, claw_angle, ba, b, -1))
                for b in balls)
            if not box_in_claw:
                open_claw(robot)
                robot["empty_closed"] = 0

        # Drive to the wait point, automatically routing around the other arms AND the
        # other boxes — but not the box being caught (the nearest live box to this
        # object), which the arm must be able to reach. Straight when clear; standoff so
        # the box seats by the teeth.
        target_ball = min((bx for bx in balls if not bx.delivered and bx.held_by is None),
                          key=lambda bx: mag(bx.pos - pos), default=None)
        ignore = (target_ball,) if target_ball is not None else ()
        # Stationary re-centring: once the arm has arrived, nudge the wait point sideways
        # by the object's residual offset from the claw axis so the grab point servos
        # onto it — the object ends up centred between the fingers instead of against one
        # (the usual misgrab). Perpendicular only; depth/seating stays with GRAB_STANDOFF.
        if arrived and dot(vel, vel) < 1e-9:
            tip, arm_dir, up_perp, side = claw_frame(robot["arm2_state"], robot["base_state"]["angle"])
            off = pos - tip
            lateral = up_perp * dot(off, up_perp) + side * dot(off, side)
            camp = camp + lateral * CENTER_GAIN
        elif dot(vel, vel) >= 1e-9 and mag(pos - grab_pt) < CATCH_STANDOFF_DIST:
            # Moving re-centring: the stationary servo above fires ONLY for a still box, so a
            # MOVING box was parked at _intercept_camp (nearest the base) and threaded the fingers
            # at whatever lateral offset the geometry left — measured live as the dominant miss
            # (median path offset ~0.09, a long tail past the ~0.13 catch envelope). A moving box
            # can't be centred on its CURRENT position (it's crossing); instead centre its PATH:
            # shift the wait point sideways by the box path's closest-approach offset to the claw
            # centre-line (resolved in the finger-gap / off-plane axes), so the box crosses dead
            # between the fingers. A deadband lets the arm settle once centred — vital, since the
            # RL close needs _arm_still. Reduces to the stationary servo as vel -> 0.
            tip, arm_dir, up_perp, side = claw_frame(robot["arm2_state"], robot["base_state"]["angle"])
            rel = pos - tip
            r0u, r0s = dot(rel, up_perp), dot(rel, side)    # box's perp offset from the centre-line now
            vu, vs = dot(vel, up_perp), dot(vel, side)      # box's perp velocity
            vv = vu * vu + vs * vs
            if vv > 1e-12:
                tmin = -(r0u * vu + r0s * vs) / vv          # time the path is nearest the centre-line
                if tmin < 0.0:
                    tmin = 0.0                              # closest approach already passing — don't chase
                cu, cs = r0u + vu * tmin, r0s + vs * tmin   # perp offset AT closest approach
            else:
                cu, cs = r0u, r0s                           # head-on down the axis: constant perp offset
            if hypot(cu, cs) > CENTER_DEADBAND:             # deadband -> converge, then let the arm settle
                camp = camp + (up_perp * cu + side * cs) * CENTER_GAIN
        drive_avoiding(robot, _grab_target(robot, camp), ignore=ignore, check_objects=True)

        # Time-to-go close: fire when the box is predicted to reach the grab point
        # within one claw-close-time AND its closest approach actually threads the
        # grasp prism. This self-scales the lead — a fast box triggers earlier, a slow
        # box later — instead of a fixed lead distance that can't serve both a
        # perpendicular sweep (needs a big lead) and a head-on approach (needs none).
        # t_go runs along the box's OWN path, so it never projects sideways and fires
        # early on air. A box whose closest approach misses the prism never fires.
        # Close trigger. RL Grasp (when ON) decides WHEN to close for a MOVING box — the case its
        # policy trained on (claw_env drifts the box −x across the angled claw). It's consulted
        # once the arm is IN POSITION (parked at its intercept, box within the training approach
        # range, claw open / not mid-close) so it can ANTICIPATE the ~7-frame close.
        #
        # A STATIONARY (or near-still) object is deliberately NOT sent to RL: the policy never saw
        # velocity ~0, so it would close before the arm finishes centring — an off-centre grab.
        # Those fall through to the analytic STATIONARY catch below, whose per-frame re-centring
        # servo (see `Stationary re-centring` above) seats a still object accurately, closing only
        # once it is genuinely centred. When RL is OFF, both analytic branches are used as before.
        v2 = dot(vel, vel)
        if rl_grasp_on and v2 > MOVING_SPEED_TOL * MOVING_SPEED_TOL:
            ready = (claw_angle > CATCH_OPEN_ANGLE and robot["close_lock"] <= 0
                     and _arm_still(robot) and mag(pos - grab_pt) <= RL_APPROACH_RANGE)
            centred = bool(_rl_wants_close(robot, pos, vel)) if ready else False
        elif v2 < 1e-9:                                   # stationary: require true centring
            # Close only when the object sits on the claw axis (small lateral offset in BOTH
            # perpendicular directions) at a sensible depth — not merely somewhere inside the
            # loose grab prism, which let a close clip one finger.
            u, v, w = _grab_offsets(robot, pos)
            centred = (GRAB_DEPTH_LO <= u <= GRAB_DEPTH_HI
                       and abs(v) <= GRAB_LATERAL_TOL and abs(w) <= GRAB_LATERAL_TOL)
        else:
            # Moving, but RL off (or below the moving threshold): analytic time-to-go. Lead by
            # THIS claw's real close time from its current opening, not a fixed constant.
            close_time = _claw_close_time(robot)
            t_go = dot(grab_pt - pos, vel) / v2          # time until nearest the grab point
            closest = _claw_grab_distance(robot, pos + vel * t_go)  # how well it threads the prism
            centred = 0.0 <= t_go <= close_time and closest <= CATCH_CLOSE_DIST

        # Parked-but-not-centred fallback: a stationary box the arm can't quite centre
        # closes anyway after a while. Only counts once the object has ARRIVED and the
        # arm is settled; a passing box is handled by `centred` above, not this timer.
        settled = _arm_still(robot)
        open_ready = claw_angle > CATCH_OPEN_ANGLE
        if open_ready and settled and arrived and not centred:
            robot["catch_timer"] += 1                    # arrived but not centred yet
        else:
            robot["catch_timer"] = 0
        timed_out = robot["catch_timer"] > CATCH_CLOSE_TIMEOUT
        if open_ready and (centred or timed_out):
            close_claw(robot)
            robot["catch_timer"] = 0

def auto_deliver_step():
    """When Auto Catch is on, start a delivery cycle whenever a box can move toward
    the centre: an arm holding a box hands it to the adjacent arm one grid hop closer
    to the centre (once that arm is free), and the centre arm places whatever it holds
    at its drop spot. Boxes are walked in to the centre one adjacent hop at a time.
    Several cycles run at once as long as they use disjoint arms, so one box never
    waits on an unrelated hand-off — it advances the moment ITS next arm is free."""
    hops = _hops_to_center()
    holders = [i for i in range(len(robots)) if _is_holding(robots[i])]
    if not holders:
        return
    # Advance the holder already nearest the centre first, so boxes drain inward.
    holders.sort(key=lambda i: hops.get(i, len(robots)))
    claimed = set(_delivery_busy_arms())             # arms already committed this frame
    for ai in holders:
        if ai in claimed:
            continue                                 # this holder is already delivering
        receiver = _next_hop(ai, hops)
        if receiver is None:
            _start_delivery(ai)                      # centre arm — place its box
            claimed.add(ai)
            continue
        # Only hand off to a receiver that is genuinely idle — never yank an arm
        # that's mid-catch, nor one with a box arriving within CATCH_PROTECT_TIME
        # (even if it's a moment out of reach right now), which would abort the grab,
        # nor one already claimed by another hand-off this frame.
        if (receiver not in claimed
                and not _is_holding(robots[receiver])
                and receiver not in _catching_arms
                and receiver not in _catch_protected):
            _start_delivery(ai)                      # next arm toward centre is free
            claimed.update((ai, receiver))

def toggle_auto_catch(b):
    global auto_catch_on
    auto_catch_on = not auto_catch_on
    b.text = 'Auto Catch: ON' if auto_catch_on else 'Auto Catch: OFF'

def _load_grasp_model():
    """Load the trained claw-grasp policy from GRASP_MODEL_FILE (next to this script). Imports
    SB3/torch LAZILY so the headless training import of this file never pays for them. Sets the
    module global _grasp_model to the model, or None if unavailable (None -> auto-catch silently
    falls back to the analytic close trigger). Returns the model (or None)."""
    global _grasp_model
    try:
        import os
        import numpy  # noqa: F401  (ensures numpy is present for predict())
        from stable_baselines3 import SAC
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:                              # exec'd namespace has no __file__
            here = os.getcwd()
        path = os.path.join(here, GRASP_MODEL_FILE)
        if not os.path.exists(path):
            print(f"[RL grasp] model not found at {path} - staying analytic.")
            _grasp_model = None
        else:
            _grasp_model = SAC.load(path)
            print(f"[RL grasp] loaded {GRASP_MODEL_FILE}.")
    except Exception as e:                              # SB3/torch missing, corrupt file, etc.
        print(f"[RL grasp] could not load model ({e}) - staying analytic.")
        _grasp_model = None
    return _grasp_model

def _rl_wants_close(robot, pos, vel):
    """Whether the grasp policy says to CLOSE the claw now, for `robot` catching a box at world
    `pos` moving `vel`. Builds the SAME 8-D observation the model trained on (claw_env): box
    position relative to the claw grab point (3), box velocity (3), claw angle, base angle — all
    world-frame, exactly as in training. Returns True/False, or None if no model is loaded (the
    caller then uses the analytic trigger). Used for CATCHING only; delivery stays analytic."""
    if _grasp_model is None:
        return None
    import numpy as _np
    ba = robot["base_state"]["angle"]
    grab_pt = claw_grab_point(robot["arm2_state"], ba)
    rel = pos - grab_pt
    obs = _np.array([rel.x, rel.y, rel.z, vel.x, vel.y, vel.z,
                     robot["claw_state"]["angle"], ba], dtype=_np.float32)
    action, _ = _grasp_model.predict(obs, deterministic=True)
    return float(action[0]) > GRASP_COMMIT_THRESHOLD

RL_TOGGLE_DEBOUNCE = 0.4    # seconds. The FIRST click that loads the model can block the whole
                            # (single-threaded) main loop for several seconds importing torch/SB3
                            # — with nothing visibly changing meanwhile, an impatient second click
                            # QUEUES behind it and fires immediately once the slow load finishes,
                            # flipping the state right back off with no visible net change (looks
                            # exactly like "the button didn't work"). Ignoring any click that lands
                            # within this window of the PREVIOUS click FINISHING (not starting —
                            # that's what makes this catch a click queued during a long callback,
                            # not just two fast real clicks) defeats that failure mode.
_rl_toggle_last = -1.0

def toggle_rl_grasp(b):
    global rl_grasp_on, _rl_toggle_last
    now = time.time()
    if now - _rl_toggle_last < RL_TOGGLE_DEBOUNCE:
        return                                          # likely a duplicate queued during a slow load
    if _grasp_loading:                                   # the background preload is still running —
        b.text = 'RL Grasp: Loading...'                 # don't kick off a second, redundant load
        _rl_toggle_last = now
        return
    rl_grasp_on = not rl_grasp_on
    if rl_grasp_on and _grasp_model is None:
        b.text = 'RL Grasp: Loading...'                 # best-effort feedback before the slow import
        _load_grasp_model()                            # lazy load (imports torch) — only reached if
                                                        # the background preload hasn't already run
    if rl_grasp_on and _grasp_model is None:
        rl_grasp_on = False                            # load failed -> stay analytic
        b.text = 'RL Grasp: OFF (load failed - see console)'
    else:
        b.text = 'RL Grasp: ON' if rl_grasp_on else 'RL Grasp: OFF'
    _rl_toggle_last = time.time()                        # stamped AFTER the slow work completes

def reload_rl_grasp(b):
    """Reload sac_claw.zip from disk — so a freshly retrained policy takes effect without a
    restart. If RL grasp is currently ON it keeps using the newly loaded model."""
    _load_grasp_model()

def _background_preload_grasp_model():
    """Kick off _load_grasp_model() in a daemon background thread — so the slow ONE-TIME
    torch/stable_baselines3 import happens while the user is looking at the scene, instead of
    blocking the main loop (and stalling the button) on their first RL Grasp click. Purely
    additive: touches no VPython/scene objects (safe alongside the render loop, which only runs
    on the main thread), and toggle_rl_grasp checks `_grasp_loading` so a click that lands before
    this finishes waits for it instead of starting a second, redundant load. Called once at
    startup — see the call site guard for why this only runs in a RENDERED session, never
    during headless training."""
    global _grasp_loading
    def _run():
        global _grasp_loading
        _grasp_loading = True
        _load_grasp_model()
        _grasp_loading = False
    threading.Thread(target=_run, daemon=True).start()

def pid_terms(mgr):
    c = mgr.active
    if isinstance(c, PIDController):
        return c.last_p, c.last_i, c.last_d
    return 0.0, 0.0, 0.0

def caculate_target_arm_angles(x, y, z, armlength):
    # Uses nominal length `armlength` for both links. Arm 2 is deliberately a bit
    # shorter (by the claw grab offset), so the tip lands short of the target and
    # the claw's grab point ends up on it — the tip is not meant to reach the ball.
    r = sqrt(x**2 + y**2)
    d = min(sqrt(r**2 + z**2), 2 * armlength)
    c2 = (d**2 - 2 * armlength**2) / (2 * armlength**2)
    c2 = max(-1.0, min(1.0, c2))
    s2 = sqrt(1 - c2**2)
    arm2_joint = atan2(s2, c2)
    arm1_angle = atan2(z, r) - atan2(s2, 1 + c2)
    arm2_angle = arm1_angle + arm2_joint
    return arm1_angle, arm2_angle


init()

claw_open = True
auto_catch_on = False

# ── RL grasp policy (optional, CATCHING only) ────────────────────────────────────────────────
# Off by default. When ON, the trained claw-grasp model decides WHEN to close the claw during
# auto-catch (replacing the analytic close trigger); the analytic aiming/driving and ALL of
# delivery are unchanged. The model import (torch/SB3) never happens at module load, so the
# headless training import of this file stays fast/lightweight — a RENDERED session instead
# kicks off the load in a BACKGROUND THREAD right below (see _background_preload_grasp_model),
# so the slow one-time import overlaps with the user looking at the scene instead of blocking
# their first click; toggle_rl_grasp still lazily loads on click as a fallback if you toggle
# before that finishes. To update the policy in future: retrain -> overwrite sac_claw.zip ->
# click 'Reload RL' (or restart the sim). Nothing else needs editing.
rl_grasp_on = False
_grasp_model = None
_grasp_loading = False              # True while a (background or click-triggered) load is running
GRASP_MODEL_FILE = 'sac_claw.zip'   # swap this / overwrite the file to update the policy
GRASP_COMMIT_THRESHOLD = 0.0        # action > this -> close (matches claw_env commit_threshold)
RL_APPROACH_RANGE = 1.0             # only consult the policy once the box is within this of the
                                    # grab point (matches claw_env's ~0.9 spawn gap) so it sees
                                    # the approach it trained on and can anticipate the close

if __name__ != "sim_headless":       # skip during headless training (claw_env execs with this
    _background_preload_grasp_model()  # __name__) — only preload for a real/rendered session

# Buttons (shared across all robots)
button(text='End', bind=endtask)
button(text='Pause', bind=toggle_pause)
button(text='Reset', bind=reset)
grab_btn = button(text='Grab', bind=toggle_grab)
claw_btn = button(text='Close Claw', bind=toggle_claw)
button(text='Locate Ball', bind=move_to_ball)
button(text='Catch', bind=catch_ball)
button(text='Pathfind', bind=pathfind_ball)
button(text='Deliver Ball', bind=deliver_ball)
button(text='Auto Catch: OFF', bind=toggle_auto_catch)
button(text='RL Grasp: OFF', bind=toggle_rl_grasp)     # RL decides the catch close (delivery stays analytic)
button(text='Reload RL', bind=reload_rl_grasp)         # re-load sac_claw.zip after retraining

# World coordinate axes
ax_len = 1.5
arrow(pos=vector(0,0,0), axis=vector(ax_len,0,0), color=color.red,   shaftwidth=0.04)
arrow(pos=vector(0,0,0), axis=vector(0,ax_len,0), color=color.green, shaftwidth=0.04)
arrow(pos=vector(0,0,0), axis=vector(0,0,ax_len), color=color.blue,  shaftwidth=0.04)
label(pos=vector(ax_len+0.2,0,0), text='X', color=color.red,   height=12, border=0, box=False, opacity=0)
label(pos=vector(0,ax_len+0.2,0), text='Y', color=color.green, height=12, border=0, box=False, opacity=0)
label(pos=vector(0,0,ax_len+0.2), text='Z', color=color.blue,  height=12, border=0, box=False, opacity=0)

# Build the robots — add more entries (each with its own base position) to scale up.
# 3x2 grid of arms in the y-z plane (x=0), spaced 4 apart: columns y=0,4 x rows z=0,4,8.
robots = [
    make_robot(vector(0, 0, 0), color.orange),
    make_robot(vector(0, 4, 0), color.green),
    make_robot(vector(0, 0, 4), vector(0.5, 0.2, 0.7)),
    make_robot(vector(0, 4, 4), color.yellow),
    make_robot(vector(0, 0, 8), color.white),
    make_robot(vector(0, 4, 8), vector(0.6, 0.3, 0.8)),
]


# The grabbable objects. `balls` is the list the whole system iterates over; add
# more entries to handle multiple objects. Each object carries its own velocity,
# collision radius, and `held_by` (the robot gripping it, or None). `ball` aliases
# the first object so the existing single-object task code keeps working.
# (Collision still uses an inscribed-sphere radius.)
BALL_RADIUS = 0.1   # inscribed-sphere radius of each box — used for surface-range
                    # sensing and collision

def make_ball(col=color.red):
    obj = box(pos=vector(0, 0, 0), size=vector(0.2, 0.2, 0.2), color=col)
    obj.radius = BALL_RADIUS
    obj.velocity = vector(0, 0, 0)
    obj.held_by = None          # the robot currently holding this object, or None
    obj.delivered = False       # placed at the target — ignored by tracking/catching
    obj.reserved = False        # mid hand-off — temporarily hidden from tracking
    return obj

_SCATTER_COLORS = [color.red, color.blue, color.yellow, vector(0.6, 0.2, 0.9)]
_CLUSTER_COLORS = [color.orange, color.green, color.cyan, color.white]

scatter_balls = [make_ball(c) for c in _SCATTER_COLORS]  # scattered far out, drifting in
cluster_balls = [make_ball(c) for c in _CLUSTER_COLORS]  # bunched near the arms
balls = scatter_balls + cluster_balls          # everything tracking/catching/delivering sees
ball = balls[0]                                # alias for the single-object task code

BALL_SPEED_MIN = 0.5   # each box drifts inward (−x) at its own random speed in
BALL_SPEED_MAX = 2.5   # [BALL_SPEED_MIN, BALL_SPEED_MAX]

def _scatter_pos():
    """A far-out spawn point in front of the arm grid (+x, worldX ~8..12), spread across the
    grid's y-z extent (y ~0..4, z ~0..8), from which a box drifts inward (−x) through reach."""
    return vector(random.uniform(8.0, 12.0),      # far out in front of the arms (+x)
                  random.uniform(0.0, 4.0),        # y — across the grid columns
                  random.uniform(0.0, 8.0))        # z — across the grid rows

def spawn_ball(objs=None):
    """Scatter `objs` (default: all boxes) far out in front of the arms (kept well
    apart so no two share a position) and send each drifting inward toward them (−x).
    Spread across the arm grid's y-z extent so every box passes through reach."""
    if objs is None:
        objs = balls
    placed = []
    for obj in objs:
        p = _scatter_pos()
        for _ in range(50):                        # retry until clear of the others (>=1 apart)
            if all(mag(p - q) >= 1.0 for q in placed):
                break
            p = _scatter_pos()
        placed.append(p)
        obj.pos = p
        obj.velocity = vector(-random.uniform(BALL_SPEED_MIN, BALL_SPEED_MAX), 0, 0)  # drift in (−x)
        obj.held_by = None
        obj.delivered = False
        obj.reserved = False
        obj.visible = True

CLUSTER_RADIUS = 0.8      # boxes spawn within this radius of a random cluster centre
CLUSTER_MIN_SEP = 0.35    # ...but no closer than this to each other, so they don't overlap
CLUSTER_VELOCITY = vector(0, 0, 0)  # the whole cluster drifts at this velocity (zero = stationary)

def _cluster_offset():
    """A random offset uniformly within CLUSTER_RADIUS in the x-z plane (y stays 0)."""
    ang = random.uniform(0, 2 * pi)
    r = CLUSTER_RADIUS * sqrt(random.random())     # sqrt -> uniform over the disc area
    return vector(r * cos(ang), 0, r * sin(ang))

def spawn_cluster(objs=None):
    """Spawn `objs` (default: all boxes) bunched into one small area near the arms,
    all sharing CLUSTER_VELOCITY (zero for now, so the cluster sits still). A stress
    test for the tracker's ability to separate nearby objects — where the triangle-
    inequality prune can't help and only the trilateration residual tells them apart."""
    if objs is None:
        objs = balls
    # x capped at 2.0 so that even with the +CLUSTER_RADIUS offset, a stationary box at a
    # grid mid-point (y~2 between columns, z~2/6 between rows — 2 units from every corner)
    # stays within reach.
    center = vector(random.randint(3, 4) / 2,     # x ~1.5..2.0, in front of the arms
                    random.randint(0, 8) / 2,      # y ~0..4, across the grid columns
                    random.randint(0, 16) / 2)     # z ~0..8, across the grid rows
    placed = []
    for obj in objs:
        p = center + _cluster_offset()
        for _ in range(50):                        # retry until clear of the others
            if all(mag(p - q) >= CLUSTER_MIN_SEP for q in placed):
                break
            p = center + _cluster_offset()
        placed.append(p)
        obj.pos = p
        obj.velocity = vector(CLUSTER_VELOCITY)     # whole cluster shares one velocity
        obj.held_by = None
        obj.delivered = False
        obj.reserved = False
        obj.visible = True

def spawn_all():
    """Spawn both groups at once: the scattered boxes far out and the cluster near
    the arms. The single source of truth for the boxes' initial state."""
    spawn_ball(scatter_balls)
    spawn_cluster(cluster_balls)

spawn_all()

# Sensors (no collisions — purely visual, with their positions tracked)
sensor = box(pos=vector(0, 0, 0), size=vector(0.3, 0.2, 0.3), color=color.gray(0.4))
sensor_state = {"pos": vector(0, 0, 0)}

sensor2 = box(pos=vector(0, -1, 0), size=vector(0.3, 0.2, 0.3), color=color.gray(0.4))
sensor2_state = {"pos": vector(0, -1, 0)}

sensor3 = box(pos=vector(-1, 0, 0), size=vector(0.3, 0.2, 0.3), color=color.gray(0.4))
sensor3_state = {"pos": vector(-1, 0, 0)}

# 4th sensor sits one unit down the -z axis, so the four sensors are the origin plus a
# unit step along each of -x, -y, -z. This spans all three axes (non-coplanar): a
# coplanar array can't resolve the out-of-plane direction — which is exactly where the
# objects are spread — and that made clustered objects produce exactly-consistent ghost
# detections. Giving the array a real z baseline breaks that ambiguity.
sensor4 = box(pos=vector(0, 0, -1), size=vector(0.3, 0.2, 0.3), color=color.gray(0.4))
sensor4_state = {"pos": vector(0, 0, -1)}

# Sensor order no longer matters — trilateration is now a general least-squares solve.
sensors = [sensor_state, sensor3_state, sensor2_state, sensor4_state]

# Multi-object tracker: fed the raw (untagged) sensor ranges every frame, it
# detects and Kalman-tracks every object, exposing confirmed positions/velocities.
tracker = MultiTracker(sensors, radius=BALL_RADIUS)   # sensors read surface ranges

# Diagnostic graphs track the first robot
angle_graph = graph(title='Arm Angle Over Time', xtitle='Time (s)', ytitle='Angle (degrees)',
                    width=400, height=280, align='right')
angle_curve = gcurve(graph=angle_graph, color=color.blue, label='Arm 1')
angle_curve2 = gcurve(graph=angle_graph, color=color.cyan, label='Arm 2')
target_curve = gcurve(graph=angle_graph, color=color.red, label='target')

height_graph = graph(title='Tip Height Above Base', xtitle='Time (s)', ytitle='Height',
                     width=400, height=280, align='right')
height_curve = gcurve(graph=height_graph, color=color.orange)

p_graph = graph(title='P Term', xtitle='Time (s)', ytitle='P', width=400, height=200, align='right')
p_curve1 = gcurve(graph=p_graph, color=color.blue, label='Arm 1')
p_curve2 = gcurve(graph=p_graph, color=color.cyan, label='Arm 2')

i_graph = graph(title='I Term', xtitle='Time (s)', ytitle='I', width=400, height=200, align='right')
i_curve1 = gcurve(graph=i_graph, color=color.blue, label='Arm 1')
i_curve2 = gcurve(graph=i_graph, color=color.cyan, label='Arm 2')

d_graph = graph(title='D Term', xtitle='Time (s)', ytitle='D', width=400, height=200, align='right')
d_curve1 = gcurve(graph=d_graph, color=color.blue, label='Arm 1')
d_curve2 = gcurve(graph=d_graph, color=color.cyan, label='Arm 2')

base_graph = graph(title='Base Rotation vs Target', xtitle='Time (s)', ytitle='Angle (degrees)',
                   width=400, height=280, align='right')
base_angle_curve = gcurve(graph=base_graph, color=color.orange, label='Base')
base_target_curve = gcurve(graph=base_graph, color=color.red,    label='Target')

# One coordinate input + base-angle readout per manually-driven robot. Arms beyond the
# first two are auto-catch/delivery-only, so they get no input (their widgets stay None).
for idx, robot in enumerate(robots):
    if idx >= 2:
        continue
    scene.append_to_caption(f'<br>Robot {idx+1} coords (x,y,z): ')
    robot["coord_input"] = winput(bind=lambda w, r=robot: input_coordinates(w, r), text='1,0,0', width=120)
    scene.append_to_caption('  base angle: ')
    robot["base_label"] = wtext(text=' 0.000 rad')

t = 0
_diag_frame = 0                       # frame counter, for throttling diagnostics
PLOT_EVERY = 5                        # update the diagnostic graphs every Nth frame only
ball_velocity_est = vector(0, 0, 0)   # ball velocity from the sensors, sampled each frame
while True:
    rate(100)
    if paused:                        # frozen: skip physics/logic, but keep the window
        continue                      # responsive so Resume (and End) still work
    _col_tick += 1                    # invalidate the per-frame sweep-collision cache

    # Advance every free object once; a holder (if any) repositions its object in
    # step_robot, so held objects (held_by set) are left alone here. Delivered
    # boxes stay put at the target.
    for obj in balls:
        if obj.held_by is None and not obj.delivered:
            obj.pos += obj.velocity * dt

    # Keep the sensor states tracking the sensors' positions
    sensor_state["pos"] = sensor.pos
    sensor2_state["pos"] = sensor2.pos
    sensor3_state["pos"] = sensor3.pos
    sensor4_state["pos"] = sensor4.pos

    for robot in robots:
        step_robot(robot)

    # Sample the ball's estimated velocity once per frame, so boxvelocity's
    # internal previous-position sample stays exactly one dt old.
    ball_velocity_est = vector(*boxvelocity(sensors, ball))

    # Advance the multi-object tracker on this frame's raw sensor ranges. It runs
    # detection (exact-cover, ghost-free) + association + Kalman filtering, so
    # tracker.confirmed_tracks() holds a filtered estimate per object. Only FREE
    # boxes are fed in: held boxes (already caught), delivered boxes, and mid-hand-
    # off (reserved) boxes are withheld so no other arm gets them assigned and tries
    # to close on a box it can't grip.
    live_balls = [b for b in balls
                  if b.held_by is None and not b.delivered and not b.reserved]
    tracker.step(sensor_distances(sensors, live_balls))

    # Diagnostics: once a second, print the 10 most promising tracks — their current
    # location and a 0..1 certainty (from the filter covariance; 1 = very sure).
    _diag_frame += 1
    if _diag_frame % 100 == 0:
        top = tracker.top_tracks(10)
        print(f"\n[t={t:6.2f}s] top {len(top)} tracks (most promising first):")
        for k, tr in enumerate(top, 1):
            x, y, z = tr.position
            status = "conf" if tr.confirmed else "tent"
            print(f"  {k:2d}. ({x:6.2f},{y:6.2f},{z:6.2f})  certainty={tr.certainty:.3f}"
                  f"   [{status} hits={tr.hits} miss={tr.misses} std={tr.position_std:.3f}]")

    # Auto-catch mode: drive the free arms to catch first (records who's catching),
    # then start any delivery that won't hijack a catching arm.
    if auto_catch_on:
        auto_catch_step()
        auto_deliver_step()

    # Follow an active planned pathfinding route (waypoint -> waypoint -> target)
    pathfind_step()

    # Progress an active "move to ball" (standoff -> onto the ball)
    if locate_task is not None:
        lr = locate_task["robot"]
        tgt = locate_task["target"]
        if locate_task["stage"] == "standoff":
            standoff = _rotated_standoff(lr, tgt, LOCATE_STANDOFF_ANGLE)
            if _grab_point_near(lr, standoff, tol=0.25):
                command_arm_to(lr, _grab_target(lr, tgt))   # now swing the base onto the ball
                locate_task["stage"] = "onto"
        elif locate_task["stage"] == "onto":
            if _grab_point_near(lr, tgt, tol=0.15):
                locate_task = None                   # arrived

    # Progress every active delivery cycle (they run concurrently on disjoint arms).
    # Safety: a cycle should never run indefinitely — if one does (e.g. a receiver's
    # grip keeps missing), abort it, freeing its box back into normal tracking/catching
    # and releasing its arms so nothing stays frozen.
    for task in list(handoff_tasks):
        task["total"] = task.get("total", 0) + 1
        if task["total"] > HANDOFF_TIMEOUT:
            _bx = task.get("box")
            if _bx is not None:
                _bx.reserved = False
            _free_task_arms(task)
            handoff_tasks.remove(task)
            continue
        if _progress_delivery(task):
            handoff_tasks.remove(task)

    # Diagnostics for the first robot
    r0 = robots[0]
    deg1 = degrees(r0["arm_state"]["angle"])
    deg2 = degrees(r0["arm2_state"]["angle"])
    scene.caption = f"Arm 1: {deg1:.2f}°  |  Arm 2: {deg2:.2f}°"
    # Graph updates are throttled — VPython plots aren't free, and every Nth point is
    # visually identical for these slowly-varying signals. The plot-only work (heights,
    # PID terms) is done only on plotting frames too.
    if _diag_frame % PLOT_EVERY == 0:
        angle_curve.plot(t, deg1)
        angle_curve2.plot(t, deg2)
        target_curve.plot(t, r0["arm1_target_deg"])
        a2s = r0["arm2_state"]
        tip2_y = a2s["anchor"].y + a2s["length"] * sin(a2s["angle"])
        height_curve.plot(t, tip2_y - r0["anchor"].y)
        p1, i1, d1 = pid_terms(r0["manager"])
        p2, i2, d2 = pid_terms(r0["manager2"])
        p_curve1.plot(t, p1); p_curve2.plot(t, p2)
        i_curve1.plot(t, i1); i_curve2.plot(t, i2)
        d_curve1.plot(t, d1); d_curve2.plot(t, d2)
        base_angle_curve.plot(t, degrees(r0["base_state"]["angle"]))
        base_target_curve.plot(t, r0["base_target_deg"])
    t += dt
