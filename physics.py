import math
from vpython import *

dt = 0.01
density = 1.22
FRICTION = 200            # Coulomb friction coefficient (μ) between box and claw
NORMAL_STIFFNESS = 250   # squeeze -> normal-force gain (higher = firmer grip per unit press)
RESTITUTION = 0.1        # box bounciness on impact: 0 = no bounce, 1 = fully elastic

# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def _arm_dir(angle, base_angle):
    """Unit direction of an arm raised by `angle` and rotated about the base
    by `base_angle`."""
    fwd = vector(math.cos(base_angle), 0, math.sin(base_angle))
    return fwd * math.cos(angle) + vector(0, math.sin(angle), 0)

def _claw_basis(arm2_state, base_angle):
    """Frame the claw rotates in: (tip, arm_dir, up_perp) for the second arm.
    up_perp is perpendicular to arm_dir in the arm's vertical plane."""
    angle = arm2_state["angle"]
    fwd = vector(math.cos(base_angle), 0, math.sin(base_angle))
    arm_dir = fwd * math.cos(angle) + vector(0, math.sin(angle), 0)
    up_perp = -fwd * math.sin(angle) + vector(0, math.cos(angle), 0)
    tip = arm2_state["anchor"] + arm2_state["length"] * arm_dir
    return tip, arm_dir, up_perp

def arm_tip_3d(arm_state, base_angle):
    return arm_state["anchor"] + arm_state["length"] * _arm_dir(arm_state["angle"], base_angle)

def claw_frame(arm2_state, base_angle):
    """Orthonormal frame fixed to the claw: (tip, arm_dir, up_perp, side).
    Used to hold a gripped object rigidly relative to the claw."""
    tip, arm_dir, up_perp = _claw_basis(arm2_state, base_angle)
    return tip, arm_dir, up_perp, cross(arm_dir, up_perp)

def apply_base_rotation(arm, arm_state, base_angle):
    L = arm_state["length"]
    arm_dir = _arm_dir(arm_state["angle"], base_angle)
    arm.axis = L * arm_dir
    arm.pos = arm_state["anchor"] + (L / 2) * arm_dir

def translate_degree(targetx, targety):
    return math.atan2(targety, targetx)

# ─────────────────────────────────────────────
#  Arm dynamics
# ─────────────────────────────────────────────

def update_arm(arm, arm_state):
    angle = arm_state["angle"]
    L = arm_state["length"]
    I = (1 / 3) * arm_state["mass"] * L ** 2
    # Gravity torque about the anchor (weight acts at the centre, L/2)
    torque = -arm_state["mass"] * 9.8 * (L / 2) * math.cos(angle)
    arm_state["angular_velocity"] += (torque / I) * dt
    arm_state["angle"] += arm_state["angular_velocity"] * dt

    angle = arm_state["angle"]
    arm.axis = vector(L * math.cos(angle), L * math.sin(angle), 0)
    arm.pos = arm_state["anchor"] + vector(L/2 * math.cos(angle), L/2 * math.sin(angle), 0)

def drag(arm, arm_state):
    L = arm_state["length"]
    av = arm_state["angular_velocity"]
    area = L * arm_state["width"]
    velocity = av * L
    drag_force = 0.5 * density * velocity * velocity * area * 1.28
    angular_acc = (drag_force / arm_state["mass"]) / L
    if av > 0:
        arm_state["angular_velocity"] -= angular_acc * dt
    elif av < 0:
        arm_state["angular_velocity"] += angular_acc * dt

def update_base(base_state):
    base_state["angle"] = (base_state["angle"] + base_state["angular_velocity"] * dt) % (2 * pi)

# ─────────────────────────────────────────────
#  Claw
# ─────────────────────────────────────────────

def claw_grab_point(arm2_state, base_angle):
    tip, arm_dir, _ = _claw_basis(arm2_state, base_angle)
    # How far out along the fingers the object is held. The fingers are 0.4 long
    # with the teeth hooking in at the tip, so ~0.3 seats the object near the
    # teeth instead of jamming it in the corner at the finger base (tip).
    return tip + arm_dir * 0.3

def _closest_point_on_segment(p, a, b):
    ab = b - a
    denom = dot(ab, ab)
    if denom == 0:
        return a
    t = max(0.0, min(1.0, dot(p - a, ab) / denom))
    return a + t * ab

def _segment_box_contact(a, b, box_obj, margin=0.04):
    """Contact between a finger segment AB and the axis-aligned box `box_obj`
    (grown by the finger half-thickness `margin`). Returns
    (hit, contact_point, normal, depth); `normal` is the box face normal that
    pushes the box away from the finger, and `depth` the penetration along it."""
    c = box_obj.pos
    q = _closest_point_on_segment(c, a, b)   # finger point nearest the box centre
    delta = q - c                            # box centre -> finger point
    hx = box_obj.size.x / 2 + margin
    hy = box_obj.size.y / 2 + margin
    hz = box_obj.size.z / 2 + margin
    px = hx - abs(delta.x)
    py = hy - abs(delta.y)
    pz = hz - abs(delta.z)
    if px <= 0 or py <= 0 or pz <= 0:
        return False, q, vector(0, 0, 0), 0.0
    # Penetrating — separate along the least-penetrated axis (i.e. a box face)
    if px <= py and px <= pz:
        n = vector(-1.0 if delta.x > 0 else 1.0, 0, 0); depth = px
    elif py <= pz:
        n = vector(0, -1.0 if delta.y > 0 else 1.0, 0); depth = py
    else:
        n = vector(0, 0, -1.0 if delta.z > 0 else 1.0); depth = pz
    return True, q, n, depth

def apply_claw(finger1, finger2, tooth1, tooth2, arm2_state, angle1, angle2, base_angle):
    tip, arm_dir, up_perp = _claw_basis(arm2_state, base_angle)
    finger_length = 0.4
    tooth_length = 0.15

    for finger, tooth, sign, ang in ((finger1, tooth1, 1, angle1), (finger2, tooth2, -1, angle2)):
        ca, sa = math.cos(ang), math.sin(ang)
        finger_dir = arm_dir * ca + sign * up_perp * sa
        ftip = tip + finger_dir * finger_length
        finger.pos = tip + finger_dir * (finger_length / 2)
        finger.axis = finger_dir * finger_length
        # Tooth hooks 90° inward from the finger tip
        tooth_dir = arm_dir * sa - sign * up_perp * ca
        tooth.pos = ftip + tooth_dir * (tooth_length / 2)
        tooth.axis = tooth_dir * tooth_length

def finger_contacts_ball(arm2_state, finger_angle, base_angle, ball_obj, sign):
    """True if the finger (or its tooth) at `finger_angle` overlaps the box.
    Pure test — modifies nothing. Used to decide how far a finger may close."""
    tip, arm_dir, up_perp = _claw_basis(arm2_state, base_angle)
    fl, tl = 0.4, 0.15
    ca, sa = math.cos(finger_angle), math.sin(finger_angle)
    fd = arm_dir * ca + sign * up_perp * sa
    ftip = tip + fd * fl
    td = arm_dir * sa - sign * up_perp * ca
    for a, b in ((tip, ftip), (ftip, ftip + td * tl)):
        if _segment_box_contact(a, b, ball_obj)[0]:
            return True
    return False

def resolve_finger_ball_collision(arm1_state, arm2_state, base_state, finger_angle, claw_av, ball_obj, sign):
    """Collide a single finger (and its tooth) with the ball. `sign` selects the
    finger (+1 / -1). Returns True if it made contact. The contact-point velocity
    includes base + both arm joints + claw motion, so friction carries the ball
    along as the whole arm moves (not just when the claw closes)."""
    base_angle = base_state["angle"]
    tip, arm_dir, up_perp = _claw_basis(arm2_state, base_angle)
    fwd = vector(math.cos(base_angle), 0, math.sin(base_angle))
    axis1 = cross(fwd, vector(0, 1, 0))   # axis the arms/claw pivot around
    T1 = arm2_state["anchor"]             # arm2 (and claw chain) pivot
    omega_b = vector(0, -base_state["angular_velocity"], 0)   # base spin (about Y)
    omega_1 = arm1_state["angular_velocity"] * axis1
    omega_2 = arm2_state["angular_velocity"] * axis1
    fl, tl = 0.4, 0.15
    ca, sa = math.cos(finger_angle), math.sin(finger_angle)
    elastic = RESTITUTION
    collided = False

    # Off-plane axis of the claw (perpendicular to the fingers' sweep plane). _segment_box_contact
    # returns a WORLD-axis-aligned face normal; at a diagonal (≈45°) claw orientation that world
    # axis is diagonal to the gripper, so the contact push/impulse acquires a large component along
    # `side` and ejects an otherwise dead-centred box OUT of the grasp plane before both fingers can
    # seat (measured: off-plane offset driven past the ~0.13 grip limit exactly at 45°). A thin
    # two-finger claw can't physically shove a box out of its own grasp plane, so we strip the
    # off-plane component from THIS contact's net response, keeping only the in-plane (depth +
    # finger-gap) part that actually seats/centres the box. At axis-aligned orientations `side`
    # already carries ~none of the response, so this leaves those (already-working) grips unchanged.
    side = cross(arm_dir, up_perp)
    sm = mag(side)
    side = side / sm if sm > 1e-9 else vector(0, 0, 0)

    fd = arm_dir * ca + sign * up_perp * sa
    ftip = tip + fd * fl
    td = arm_dir * sa - sign * up_perp * ca
    segments = ((tip, ftip), (ftip, ftip + td * tl))   # finger, tooth

    for a, b in segments:
        hit, closest, normal, depth = _segment_box_contact(a, b, ball_obj)
        if hit:
            collided = True
            # snapshot (explicit copies — avoid aliasing the live vpython .pos/.velocity) so we
            # can strip the off-plane part of this contact's net effect afterward
            p, v = ball_obj.pos, ball_obj.velocity
            pos0 = vector(p.x, p.y, p.z)
            vel0 = vector(v.x, v.y, v.z)
            ball_obj.pos = ball_obj.pos + normal * depth   # push the box out along the face normal
            # Velocity of the finger material at the contact point (full chain:
            # base + arm1 (pivot O) + arm2 (pivot T1) + claw (pivot tip))
            v_contact = (cross(omega_b, closest) + cross(omega_1, closest)
                         + cross(omega_2, closest - T1)
                         + claw_av * cross(axis1, closest - tip))
            v_rel_n = dot(ball_obj.velocity - v_contact, normal)
            # Normal force this contact: a penalty term from the squeeze
            # (∝ penetration depth)...
            jn = NORMAL_STIFFNESS * depth
            if v_rel_n < 0:                    # ...plus a restitution impulse while approaching
                jimp = -(1 + elastic) * v_rel_n
                ball_obj.velocity = ball_obj.velocity + jimp * normal
                jn += jimp
            # Coulomb friction: tangential impulse capped at μ·(normal force).
            # A firm squeeze (large jn) removes all tangential slip; a light
            # touch (small jn) lets the box slide.
            v_rel = ball_obj.velocity - v_contact
            v_t = v_rel - dot(v_rel, normal) * normal
            vt_mag = mag(v_t)
            if vt_mag > 1e-9:
                jt = min(vt_mag, FRICTION * jn)
                ball_obj.velocity = ball_obj.velocity - (jt / vt_mag) * v_t
            # Strip the off-plane (`side`-axis) part of this contact's net effect (see above).
            dpos = ball_obj.pos - pos0
            dvel = ball_obj.velocity - vel0
            ball_obj.pos = pos0 + (dpos - dot(dpos, side) * side)
            ball_obj.velocity = vel0 + (dvel - dot(dvel, side) * side)
    return collided

CLAW_MAX_AV = 16  # cap on the claw's angular speed; the close is velocity-limited (the huge
                  # torque/inertia ratio pins the fingers at this cap even near the shut
                  # floor), so this IS the close speed. simulation._claw_close_time reads it,
                  # so the catch-timing lead auto-scales with it. The empty-claw recovery is
                  # gated on the claw's ANGLE (CLAW_SHUT_ANGLE) and on hold-time frame counts
                  # (CLOSE_LOCK_FRAMES / EMPTY_CLOSE_FRAMES), not on the close duration, so
                  # they need no change; the bottomed-out grip still catches a box even if a
                  # fast close skips the mid-close contact frame.

def update_claw(claw_state):
    av = max(-CLAW_MAX_AV, min(CLAW_MAX_AV, claw_state["angular_velocity"]))
    claw_state["angular_velocity"] = av * 0.98
    claw_state["angle"] += claw_state["angular_velocity"] * dt
    claw_state["angle"] = max(0.45, min(math.pi / 2, claw_state["angle"]))
