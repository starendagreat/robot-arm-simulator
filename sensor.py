from vpython import mag, sqrt
from itertools import product
import numpy as np
dt = 0.01
def _surface_range(sensor_pos, obj):
    """Distance from a sensor to the SURFACE of an object, like a real range sensor:
    the distance to the object centre minus its radius (the object is modelled as a
    sphere of `obj.radius`, matching how the rest of the code treats it). Clamped at 0
    so a sensor can't read through the object. To recover the CENTRE, a consumer adds
    the radius back — the same constant it was reduced by, so the centre is exact."""
    r = getattr(obj, "radius", 0.0)
    return max(0.0, mag(obj.pos - sensor_pos) - r)

def _object_distances(sensors, box_obj):
    """Surface range from each sensor to one object, in sensor order (see
    `_surface_range`). `trilateration` adds the radius back to recover the centre."""
    return [_surface_range(s["pos"], box_obj) for s in sensors]

def sensor_distances(sensors, objects):
    """All ranges from the sensors to all objects — each the distance to the object's
    SURFACE (centre distance minus the object's radius), like a real distance sensor.

    Parameters
    ----------
    sensors : iterable of sensor-state dicts, each holding a "pos" vector.
    objects : iterable of objects (each with a `.pos` centre and a `.radius`).

    Returns one entry per sensor (in sensor order); each  the radius back (the
    MultiTracker does this via its `radius`).entry is a flat list of that
    sensor's surface ranges to the objects. The ranges are NOT tagged with which object
    they came from — associating each range back to an object is left to later
    processing. A consumer that wants object CENTRES adds
    """
    return [[_surface_range(s["pos"], obj) for obj in objects] for s in sensors]

_trilat_geom = {"key": None}   # cached range-independent multilateration terms

def _trilat_geometry(sensor_positions):
    """Cache the parts of the multilateration that depend only on the SENSOR LAYOUT:
    the pseudo-inverse of A = 2(s_i - s_0) and the |s|^2 offsets. The sensors don't move
    between candidate range-tuples (or, usually, between frames), so this pseudo-inverse
    (an SVD) is computed once and reused for every solve — detection enumerates up to
    N**(#sensors) tuples per frame, so this turns each from a least-squares solve into a
    single matrix-vector product. Recomputed only if the sensor positions change."""
    S = np.array([[s.x, s.y, s.z] for s in sensor_positions], dtype=float)
    key = S.tobytes()
    if _trilat_geom["key"] != key:
        s0 = S[0]
        A = 2.0 * (S[1:] - s0)                   # rows: 2(s_i - s_0)
        _trilat_geom.update(key=key,
                            pinvA=np.linalg.pinv(A),         # solve A·p = b as pinvA @ b
                            s1sq=np.sum(S[1:] ** 2, axis=1), # |s_i|^2 for i >= 1
                            s0sq=float(s0 @ s0))             # |s_0|^2
    return _trilat_geom

def trilaterate_from_ranges(sensor_positions, ranges):
    """Least-squares multilateration from raw ranges (one per sensor) to a single
    point, for ANY sensor geometry — including a non-coplanar array.

    Subtracting the first sensor's range equation from each of the others cancels the
    quadratic |p|^2 term and linearises |p - s_i| = r_i into A·p = b. A depends only on
    the sensor layout, so its pseudo-inverse is precomputed once (see _trilat_geometry)
    and each solve is just pinvA @ b. Works for >=4 sensors and resolves all three axes
    when the sensors span 3D. (The old closed form assumed the sensors were coplanar with
    a canonical origin/+x/+y layout, so the out-of-plane axis was only recovered through
    an ambiguous sqrt — which made mismatched range-tuples from clustered objects exactly
    consistent and produced ghost detections.) Returns (x, y, z)."""
    g = _trilat_geometry(sensor_positions)
    r = np.asarray(ranges, dtype=float)
    b = (g["s1sq"] - r[1:] ** 2) - (g["s0sq"] - r[0] ** 2)
    p = g["pinvA"] @ b
    return (float(p[0]), float(p[1]), float(p[2]))

def trilateration(sensors, box_obj):
    sensor_positions = [s["pos"] for s in sensors]
    r = getattr(box_obj, "radius", 0.0)
    # Sensors read the surface; add the radius back to each range to locate the centre.
    centre_ranges = [d + r for d in _object_distances(sensors, box_obj)]
    return trilaterate_from_ranges(sensor_positions, centre_ranges)

def tuple_residual(sensors, ranges):
    """Gating / consistency score for a candidate range-tuple — one range per
    sensor, hypothesised to come from a *single* object.

    We solve the tuple by least-squares multilateration (over-determined: 4+ ranges,
    3 unknowns), then measure how far the resulting point's true range to *each*
    sensor is from the measured range, and take the worst one. If the ranges genuinely
    belong to one object the point reproduces them all and the residual is ~0; a
    mismatched tuple (ranges from different objects) can't be fit by any single point
    once the sensors span 3D, so at least one sensor is left badly off and the residual
    is large.

    Returns (position, residual). The caller gates by comparing residual to a
    tolerance; small = a plausible detection, large = reject the tuple.
    """
    sensor_positions = [s["pos"] for s in sensors]
    pos = trilaterate_from_ranges(sensor_positions, ranges)
    px, py, pz = pos
    residual = 0.0
    for s, r in zip(sensors, ranges):
        sp = s["pos"]
        d = sqrt((px - sp.x)**2 + (py - sp.y)**2 + (pz - sp.z)**2)
        residual = max(residual, abs(d - r))
    return pos, residual

GATE_TOL = 0.05   # max per-sensor range error (world units) for a tuple to be
                  # accepted as a single-object detection

def _sensor_gaps(sensors):
    """Pairwise distances |s_i - s_j| between sensors. This is the most a single
    object's ranges to those two sensors can differ by (triangle inequality), so it
    caps |r_i - r_j| for any real detection."""
    return [[mag(a["pos"] - b["pos"]) for b in sensors] for a in sensors]

def _one_object_possible(ranges, gaps, margin):
    """True if `ranges` could come from ONE object: every pair obeys
    |r_i - r_j| <= |s_i - s_j| (+ margin for measurement noise). A cheap NECESSARY
    condition used to reject impossible range-tuples before the costly trilateration —
    ghost tuples that splice ranges from different objects usually fail it outright."""
    n = len(ranges)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(ranges[i] - ranges[j]) > gaps[i][j] + margin:
                return False
    return True

def initial_tracks(sensors, measurements, gate_tol=GATE_TOL):
    """Stage 2 / detection core — recover object positions from the untagged
    per-sensor ranges using the global range-assignment (exact-cover) constraint.

    Each real object contributes exactly one range to each sensor, so the correct
    scene is a set of consistent range-tuples that *partition* the measurements:
    every sensor's ranges each used once, none reused. A ghost tuple is built by
    reusing ranges that already belong to real objects, so it cannot fit into that
    partition alongside them.

    We enumerate every candidate tuple (one range-slot per sensor), keep the ones
    that trilaterate consistently, then greedily select the lowest-residual tuples
    that are mutually range-disjoint (no sensor slot reused). The real tuples
    (residual ~0) are taken first and consume every slot; conflicting ghosts are
    then rejected. This resolves ghosts on a single frame — even a stationary one,
    where temporal persistence can't help.

    Returns a list of (x, y, z) position estimates, one per detected object.

    Cost is N**(#sensors) tuples (N objects) in the worst case, but the triangle-
    inequality prune below discards most impossible tuples before the trilateration,
    so in practice only the near-feasible ones pay the geometric solve.
    """
    n_sensors = len(measurements)
    gaps = _sensor_gaps(sensors)   # per-pair cap on a real object's range differences
    # Enumerate one range-slot per sensor; keep the geometrically consistent tuples,
    # remembering which slot (sensor i, range index j) each one used.
    consistent = []   # (residual, pos, slots)
    slot_choices = [range(len(m)) for m in measurements]
    for idx in product(*slot_choices):
        ranges = [measurements[i][idx[i]] for i in range(n_sensors)]
        # Triangle-inequality prune: a tuple whose ranges differ by more than the
        # sensor spacing can't be one object — skip it before the costly trilateration.
        if not _one_object_possible(ranges, gaps, gate_tol):
            continue
        pos, residual = tuple_residual(sensors, ranges)
        if residual <= gate_tol:
            slots = tuple((i, idx[i]) for i in range(n_sensors))
            consistent.append((residual, pos, slots))

    # Greedily accept the best tuples that don't reuse any sensor slot (exact
    # cover). Real objects, being exactly consistent, sort first and use up every
    # slot, leaving ghosts with nothing but conflicts.
    consistent.sort(key=lambda c: c[0])
    used = set()
    detected = []
    for residual, pos, slots in consistent:
        if any(slot in used for slot in slots):
            continue
        used.update(slots)
        detected.append(pos)
    return detected

# ─────────────────────────────────────────────
#  Stage 3 — Kalman track (constant-velocity model)
# ─────────────────────────────────────────────

def _cv_matrices(dt, accel_var, meas_var):
    """Build the constant-velocity Kalman matrices for state [px,py,pz,vx,vy,vz]
    and position-only measurement [px,py,pz]."""
    F = np.eye(6)                       # x_{k+1} = x_k + v*dt
    for i in range(3):
        F[i, i + 3] = dt
    H = np.zeros((3, 6))                # we only measure position
    for i in range(3):
        H[i, i] = 1.0
    # White-noise-acceleration process noise, per axis.
    Q = np.zeros((6, 6))
    q = accel_var
    for i in range(3):
        Q[i, i]         = q * dt**4 / 4
        Q[i, i + 3]     = q * dt**3 / 2
        Q[i + 3, i]     = q * dt**3 / 2
        Q[i + 3, i + 3] = q * dt**2
    R = np.eye(3) * meas_var
    return F, H, Q, R

class KalmanTrack:
    """Tracks one object in 3D with a constant-velocity Kalman filter.

    State  x = [px, py, pz, vx, vy, vz];  measurement z = [px, py, pz] (a
    trilaterated position). Seeded from a single position, so velocity starts at
    zero with a *large* covariance — the filter learns it from the first few
    position updates rather than trusting the initial guess.

    The sensor array is non-coplanar, so all three axes (z-sign included) are fully
    observable — the measured position is unambiguous and no +z half-space assumption
    is needed.
    """
    def __init__(self, pos, dt=dt, accel_std=5.0, meas_std=0.02,
                 init_pos_std=0.1, init_vel_std=10.0):
        self.dt = dt
        self.F, self.H, self.Q, self.R = _cv_matrices(dt, accel_std**2, meas_std**2)
        self.x = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([init_pos_std**2] * 3 + [init_vel_std**2] * 3).astype(float)
        # Lifecycle bookkeeping (managed by MultiTracker, Stage 5).
        self.hits = 1          # detections associated so far (seeded from one)
        self.misses = 0        # consecutive frames since the last detection
        self.age = 0           # frames since the track was created
        self.confirmed = False # promoted to a real object once it persists

    def predict(self):
        """Advance the state one frame (call once per frame, before update)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.position

    def update(self, meas):
        """Fuse a measured position (x, y, z) into the state."""
        z = np.asarray(meas, dtype=float)
        y = z - self.H @ self.x                     # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)    # Kalman gain
        self.x = self.x + K @ y
        # Joseph-form covariance update: P = (I-KH)P(I-KH)^T + K R K^T.
        # Algebraically equal to (I-KH)P but stays symmetric positive-definite
        # under round-off, so P doesn't drift/blow up over long runs.
        IKH = np.eye(6) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        return self.position

    def mahalanobis(self, meas):
        """Normalised (innovation-covariance-scaled) distance of a measurement to
        this track's predicted position. Call after predict(); Stage 4 uses it to
        gate/associate measurements to tracks."""
        z = np.asarray(meas, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(np.sqrt(y @ np.linalg.inv(S) @ y))

    @property
    def position(self):
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def velocity(self):
        return (float(self.x[3]), float(self.x[4]), float(self.x[5]))

    @property
    def position_std(self):
        """RMS positional uncertainty (world units) from the covariance — the smaller
        it is, the more sure the filter is of where the object is."""
        return float(np.sqrt(np.trace(self.P[:3, :3])))

    @property
    def certainty(self):
        """A 0..1 confidence in this track's position: ~1 when the covariance is tiny
        (well-observed object), falling toward 0 as the positional uncertainty grows."""
        return 1.0 / (1.0 + self.position_std)

# ─────────────────────────────────────────────
#  Stage 4 — per-frame association
# ─────────────────────────────────────────────

def detections(sensors, measurements, gate_tol=GATE_TOL):
    """This frame's detected positions — the exact-cover selection from
    `initial_tracks`, which already rejects ghosts via the range-assignment
    constraint. Cost is N**(#sensors) tuples per call — fine for a handful of
    objects."""
    return initial_tracks(sensors, measurements, gate_tol)

GATE_MAHALANOBIS = 5.0   # max innovation distance for a detection to update a track

def associate(tracks, dets, gate=GATE_MAHALANOBIS):
    """Assign this frame's detections to predicted tracks. Each track must already
    have had predict() called. We score every (track, detection) pair by the
    track's Mahalanobis distance to the detection, drop pairs beyond `gate`, then
    greedily take the closest pairs so each track and detection is used at most
    once (global-nearest-neighbour).

    Returns (matches, unmatched_tracks, unmatched_dets) where `matches` is a list
    of (track_index, det_index), and the unmatched lists are indices.
    """
    pairs = []
    for ti, t in enumerate(tracks):
        for dj, d in enumerate(dets):
            m = t.mahalanobis(d)
            if m <= gate:
                pairs.append((m, ti, dj))
    pairs.sort()
    used_t, used_d, matches = set(), set(), []
    for m, ti, dj in pairs:
        if ti in used_t or dj in used_d:
            continue
        matches.append((ti, dj))
        used_t.add(ti)
        used_d.add(dj)
    unmatched_tracks = [ti for ti in range(len(tracks)) if ti not in used_t]
    unmatched_dets = [dj for dj in range(len(dets)) if dj not in used_d]
    return matches, unmatched_tracks, unmatched_dets

def track_step(tracks, sensors, measurements, gate=GATE_MAHALANOBIS):
    """One frame of association + filtering for a list of KalmanTracks (updated in
    place): predict all tracks, detect, associate, update the matched tracks.

    Returns a dict for Stage 5 to act on:
        "matched"           — set of track indices that got a measurement,
        "unmatched_tracks"  — track indices with no detection (missed -> maybe delete),
        "unmatched_dets"    — leftover detection positions (-> maybe spawn a track).
    """
    dets = detections(sensors, measurements)
    for t in tracks:
        t.predict()
    matches, unmatched_tracks, unmatched_dets = associate(tracks, dets, gate)
    for ti, dj in matches:
        tracks[ti].update(dets[dj])
    return {
        "matched": {ti for ti, _ in matches},
        "unmatched_tracks": unmatched_tracks,
        "unmatched_dets": [dets[j] for j in unmatched_dets],
    }

# ─────────────────────────────────────────────
#  Stage 5 — track management
# ─────────────────────────────────────────────

class MultiTracker:
    """Full multi-object tracker: ties Stages 1-4 together and manages the track
    set over time.

    Each frame `step(measurements)`:
      1. predict + associate + update existing tracks (Stage 4),
      2. score them — a hit resets misses and counts toward confirmation; a miss
         (no detection this frame) increments the miss counter,
      3. delete tracks that have missed too long (tentative ones quickly, confirmed
         ones only after a longer coast so brief occlusions/dropouts survive),
      4. spawn a new tentative track from every leftover (unmatched) detection.

    A tentative track is promoted to *confirmed* once it collects `confirm_hits`
    detections. Because detections are ghost-free (exact cover), a real object
    confirms within a few frames while a stray detection never accumulates hits.
    `confirmed_tracks()` is the clean output — one entry per real object.
    """
    def __init__(self, sensors, gate=GATE_MAHALANOBIS, confirm_hits=3,
                 tentative_max_misses=3, confirmed_max_misses=20, track_kwargs=None,
                 radius=0.0):
        self.sensors = sensors
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.tentative_max_misses = tentative_max_misses
        self.confirmed_max_misses = confirmed_max_misses
        self.track_kwargs = track_kwargs or {}
        self.tracks = []
        # Sensors report SURFACE ranges; objects share this radius, so adding it back to
        # every range converts them to centre ranges before detection/trilateration.
        self.radius = radius

    def _should_delete(self, t):
        limit = self.confirmed_max_misses if t.confirmed else self.tentative_max_misses
        return t.misses > limit

    def step(self, measurements):
        """Advance the tracker one frame. Returns the confirmed tracks.

        `measurements` are per-sensor SURFACE ranges (from `sensor_distances`); the
        object radius is added back so detection localises object centres."""
        if self.radius:
            measurements = [[r + self.radius for r in m] for m in measurements]
        result = track_step(self.tracks, self.sensors, measurements, self.gate)
        matched = result["matched"]

        # Score every existing track from this frame's association.
        for i, t in enumerate(self.tracks):
            t.age += 1
            if i in matched:
                t.hits += 1
                t.misses = 0
                if not t.confirmed and t.hits >= self.confirm_hits:
                    t.confirmed = True
            else:
                t.misses += 1

        # Delete tracks that have gone missing too long.
        self.tracks = [t for t in self.tracks if not self._should_delete(t)]

        # Birth: every leftover detection seeds a new tentative track.
        for pos in result["unmatched_dets"]:
            self.tracks.append(KalmanTrack(pos, **self.track_kwargs))

        return self.confirmed_tracks()

    def reset(self):
        """Drop all tracks (e.g. when the scene is reset and objects teleport)."""
        self.tracks = []

    def confirmed_tracks(self):
        return [t for t in self.tracks if t.confirmed]

    def top_tracks(self, n=10):
        """The n most promising tracks (all tracks, confirmed or tentative), ranked by
        how object-like they are: confirmed first, then more net hits (hits - misses),
        then higher certainty. Handy for diagnostics."""
        ranked = sorted(self.tracks,
                        key=lambda t: (t.confirmed, t.hits - t.misses, t.certainty),
                        reverse=True)
        return ranked[:n]

_prev_pos = None

def boxvelocity(sensors, box_obj):
    """Estimate the box's world velocity from successive trilaterated positions.
    Must be called once per frame (dt apart) so the stored sample stays fresh."""
    global _prev_pos
    cur = trilateration(sensors, box_obj)      # (x, y, z) at this instant
    if _prev_pos is None:
        _prev_pos = cur
        return (0.0, 0.0, 0.0)
    v = tuple((c - p) / dt for c, p in zip(cur, _prev_pos))
    _prev_pos = cur
    return v
