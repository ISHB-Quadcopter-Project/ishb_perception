import numpy as np

# ---- Mission / rectangle parameters (edit these; they are read at call time) ----
RECT = dict(x_min=-26.0, x_max=50.0, y_min=-13.0, y_max=4.0)
EDGE_BUFFER   = 10.0     # keep-out margin from every rectangle edge (m)
ROW_SPACING   = 7.0    # <-- tune: target distance between sweep rows (m)
POINT_SPACING = 6.0    # <-- tune: target distance between consecutive waypoints along ANY leg (m)
ORIGIN        = (0.0, 0.0)   # drone start / home position


def leg_interior(p0, p1, spacing):
    """Interior points along p0->p1 spaced ~`spacing` apart (endpoints excluded).

    Sampling by SPACING (not a fixed count) keeps density uniform: a 48 m row
    and a 3.25 m transition get proportional numbers of points.
    """
    L = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    n = max(1, int(round(L / spacing)))          # number of equal intervals
    return [(p0[0] + (p1[0] - p0[0]) * (i / n),
             p0[1] + (p1[1] - p0[1]) * (i / n)) for i in range(1, n)]


def build_lawnmower(rect=None, edge_buffer=None, row_spacing=None,
                    point_spacing=None, origin=None):
    """
    Boustrophedon (lawnmower / square-wave) coverage path.

    All parameters default to None and are resolved from the module-level
    constants AT CALL TIME. This is deliberate: it means editing ROW_SPACING /
    POINT_SPACING at the top of the file (or setting m.ROW_SPACING at runtime)
    actually takes effect, instead of being frozen at function-definition time.
    Pass an explicit value to override the constant for a single call.

    Returns (waypoints, row_ys):
      waypoints : list of (x, y) tuples in flight order (endpoints + interior)
      row_ys    : the row y-values actually used
    """
    if rect is None:          rect = RECT
    if edge_buffer is None:   edge_buffer = EDGE_BUFFER
    if row_spacing is None:   row_spacing = ROW_SPACING
    if point_spacing is None: point_spacing = POINT_SPACING
    if origin is None:        origin = ORIGIN

    sx_min, sx_max = rect["x_min"] + edge_buffer, rect["x_max"] - edge_buffer
    sy_min, sy_max = rect["y_min"] + edge_buffer, rect["y_max"] - edge_buffer
    x_start, x_end = origin[0], sx_max

    # nearest y-extreme to the origin becomes the first row
    if abs(sy_max - origin[1]) <= abs(origin[1] - sy_min):
        first_y, last_y = sy_max, sy_min
    else:
        first_y, last_y = sy_min, sy_max

    # linspace guarantees the whole safe band is covered edge-to-edge with
    # spacing <= row_spacing (avoids a leftover gap if height doesn't divide evenly)
    n_rows = max(2, int(np.ceil(abs(first_y - last_y) / row_spacing)) + 1)
    row_ys = np.linspace(first_y, last_y, n_rows)

    legs = []
    if (x_start, row_ys[0]) != tuple(origin):
        legs.append((tuple(origin), (x_start, row_ys[0])))            # entry leg
    for i, y in enumerate(row_ys):
        a, b = ((x_start, y), (x_end, y)) if i % 2 == 0 else ((x_end, y), (x_start, y))
        legs.append((a, b))                                           # horizontal row
        if i < len(row_ys) - 1:
            legs.append((b, (b[0], row_ys[i + 1])))                   # vertical transition

    waypoints = [legs[0][0]]
    for p0, p1 in legs:
        waypoints += leg_interior(p0, p1, point_spacing)
        waypoints.append(p1)
    return waypoints, row_ys


def build_persisted_array(rect=None, edge_buffer=None, row_spacing=None,
                          point_spacing=None, origin=None):
    """Returns ((N,5) float32 array as [x, y, 0.0, -(index+1), 0.0], row_ys).

    Parameters are passed straight through to build_lawnmower (None -> current
    module constant). No parameter is captured at definition time.
    """
    waypoints, row_ys = build_lawnmower(rect, edge_buffer, row_spacing,
                                        point_spacing, origin)
    arr = np.array(
        [[round(x, 3), round(y, 3), 0.0, -(i + 1), 0.0] for i, (x, y) in enumerate(waypoints)],
        dtype=np.float32,
    )
    return arr, row_ys


# ---- Usage inside your class ----
#   self.all_persisted_array, _ = build_persisted_array()
#   # override per-call without touching the constants:
#   self.all_persisted_array, _ = build_persisted_array(row_spacing=6.0, point_spacing=4.0)


# if __name__ == "__main__":
#     np.set_printoptions(suppress=True)
#     arr, row_ys = build_persisted_array()
#     print(f"rows: {len(row_ys)} at {abs(row_ys[1]-row_ys[0]):.3f} m | "
#           f"point_spacing: {POINT_SPACING} m | total waypoints: {len(arr)}")
#     print(arr)