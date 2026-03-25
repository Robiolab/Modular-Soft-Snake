import numpy as np
from sympy import symbols, sqrt, lambdify, Matrix
from sympy.physics.mechanics import ReferenceFrame, dynamicsymbols


# ---------------------------------------------------------------------------
# Symbolic parameters (shared across all modules)
# ---------------------------------------------------------------------------
N = ReferenceFrame('N')
d = symbols('d')

# Default numeric values (metres)
L1_REST = 12e-3   # rest length of prismatic link
L2_VAL = 8e-3
D_VAL = 30e-3


# ---------------------------------------------------------------------------
# Triangle geometry
# ---------------------------------------------------------------------------

def equilateral_triangle(d_val=D_VAL):
    """Vertices of an equilateral triangle with side *d_val*.
    The triangle sits on the XY plane with centroid at the origin.

    Parameters
    ----------
    d_val : float
        Side length [m].

    Returns
    -------
    ndarray (3, 2)
        Row *i* is the (x, y) coordinates of vertex *i*.
    """
    s = d_val
    h = s * np.sqrt(3) / 2
    # Centroid at the origin
    cx = s / 2
    cy = h / 3
    return np.array([
        [0.0   - cx,  0.0 - cy],
        [s     - cx,  0.0 - cy],
        [s / 2 - cx,  h   - cy],
    ])


# ---------------------------------------------------------------------------
# Open serial chain
# ---------------------------------------------------------------------------

def open_chain(name, Ox=0, Oy=0):
    """Open serial chain for one module.

    Kinematic chain (physically distinct points)
    ---------------------------------------------
        O  --[l1 along B.x]--> C  --[l2 along E.x]--> F

    Joints at O: theta1 (about N.y), theta2 (about A.x)  — 2-DOF joint
    Prismatic:   l1 = (L1_REST - h) * B.x
    Joints at C: theta3 (about B.y), theta4 (about D.x)  — 2-DOF joint
    Rigid link:  l2 = L2_VAL * E.x

    Parameters
    ----------
    name : str
        Identifier used to name frames and joint symbols (e.g. '1', 'A').
    Ox, Oy : float or sympy expr
        Base origin coordinates in the ground frame.

    Returns
    -------
    dict
        frames – {'N', 'A', 'B', 'D', 'E'} sympy ReferenceFrames
        points – {'O', 'C', 'F'} sympy Vector positions in N
        qpos   – [theta1, theta2, theta3, theta4, h]
    """
    # --- joint variables ---
    theta1 = dynamicsymbols(f'theta1_{name}')
    theta2 = dynamicsymbols(f'theta2_{name}')
    theta3 = dynamicsymbols(f'theta3_{name}')
    theta4 = dynamicsymbols(f'theta4_{name}')
    h = dynamicsymbols(f'h{name}')

    # --- frames ---
    A = N.orientnew(f'A_{name}', 'Axis', [theta1, N.y])
    B = A.orientnew(f'B_{name}', 'Axis', [theta2, A.x])
    D = B.orientnew(f'D_{name}', 'Axis', [theta3, B.y])
    E = D.orientnew(f'E_{name}', 'Axis', [theta4, D.x])

    # --- link vectors ---
    r1 = (L1_REST - h) * B.z
    r2 = L2_VAL * E.z

    # --- point position vectors in N ---
    O = Ox * N.x + Oy * N.y
    C1 = O + r1
    E1 = C1 + r2

    return {
        'frames': {'N': N, 'A': A, 'B': B, 'D': D, 'E': E},
        'points': {'base': O, 'mid': C1, 'EEF': E1},
        'qpos': [theta1, theta2, theta3, theta4, h],
    }


# ---------------------------------------------------------------------------
# Numeric helpers – build once, evaluate many times
# ---------------------------------------------------------------------------
def build_numeric_chain(chain):
    """Compile a symbolic chain dict into fast numpy-callable functions.

    Iterates over ``chain['points']`` and ``chain['frames']``, lambdifying
    every entry.

    Parameters
    ----------
    chain : dict
        As returned by :func:`open_chain`.

    Returns
    -------
    dict
        For each key in chain['points'], a matching key with:
          - an ndarray if the point is fully numeric (constant),
          - a callable(*qpos) → list[float] otherwise.
        For each non-N frame, 'R_<name>': callable(*qpos) → Matrix (3,3).
    """
    # Create plain (non-dynamic) symbols for lambdify
    n_qpos = len(chain['qpos'])
    static = symbols(f'q0:{n_qpos}')
    subs = dict(zip(chain['qpos'], static))
    args = list(static)

    result = {'points': {}, 'dcm': {}}
    N_frame = chain['frames']['N']

    # --- points (sympy Vectors → project onto N.x, N.y, N.z) ---
    for name, vec in chain['points'].items():
        exprs = [vec.dot(N_frame.x).subs(subs),
                 vec.dot(N_frame.y).subs(subs),
                 vec.dot(N_frame.z).subs(subs)]        
        result['points'][name] = lambdify(args, exprs, modules='numpy')

    result['dcm']=dict()
    # --- frames (DCM w.r.t. N) ---
    N_frame = chain['frames']['N']
    for name, frame in chain['frames'].items():
        if name == 'N':
            continue
        R_expr = frame.dcm(N_frame).subs(subs)
        result['dcm'][f'R_{name}'] = lambdify(args, R_expr, modules='numpy')

    return result

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Loop closure for the parallel robot
# ---------------------------------------------------------------------------

def closure_residuals(x, H_vec, numeric_chains, top_verts_local):
    """Compute 15 residuals for the closed-loop parallel robot.

    Parameters
    ----------
    x : array-like, shape (15,)
        Unknowns: [theta1_1..theta4_1, theta1_2..theta4_2,
                    theta1_3..theta4_3, Gx, Gy, Gz].
    H_vec : array-like, shape (3,)
        Prismatic actuator displacements [H1, H2, H3].
    numeric_chains : list of 3 dicts
        From :func:`build_numeric_chain` (one per leg).
    top_verts_local : ndarray, shape (3, 3)
        Top platform vertices in the platform (E) frame, centroid at origin.

    Returns
    -------
    list of 15 floats
        9 position + 6 orientation residuals.
    """
    from scipy.spatial.transform import Rotation as Rot

    G = np.array(x[12:15])

    # Joint vectors: [4 angles, H_i] for each chain
    qpos = [list(x[4*i : 4*(i+1)]) + [H_vec[i]] for i in range(3)]

    # Evaluate forward kinematics
    EEFs = [np.asarray(nc['points']['EEF'](*q), dtype=float)
            for nc, q in zip(numeric_chains, qpos)]
    DCMs = [np.asarray(nc['dcm']['R_E'](*q), dtype=float)
            for nc, q in zip(numeric_chains, qpos)]

    # --- Position residuals (9) ---
    # Platform orientation from chain 1:  E.dcm(N) maps N→E, so .T maps E→N
    R_platform = DCMs[0].T
    r_pos = []
    for i in range(3):
        expected = G + R_platform @ top_verts_local[i]
        r_pos.extend(EEFs[i] - expected)

    # --- Orientation residuals (6) ---
    quats = [Rot.from_matrix(R).as_quat() for R in DCMs]  # [x,y,z,w]
    r_ori = []
    for i in (1, 2):
        qi = quats[i]
        if np.dot(quats[0], qi) < 0:       # quaternion sign consistency
            qi = -qi
        r_ori.extend(quats[0][:3] - qi[:3])

    return r_pos + r_ori


def solve_closure(H_vec, numeric_chains, top_verts_local=None,
                  initial_guess=None, verbose=0):
    """Solve closed-loop forward kinematics for the parallel robot.

    Given actuator heights *H_vec*, find the 12 passive joint angles
    and the platform centroid G that satisfy loop-closure.

    Parameters
    ----------
    H_vec : array-like, shape (3,)
        Prismatic actuator displacements [m].
    numeric_chains : list of 3 dicts
        From :func:`build_numeric_chain`.
    top_verts_local : ndarray (3, 3), optional
        Top platform vertices in the E-frame (centroid at origin).
        Default: same equilateral triangle as the base, in the XY-plane.
    initial_guess : array-like (15,), optional
        Starting point [12 angles, Gx, Gy, Gz].
        Default: 5 deg for all angles, G from average EEF positions.
    verbose : int
        Passed to :func:`scipy.optimize.least_squares`.

    Returns
    -------
    dict
        angles : ndarray (3, 4) – joint angles per chain [rad]
        G      : ndarray (3,)   – platform centroid [m]
        raw    : ndarray (15,)  – full solution vector
        result : OptimizeResult from least_squares
    """
    from scipy.optimize import least_squares

    H_vec = np.asarray(H_vec, dtype=float)

    if top_verts_local is None:
        v2d = equilateral_triangle()
        top_verts_local = np.column_stack([v2d, np.zeros(3)])

    if initial_guess is None:
        ig_angle = np.deg2rad(5)
        G_pts = []
        for i, nc in enumerate(numeric_chains):
            q = [ig_angle] * 4 + [H_vec[i]]
            G_pts.append(np.asarray(nc['points']['EEF'](*q), dtype=float))
        G_ig = np.mean(G_pts, axis=0)
        initial_guess = [ig_angle] * 12 + list(G_ig)

    res = least_squares(
        closure_residuals,
        initial_guess,
        args=(H_vec, numeric_chains, top_verts_local),
        bounds=([-np.pi / 2] * 12 + [-0.05] * 3,
                [np.pi / 2] * 12 + [0.05] * 3),
        method='trf',
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=50000,
        verbose=verbose,
    )

    sol = res.x
    return {
        'angles': sol[:12].reshape(3, 4),
        'G':      sol[12:15],
        'raw':    sol,
        'result': res,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_chain(ax, nc, qpos):               
    """Plot a single open chain on a Matplotlib 3-D *ax*.

    Parameters
    ----------
    ax          : Axes3D
    nc          : dict as returned by :func:`build_numeric_chain`
    qpos        : array-like of 5 joint values [t1, t2, t3, t4, h]
    label       : legend prefix
    frame_scale : length of the E-frame axes arrows [m]

    Returns
    -------
    plotted points base, mid, EEF
    """
    O = np.asarray(nc['points']['base'](*qpos), dtype=float)
    C = np.asarray(nc['points']['mid'](*qpos), dtype=float)
    F = np.asarray(nc['points']['EEF'](*qpos), dtype=float)
    R = np.asarray(nc['dcm']['R_E'](*qpos), dtype=float)

    # l1: O → C
    ax.quiver(*O, *(C - O), color="#1a44cf", alpha=0.2, linewidth=3,
              arrow_length_ratio=0.1)
    
    
    # l2: C → F
    ax.quiver(*C, *(F - C), color='#fcba03', linewidth=2,
              arrow_length_ratio=0.1, alpha=0.85)
    
    SCALE=2e-2
    ax.set_xlim(-1*SCALE, 1*SCALE)
    ax.set_ylim(-1*SCALE, 1*SCALE)
    ax.set_zlim(-1*SCALE, 1*SCALE)
    

    


def plot_robot(numeric_chains, qpos_list):
    """Plot the full parallel robot.

    Parameters
    ----------
    numeric_chains : list of dicts
        Each element as returned by :func:`build_numeric_chain`.
    qpos_list      : list of array-like
        Joint values for each chain [t1, t2, t3, t4, h].    

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = ['royalblue', 'tomato', 'seagreen']
    labels = ['Brazo 1', 'Brazo 2', 'Brazo 3']

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')

    Os, Cs, Fs = [], [], []
    for nc, qpos, color, lbl in zip(numeric_chains, qpos_list, colors, labels):
        plot_chain(ax, nc, qpos)
        Os.append(np.asarray(nc['points']['base'](*qpos), dtype=float))
        Cs.append(np.asarray(nc['points']['mid'](*qpos), dtype=float))
        Fs.append(np.asarray(nc['points']['EEF'](*qpos), dtype=float))

    # EEF triangle
    puntas = np.array(Fs)
    ax.add_collection3d(Poly3DCollection(
        [puntas], alpha=0.3, facecolor='gold', edgecolor='darkorange',
        linewidth=2))

    # centroid
    G = puntas.mean(axis=0)
    ax.scatter(*G, color='black', s=180, marker='*', zorder=7,
               label='G centroide')
    ax.text(*(G + 0.001),
            f"G\n({G[0]*1e3:.2f},{G[1]*1e3:.2f},{G[2]*1e3:.2f})mm",
            fontsize=8, color='black')

    # G → each vertex
    for F_pt in Fs:
        ax.plot(*zip(G, F_pt), color='gray', lw=0.8, ls='--', alpha=0.5)

    # base triangle
    ax.add_collection3d(Poly3DCollection(
        [np.array(Os)], alpha=0.08, facecolor='lightgray', edgecolor='gray'))

    # uniform scale
    todos = np.vstack(Os + Cs + Fs + [G])
    centro = (todos.max(0) + todos.min(0)) / 2
    r = np.max(todos.max(0) - todos.min(0)) / 2 + 0.005
    ax.set_xlim(centro[0] - r, centro[0] + r)
    ax.set_ylim(centro[1] - r, centro[1] + r)
    ax.set_zlim(centro[2] - r, centro[2] + r)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    H_vals = [qpos[4] for qpos in qpos_list]
    ax.set_title('  '.join(f'H{i+1}={h*1e3:.1f} mm' for i, h in enumerate(H_vals)))

    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.show()
    return fig, ax
