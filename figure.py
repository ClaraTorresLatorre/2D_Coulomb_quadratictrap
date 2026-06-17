"""
Constrained log-gas equilibrium on a sliding half-space (planar case N=2, s=0).

Solves, for each constraint level a, the minimization of
    E[mu] = -int int log|z-w| dmu(z) dmu(w) + int |z|^2 dmu(z)
over probability measures supported in the half-plane {x_2 >= a}, approximated
by n point particles. The constraint x_2 >= a is a simple box bound handled by
L-BFGS-B. Particles are evolved by projected quasi-Newton descent on the
(softened) empirical energy. Each constraint level restarts from a fresh unit
disk rather than warm-starting from the previous level: this keeps the bulk
present in the initial condition and avoids inheriting an over-concentrated
state that spuriously collapses onto the wall just below threshold. A
perturb-and-resolve pass (basin-hopping diagnostic) guards against finite-n
metastability; see run_sweep.

Outputs:
    equilibrium_N1000.npz          particle configurations + run metadata
    equilibrium_geqa_N1000.png     2x3 panel figure with marginal density strips
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
import time
import os

# ---------------------------------------------------------------- parameters
n = 1000                     # number of particles
eps = 1e-8                   # softening of the log singularity
seed = 0
a_values = [-1.0, -0.3, 0.5, 0.9, 1.2, 1.5]   # constraint levels {x_2 >= a}
WALL_C = 0.5                 # on-wall tolerance constant: tol = WALL_C / sqrt(n)
OUTDIR = "/mnt/user-data/outputs"
NPZ = os.path.join(OUTDIR, "equilibrium_N1000.npz")
PNG = os.path.join(OUTDIR, "equilibrium_geqa_N1000.png")


def on_wall(X, a, n):
    """Boolean mask for particles counted as sitting on the wall {x_2 = a}.

    The tolerance is scaled to the microscopic spacing n^{-1/2} so the
    'on-wall' criterion has a consistent meaning across particle counts.
    Diagnostic/coloring cutoff only; does not enter the energy or optimization.
    """
    return X[:, 1] < a + WALL_C / np.sqrt(n)


# ---------------------------------------------------------------- solver
def init_disk(n, seed):
    """Uniform sample of exactly n points in the unit disk (polar, r = sqrt(u))."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n))          # sqrt for uniform area density
    t = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def energy_and_grad(x_flat, eps, n):
    """Softened empirical energy and gradient for potential V(z)=|z|^2.

    Discretizes  E[mu] = -int int log|z-w| dmu dmu + int |z|^2 dmu  with the
    empirical measure mu = (1/n) sum_i delta_{z_i}. The double integral over
    mu (x) mu counts each unordered pair twice, so the interaction term is

        -(1/n^2) sum_{i != j} log|z_i - z_j|  =  -(2/n^2) sum_{i<j} log|z_i - z_j|,

    which is what is computed below (logS = log|.|^2 = 2 log|.|, the 0.5
    prefactor cancelling the square). Softened by eps in the log and reciprocal.
    Dense O(n^2) in time and memory; fine for n ~ 1e3, not for n >> 1e4.
    """
    X = x_flat.reshape(n, 2)
    D = X[:, None, :] - X[None, :, :]          # (n, n, 2)
    S = np.sum(D**2, axis=-1)                   # (n, n)
    off = ~np.eye(n, dtype=bool)                # mask out self term
    E = -(0.5 / (n * n)) * np.log(S[off] + eps).sum() + np.sum(X**2) / n
    invS = np.where(off, 1.0 / (S + eps), 0.0)  # 0 on the diagonal
    G = -(2.0 / (n * n)) * np.einsum("ij,ijk->ik", invS, D) + (2.0 / n) * X
    return E, G.flatten()


def solve(a, X_init, n, eps, maxiter=600, ftol=1e-11, gtol=1e-7):
    """Minimize with the box constraint x_2 >= a (x_1 free)."""
    lb = np.full((n, 2), -np.inf); lb[:, 1] = a   # lower bound a on x_2 only
    ub = np.full((n, 2), np.inf)
    res = minimize(
        energy_and_grad, X_init.flatten(), args=(eps, n),
        method="L-BFGS-B", jac=True,
        bounds=list(zip(lb.flatten(), ub.flatten())),
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )
    return res.x.reshape(n, 2), res


def run_sweep():
    stored, meta = {}, {}
    rng = np.random.default_rng(seed + 999)
    for a in a_values:
        # Restart each level from a fresh unit disk so the bulk is present in
        # the initial condition (avoids inheriting an over-concentrated state
        # that spuriously collapses onto the wall just below threshold).
        X0 = init_disk(n, seed)
        X0[:, 1] = np.maximum(X0[:, 1], a)        # project into {x_2 >= a}
        t0 = time.time()
        X, res = solve(a, X0, n, eps)

        # Perturb-and-resolve: kick the on-wall mass into the interior and
        # re-minimize, keeping the result only if it lowers the energy. A true
        # collapse (a >= a_c) falls straight back to the wall, so this cannot
        # manufacture a bulk that should not exist.
        on = on_wall(X, a, n)
        if on.sum() > 0:
            spacing = n ** (-0.5)
            Xp = X.copy()
            Xp[on, 1] += np.abs(rng.normal(0, 3 * spacing, size=on.sum()))  # push up
            Xp[on, 0] += rng.normal(0, spacing, size=on.sum())
            Xp[:, 1] = np.maximum(Xp[:, 1], a)
            Xp, resp = solve(a, Xp, n, eps)
            if resp.fun < res.fun:
                X, res = Xp, resp

        dt = time.time() - t0
        on_line = int(on_wall(X, a, n).sum())
        print(f"a = {a:5.2f}: E = {res.fun:.6f}, iters = {res.nit:3d}, "
              f"on-line = {on_line}/{n}, t = {dt:.1f}s", flush=True)
        stored[f"a_{a}"] = X.copy()
        meta[f"a_{a}"] = np.array([a, res.fun, res.nit, dt])

        save = dict(stored)
        for k, v in meta.items():
            save["meta_" + k] = v
        save["a_values"] = np.array(a_values)
        save["n"] = np.array([n])
        save["eps"] = np.array([eps])
        np.savez_compressed(NPZ, **save)
    return stored


# ---------------------------------------------------------------- figure
def make_figure(stored):
    d = stored
    theta = np.linspace(0, 2 * np.pi, 200)
    ux, uy = np.cos(theta), np.sin(theta)

    XLIM = (-1.6, 1.6)     # horizontal = free coordinate x_1
    YLIM = (-1.5, 1.7)     # vertical   = constrained coordinate x_2

    # marginal density strips: histogram of x_1 over the on-wall mass,
    # as a fraction of all n particles -> shared scale across panels
    BINS = 40
    hist = {}
    ymax = 0.0
    for a in a_values:
        X = d[f"a_{a}"]
        counts, edges = np.histogram(X[on_wall(X, a, n), 0], bins=BINS, range=XLIM)
        frac = counts / n
        hist[a] = (frac, edges)
        ymax = max(ymax, frac.max() if frac.size else 0.0)
    strip_ymax = ymax * 1.30   # headroom so bars do not hug the top

    navy, orange = "#1a2a4a", "#e8853a"

    fig = plt.figure(figsize=(13, 10))
    outer = GridSpec(2, 3, figure=fig, hspace=0.20, wspace=0.10)
    scatter_axes, strip_axes = [], []

    for idx, a in enumerate(a_values):
        r, c = idx // 3, idx % 3
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[4, 1.9], hspace=0.10)
        ax = fig.add_subplot(inner[0])
        axd = fig.add_subplot(inner[1])

        X = d[f"a_{a}"]
        px, py = X[:, 0], X[:, 1]
        on = on_wall(X, a, n)
        ax.scatter(px[~on], py[~on], s=6.75, alpha=0.55, color=navy)
        ax.scatter(px[on], py[on], s=9.0, alpha=0.7, color=orange)
        ax.plot(ux, uy, "k--", alpha=0.5, lw=0.9)
        ax.axhline(a, color="0.35", alpha=0.9, lw=1.0)
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.set_aspect("equal", anchor="S")
        ax.set_title(f"$a = {a}$", fontsize=13)
        ax.grid(alpha=0.3)

        frac, edges = hist[a]
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        axd.bar(centers, frac, width=width, color=orange, alpha=0.9, align="center")
        axd.set_xlim(*XLIM); axd.set_ylim(0, strip_ymax)
        axd.set_yticks([]); axd.grid(alpha=0.2, axis="x")
        axd.set_anchor("N")

        if c != 0:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelbottom=False)
        if r != 1:
            axd.tick_params(labelbottom=False)

        scatter_axes.append(ax); strip_axes.append(axd)

    # match each strip's horizontal extent to its scatter box (equal-aspect
    # shrinks the scatter box; copy its left edge + width onto the strip)
    fig.canvas.draw()
    for ax, axd in zip(scatter_axes, strip_axes):
        bs = ax.get_position()
        bd = axd.get_position()
        axd.set_position([bs.x0, bd.y0, bs.width, bd.height])

    fig.savefig(PNG, dpi=140, bbox_inches="tight")
    print("saved", PNG)


def check_gradient(n_small=20, eps_small=1e-6, seed_small=1):
    """Finite-difference check of energy_and_grad on a small system."""
    from scipy.optimize import check_grad
    rng = np.random.default_rng(seed_small)
    X = rng.uniform(-0.8, 0.8, size=(n_small, 2)).flatten()
    f = lambda x: energy_and_grad(x, eps_small, n_small)[0]
    g = lambda x: energy_and_grad(x, eps_small, n_small)[1]
    err = check_grad(f, g, X)
    print(f"check_grad (n={n_small}): {err:.3e}")
    return err


if __name__ == "__main__":
    import sys
    if "--check-grad" in sys.argv:
        check_gradient()
    else:
        stored = run_sweep()
        make_figure(stored)
