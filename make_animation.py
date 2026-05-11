"""
Local-only driver: produce an animated thumbnail by sweeping the QD energy
e_d across the Kitaev chain + quantum-dot model and watching the Majorana
zero mode disappear from the gap as the dot is tuned away from resonance.

Left panel: spectrum vs e_d, with a moving cursor at the current value.
Right panel: lowest-|E| eigenstate's coefficients vs operator index — the
classic Majorana edge profile when in the topological window, a bulk state
otherwise.

Pure numpy + matplotlib. No Julia needed.
"""

import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.path.join(REPO, "anim_frames")

T = 1.0
DELTA = 0.2
MU = 0.0
T_PRIME = 0.2
V = 1.0
EPS_D = -0.5
SCF_ITER = 40
N_MIN, N_MAX = 4, 40
N_VALUES = list(range(N_MIN, N_MAX + 1))

BG = "#fbfaf6"
INK = "#1f2933"
MUTED = "#7b8794"
SPECTRUM = "#3d5a80"
SPECTRUM_FAINT = "#b6c4d6"
CURSOR = "#e07a5f"
WF = "#e07a5f"
WF_FILL = "#f5d4c5"
ACCENT = "#f4c430"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 18,
    "font.weight": 300,
    "axes.titleweight": 400,
    "axes.titlesize": 22,
    "axes.titlecolor": INK,
    "axes.labelweight": 300,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 18,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.color": MUTED,
    "grid.alpha": 0.14,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 15,
})


def build_H(N_chain, eps_d, t=T, delta=DELTA, mu=MU, t_prime=T_PRIME):
    N = N_chain + 1
    H = np.zeros((2 * N, 2 * N))
    # Chain block: loop excludes the QD rows/cols (indices 1, 2 in Julia => 0, 1 in Python).
    for i in range(3, 2 * N + 1):
        for j in range(3, 2 * N + 1):
            i1 = (i + 1) // 2
            j1 = (j + 1) // 2
            i_odd = (i % 2 == 1)
            j_odd = (j % 2 == 1)
            if i_odd and j_odd:
                v = 0.0
                if j1 == i1: v += -mu
                if j1 == i1 + 1: v += -t
                if j1 == i1 - 1: v += -t
            elif (not i_odd) and j_odd:
                v = 0.0
                if i1 == j1 - 1: v += delta
                if i1 == j1 + 1: v += -delta
            elif i_odd and (not j_odd):
                v = 0.0
                if i1 == j1 + 1: v += delta
                if i1 == j1 - 1: v += -delta
            else:
                v = 0.0
                if i1 == j1: v += mu
                if i1 == j1 + 1: v += t
                if i1 == j1 - 1: v += t
            H[i - 1, j - 1] = v
    H[2, 0] += -t_prime
    H[3, 1] += t_prime
    H[0, 2] += -t_prime
    H[1, 3] += t_prime
    H[0, 0] = eps_d
    H[1, 1] = -eps_d
    return H


def apply_HF(H, evs, V):
    val, hop, ocd, oc1 = evs[0], evs[1], evs[2] - 0.5, evs[3] - 0.5
    H[3, 0] += -V * val
    H[0, 3] += -V * val
    H[1, 2] +=  V * val
    H[2, 1] +=  V * val
    H[0, 2] += -V * hop
    H[2, 0] += -V * hop
    H[1, 3] +=  V * hop
    H[3, 1] +=  V * hop
    H[0, 0] +=  V * oc1
    H[1, 1] += -V * oc1
    H[2, 2] +=  V * ocd
    H[3, 3] += -V * ocd


def scf(N_chain, eps_d, V, n_iter=SCF_ITER):
    H = build_H(N_chain, eps_d)
    evs = np.array([0.133222, 0.315101, 0.531797, 0.489767])
    for _ in range(n_iter):
        apply_HF(H, evs, V)
        w, v = np.linalg.eigh(H)
        new_evs = np.zeros(4)
        for i in range(len(w)):
            if w[i] < 0.0:
                factor = 0.5 if abs(w[i]) < 1e-8 else 1.0
                new_evs[0] += np.conj(v[0, i]) * v[3, i] * factor
                new_evs[1] += np.conj(v[0, i]) * v[2, i] * factor
                new_evs[2] += np.conj(v[0, i]) * v[0, i] * factor
                new_evs[3] += np.conj(v[2, i]) * v[2, i] * factor
        apply_HF(H, evs, -V)
        evs = new_evs.real
    apply_HF(H, evs, V)
    w, v = np.linalg.eigh(H)
    return w, v, evs


def lowest_state_from(w, v):
    idx = int(np.argmin(np.abs(w)))
    return w[idx], v[:, idx]


def main():
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)

    print(f"Sweeping chain length N from {N_MIN} to {N_MAX} (eps_d={EPS_D}, V={V})...")

    lowest_E = []
    site_probs = []
    for n_chain in N_VALUES:
        w, v, _ = scf(n_chain, EPS_D, V)
        e, psi = lowest_state_from(w, v)
        # site-wise probability |u|^2 + |v|^2
        probs = psi[0::2] ** 2 + psi[1::2] ** 2
        lowest_E.append(abs(e))
        site_probs.append(probs)

    lowest_E = np.array(lowest_E)
    print(f"|E_min| ranges: {lowest_E.min():.2e} to {lowest_E.max():.2e}")

    wf_xmax = max(len(p) for p in site_probs)
    wf_ymax = max(p.max() for p in site_probs) * 1.1
    e_ymax = float(lowest_E.max() * 1.5)
    e_ymin = max(1e-4, float(lowest_E.min() * 0.5))

    for k in range(len(N_VALUES)):
        n_chain = N_VALUES[k]
        n_sites = n_chain + 1
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.2), dpi=160,
                                 gridspec_kw={"width_ratios": [1.0, 1.0]})
        ax_e, ax_w = axes

        # Lowest |E| vs N traced up to current point
        Ns = np.array(N_VALUES[: k + 1])
        Es = lowest_E[: k + 1]
        ax_e.plot(Ns, Es, color=SPECTRUM, linewidth=1.4, alpha=0.85)
        ax_e.scatter(Ns, Es, color=SPECTRUM, s=18, zorder=3)
        ax_e.scatter([n_chain], [lowest_E[k]], color=CURSOR, s=80,
                     edgecolors="white", linewidths=1.2, zorder=4)
        ax_e.set_yscale("log")
        ax_e.set_xlim(N_MIN - 1, N_MAX + 1)
        ax_e.set_ylim(e_ymin, e_ymax)
        ax_e.set_xlabel("chain length  N")
        ax_e.set_ylabel("|E_min|   (log)")
        ax_e.set_title("Majorana splitting", loc="left", pad=12)
        ax_e.grid(True, which="both", axis="both")

        # Wavefunction at current N
        p = site_probs[k]
        x = np.arange(len(p))
        ax_w.fill_between(x, p, 0, color=WF_FILL, alpha=0.78, linewidth=0)
        ax_w.plot(x, p, color=WF, linewidth=2.2,
                  marker="o", markersize=5.0, mew=0)
        ax_w.set_xlim(-0.5, wf_xmax - 0.5)
        ax_w.set_ylim(0, wf_ymax)
        tick_positions = [0] + list(range(10, wf_xmax, 10))
        tick_labels = ["QD"] + [str(i) for i in range(10, wf_xmax, 10)]
        ax_w.set_xticks(tick_positions)
        ax_w.set_xticklabels(tick_labels)
        ax_w.set_xlabel("site  (QD, then chain 1 ··· N)")
        ax_w.set_ylabel("|ψ_site|²")
        ax_w.set_title("Majorana wavefunction", loc="left", pad=12)
        ax_w.grid(True, axis="both")

        fig.text(0.04, 0.955,
                 f"self-consistent HF  ·  Δ = {DELTA}, t = {T}, "
                 f"t' = {T_PRIME}, V = {V}, ε_d = {EPS_D}",
                 fontsize=15, color=MUTED)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        ax_e.text(0.97, 0.93, f"N = {n_chain}",
                  transform=ax_e.transAxes, ha="right", va="top",
                  fontsize=18, color=INK, weight=500)
        fig.savefig(os.path.join(FRAME_DIR, f"frame_{k:03d}.png"))
        plt.close(fig)

    print(f"Wrote {len(N_VALUES)} frames to {FRAME_DIR}/")


if __name__ == "__main__":
    main()
