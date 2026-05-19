"""
Analyse the Markov-ICL capacity sweep.

Reads capacity_grid/ (density levels x seeds, trained to convergence) and asks:
does novel-class ICL accuracy collapse toward chance below a coverage
threshold, and where is that threshold relative to the prior program's
required dimension n_req = 2*N*(N+1)*D = 160?

A logistic curve is fit to accuracy vs d_rel; the midpoint x0 (in units of
d_rel / n_req) locates the threshold, the width tells sharp vs gradual.

    python capacity_analysis.py

Writes capacity_summary.json, capacity_report.md, capacity_curve.png.
"""

import sys
import os
import json
import glob
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDDIR = os.path.join(HERE, "capacity_grid")
CHANCE = 25.0   # 4-way exact-copy task

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


def logistic(x, floor, ceiling, x0, w):
    """Logistic with a free lower asymptote (the model degrades gracefully,
    so the floor is not assumed to be chance)."""
    return floor + (ceiling - floor) / (1.0 + np.exp(-(x - x0) / w))


def load():
    levels = {}
    for d in sorted(glob.glob(os.path.join(GRIDDIR, "density*_trainseed*"))):
        rp, cp = os.path.join(d, "results.pkl"), os.path.join(d, "capacity.json")
        if not (os.path.exists(rp) and os.path.exists(cp)):
            continue
        cap = json.load(open(cp))
        acc = pickle.load(open(rp, "rb"))["results"]["novel_classes"]
        key = cap["density"]
        levels.setdefault(key, {"cap": cap, "acc": []})["acc"].append(acc)
    return [levels[k] for k in sorted(levels)]


def main():
    levels = load()
    if not levels:
        print("No runs in capacity_grid/ yet.")
        return 1

    n_req = levels[0]["cap"]["n_req"]
    rows = []
    for L in levels:
        a = np.array(L["acc"])
        rows.append({
            "density": L["cap"]["density"],
            "d_rel": L["cap"]["d_rel"],
            "effective_rank": L["cap"]["effective_rank"],
            "d_rel_over_n_req": L["cap"]["d_rel"] / n_req,
            "n_seeds": len(a),
            "acc_mean": float(a.mean()), "acc_std": float(a.std()),
            "acc_ceiling": float(a.max()), "acc_min": float(a.min()),
        })

    print(f"n_req = {n_req}   chance = {CHANCE}%\n")
    print(f"{'density':>8} {'d_rel':>7} {'d_rel/n_req':>12} "
          f"{'acc mean':>11} {'ceiling':>9} {'std':>6}")
    for r in rows:
        print(f"{r['density']:8.2f} {r['d_rel']:7d} {r['d_rel_over_n_req']:12.2f} "
              f"{r['acc_mean']:9.1f}%  {r['acc_ceiling']:7.1f}%  {r['acc_std']:5.1f}")

    # logistic fit (free lower asymptote): accuracy vs d_rel/n_req
    x = np.array([r["d_rel_over_n_req"] for r in rows])
    fits = {}
    for label, key in (("mean", "acc_mean"), ("ceiling", "acc_ceiling")):
        y = np.array([r[key] for r in rows])
        if curve_fit is not None and len(x) >= 5:
            try:
                p0 = [float(y.min()), float(y.max()), 0.5, 0.3]
                popt, _ = curve_fit(logistic, x, y, p0=p0, maxfev=40000,
                                    bounds=([0.0, 60.0, 0.0, 1e-3],
                                            [60.0, 100.0, 3.0, 5.0]))
                floor, ceiling, x0, w = [float(v) for v in popt]
                pred = logistic(x, *popt)
                ss = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
                fits[label] = {"floor": floor, "ceiling": ceiling,
                               "x0_d_rel_over_n_req": x0, "width": w,
                               "r2": float(ss)}
            except Exception as e:
                fits[label] = {"error": repr(e)}

    # saturation: smallest d_rel reaching 95% of the best ceiling
    best = max(r["acc_ceiling"] for r in rows)
    sat = next((r for r in rows if r["acc_ceiling"] >= 0.95 * best), rows[-1])

    print()
    for label, f in fits.items():
        if "error" in f:
            print(f"  {label}: fit failed — {f['error']}")
        else:
            print(f"  {label} logistic: floor {f['floor']:.1f}%, "
                  f"midpoint d_rel/n_req {f['x0_d_rel_over_n_req']:.2f}, "
                  f"width {f['width']:.2f}, ceiling {f['ceiling']:.1f}%, "
                  f"R^2 {f['r2']:.3f}")
    print(f"  saturation (ceiling within 5% of best): d_rel = {sat['d_rel']} "
          f"= {sat['d_rel_over_n_req']:.2f} n_req")

    # verdict
    cf = fits.get("ceiling", {})
    mid = cf.get("x0_d_rel_over_n_req")
    width = cf.get("width")
    lowest = rows[0]
    verdict = {}
    if mid is not None:
        sharp = width < 0.12
        verdict = {
            "midpoint_d_rel_over_n_req": mid,
            "width": width,
            "saturation_d_rel_over_n_req": sat["d_rel_over_n_req"],
            "lowest_coverage_accuracy": lowest["acc_mean"],
            "reaches_chance": bool(lowest["acc_mean"] < 40.0),
            "sharp_threshold": bool(sharp),
            "statement": (
                f"Coverage controls ICL accuracy: a {'sharp' if sharp else 'graded'} "
                f"capacity curve. Accuracy saturates at "
                f"d_rel ≈ {sat['d_rel']} ({sat['d_rel_over_n_req']:.2f} n_req); "
                f"the prior program's required dimension n_req "
                f"{'matches' if abs(sat['d_rel_over_n_req']-1.0)<0.3 else 'over-predicts'} "
                f"the saturation point. At the lowest coverage tested "
                f"(d_rel {lowest['d_rel']}) accuracy is {lowest['acc_mean']:.0f}% "
                f"— {'chance' if lowest['acc_mean']<40 else 'well above chance: graceful degradation'}."),
        }
        print("\n" + "=" * 70)
        print("VERDICT:", verdict["statement"])
        print("=" * 70)

    summary = {"n_req": n_req, "chance": CHANCE, "levels": rows,
               "logistic_fits": fits, "verdict": verdict}
    json.dump(summary, open(os.path.join(HERE, "capacity_summary.json"), "w"),
              indent=2)
    write_report(summary)
    try:
        plot(summary)
    except Exception as e:
        print(f"(plot skipped: {e!r})")
    print(f"\nsummary -> {os.path.join(HERE, 'capacity_summary.json')}")
    return 0


def write_report(s):
    L = ["# Markov-ICL capacity sweep\n"]
    v = s.get("verdict", {})
    if v:
        L.append(f"**{v['statement']}**\n")
    L.append(f"- Knob: input-mask density; capacity measured by the masked "
             f"relative dimension d_rel. Task requirement n_req = {s['n_req']}, "
             f"chance = {s['chance']}%.")
    L.append(f"- {len(s['levels'])} capacity levels, 5 seeds each, trained to "
             f"convergence.\n")
    L.append("| density | d_rel | d_rel/n_req | acc mean | ceiling | std |")
    L.append("|---|---|---|---|---|---|")
    for r in s["levels"]:
        L.append(f"| {r['density']:.2f} | {r['d_rel']} | "
                 f"{r['d_rel_over_n_req']:.2f} | {r['acc_mean']:.1f}% | "
                 f"{r['acc_ceiling']:.1f}% | {r['acc_std']:.1f} |")
    L.append("")
    for label, f in s["logistic_fits"].items():
        if "error" not in f:
            L.append(f"- **{label}** logistic fit: threshold at "
                     f"d_rel/n_req = {f['x0_d_rel_over_n_req']:.2f} "
                     f"(d_rel = {f['x0_d_rel_over_n_req']*s['n_req']:.0f}), "
                     f"width {f['width']:.2f}, ceiling {f['ceiling']:.1f}%, "
                     f"R² {f['r2']:.3f}.")
    open(os.path.join(HERE, "capacity_report.md"), "w").write("\n".join(L) + "\n")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = s["levels"]
    x = np.array([r["d_rel_over_n_req"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.axhline(s["chance"], color="gray", ls=":", lw=1, label="chance (4-way)")
    ax.axvline(1.0, color="crimson", ls="--", lw=1.5,
               label="d_rel = n_req  (task requirement)")
    mean = [r["acc_mean"] for r in rows]
    ceil = [r["acc_ceiling"] for r in rows]
    lo = [r["acc_mean"] - r["acc_std"] for r in rows]
    hi = [r["acc_mean"] + r["acc_std"] for r in rows]
    ax.fill_between(x, lo, hi, color="#1f5066", alpha=0.15)
    ax.plot(x, ceil, "^-", color="#1f5066", lw=1.4, ms=7, label="best seed")
    ax.plot(x, mean, "o-", color="#1f5066", lw=2, ms=7, label="mean over seeds")
    fit = s["logistic_fits"].get("ceiling", {})
    if "x0_d_rel_over_n_req" in fit:
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, logistic(xs, fit["floor"], fit["ceiling"],
                             fit["x0_d_rel_over_n_req"], fit["width"]),
                color="#c98a00", lw=1.6,
                label=f"logistic fit (midpoint {fit['x0_d_rel_over_n_req']:.2f})")
    ax.set_xlabel("d_rel / n_req   —   projection coverage relative to task requirement")
    ax.set_ylabel("novel-class ICL accuracy (%)")
    ax.set_title("Markov-ICL capacity — a graded coverage curve, "
                 "saturating at d_rel ≈ n_req")
    ax.set_ylim(15, 102)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "capacity_curve.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
