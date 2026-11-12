"""Screen-then-confirm discovery of tenure-localised churn effects.

Customers are split in half. On the discovery half, a forward search over
attribute-by-tenure-window interactions in a discrete-time hazard model picks
the terms that survive a Bonferroni-adjusted likelihood-ratio test. The
confirmation half is used only to estimate and test the selected terms and to
score predictions. The whole pipeline is repeated over five splits.
"""

import json
import os
import shutil
import sys
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(ROOT, "data", "telco_customers.csv")
OUT_DIR = os.path.join(ROOT, "results")
SEEDS = (42, 7, 19, 101, 2024)
PRIMARY_SEED = 42
MAX_MONTH = 24
BRIER_MONTHS = (1, 3, 6, 12, 24)
ALPHA = 0.05
MAX_TERMS = 8
WINDOWS = {
    "m1": (1, 1),
    "m2_6": (2, 6),
    "m7_24": (7, 24),
}
WINDOW_LABEL = {"m1": "month 1", "m2_6": "months 2-6", "m7_24": "months 7-24"}
CALIB_WINDOWS = {"m1": (1, 1), "m2_6": (2, 6), "m7_24": (7, 24)}
LABELS = {
    "senior": "Senior citizen",
    "male": "Male",
    "partner": "Has partner",
    "dependents": "Has dependents",
    "phone": "Phone service",
    "multi_lines": "Multiple lines",
    "fiber": "Fiber optic",
    "no_internet": "No internet",
    "no_security": "No online security",
    "no_backup": "No online backup",
    "no_protection": "No device protection",
    "no_support": "No tech support",
    "stream_tv": "Streaming TV",
    "stream_movies": "Streaming movies",
    "paperless": "Paperless billing",
    "electronic_check": "Electronic check",
    "mailed_check": "Mailed check",
    "credit_card": "Credit card (auto)",
    "charges_z": "Monthly charge (SD)",
    "one_year": "One-year contract",
    "two_year": "Two-year contract",
}
LEVEL_ONLY = ("one_year", "two_year")
MODEL_NAMES = {
    "screened": "Screened hazard (ours)",
    "level": "Level hazard",
    "cox": "Cox PH",
    "gbm": "Boosted hazard",
}
MODEL_COLORS = {
    "screened": "#1d4ed8",
    "level": "#94a3b8",
    "cox": "#b45309",
    "gbm": "#0f766e",
    "observed": "#0f172a",
}


def load_customers(path):
    raw = pd.read_csv(path)
    n_raw = len(raw)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    dropped = int(raw["TotalCharges"].isna().sum())
    raw = raw.dropna(subset=["TotalCharges"]).copy()
    raw["time"] = raw["tenure"].astype(int).clip(lower=1)
    raw["event"] = raw["Churn"].eq("Yes").astype(int)
    raw = raw.drop(columns=["customerID", "tenure", "TotalCharges", "Churn"]).reset_index(drop=True)
    meta = {
        "n_raw": n_raw,
        "dropped_blank_charges": dropped,
        "n": int(len(raw)),
        "events": int(raw["event"].sum()),
        "censored": int((raw["event"] == 0).sum()),
    }
    return raw, meta


def standing(frame, charge_mean, charge_std):
    def yes(col):
        return frame[col].eq("Yes").to_numpy(dtype=float)

    def no(col):
        return frame[col].eq("No").to_numpy(dtype=float)

    pay = frame["PaymentMethod"].to_numpy()
    net = frame["InternetService"].to_numpy()
    con = frame["Contract"].to_numpy()
    return pd.DataFrame(
        {
            "senior": frame["SeniorCitizen"].to_numpy(dtype=float),
            "male": frame["gender"].eq("Male").to_numpy(dtype=float),
            "partner": yes("Partner"),
            "dependents": yes("Dependents"),
            "phone": yes("PhoneService"),
            "multi_lines": yes("MultipleLines"),
            "fiber": (net == "Fiber optic").astype(float),
            "no_internet": (net == "No").astype(float),
            "no_security": no("OnlineSecurity"),
            "no_backup": no("OnlineBackup"),
            "no_protection": no("DeviceProtection"),
            "no_support": no("TechSupport"),
            "stream_tv": yes("StreamingTV"),
            "stream_movies": yes("StreamingMovies"),
            "paperless": yes("PaperlessBilling"),
            "electronic_check": (pay == "Electronic check").astype(float),
            "mailed_check": (pay == "Mailed check").astype(float),
            "credit_card": (pay == "Credit card (automatic)").astype(float),
            "charges_z": (frame["MonthlyCharges"].to_numpy(dtype=float) - charge_mean) / charge_std,
            "one_year": (con == "One year").astype(float),
            "two_year": (con == "Two year").astype(float),
        },
        index=frame.index,
    )


def expand(customers, horizon=None):
    times = customers["time"].to_numpy()
    reps = times if horizon is None else np.full(len(customers), horizon)
    idx = np.repeat(np.arange(len(customers)), reps)
    month = np.concatenate([np.arange(1, r + 1) for r in reps])
    person = customers.iloc[idx].reset_index(drop=True)
    person["cid"] = idx
    person["month"] = month
    person["y"] = ((month == times[idx]) & (customers["event"].to_numpy()[idx] == 1)).astype(int)
    return person


def baseline_time(month):
    m = month.astype(float)
    return pd.DataFrame(
        {
            "const": np.ones(len(m)),
            "t1": (m == 1).astype(float),
            "t2": (m == 2).astype(float),
            "t3": (m == 3).astype(float),
            "log_month": np.log(m),
            "month_lin": m / 12.0,
            "month_quad": (m / 12.0) ** 2,
        }
    )


def window_indicator(month, window):
    lo, hi = WINDOWS[window]
    return ((month >= lo) & (month <= hi)).astype(float)


def term_column(stand, month, term):
    var, window = term.split("@")
    return stand[var].to_numpy() * window_indicator(month, window)


def design(person, stand, terms=(), renewal=False):
    month = person["month"].to_numpy()
    X = pd.concat([baseline_time(month), stand.reset_index(drop=True)], axis=1)
    for term in terms:
        X[term] = term_column(stand.reset_index(drop=True), month, term)
    if renewal:
        con = person["Contract"].to_numpy()
        one = (con == "One year") & (month >= 11) & (month <= 13)
        two = (con == "Two year") & (month >= 23) & (month <= 25)
        X["renewal_window"] = (one | two).astype(float)
    return X


def fit_logit(X, y, start=None, groups=None):
    model = sm.Logit(np.asarray(y, dtype=float), X)
    kwargs = {"disp": False, "method": "newton", "maxiter": 60}
    if start is not None:
        kwargs["start_params"] = start
    try:
        result = model.fit(**kwargs)
    except Exception:
        result = model.fit(disp=False, method="lbfgs", maxiter=500)
    if groups is None:
        return result
    try:
        return model.fit(
            disp=False,
            method="newton",
            maxiter=60,
            start_params=result.params.to_numpy(),
            cov_type="cluster",
            cov_kwds={"groups": np.asarray(groups)},
        )
    except Exception:
        return result


def candidate_terms():
    return [f"{var}@{w}" for var in LABELS if var not in LEVEL_ONLY for w in WINDOWS]


def screen(person, stand, y):
    stand_rows = stand.iloc[person["cid"].to_numpy()].reset_index(drop=True)
    month = person["month"].to_numpy()
    base = design(person, stand_rows)
    current = fit_logit(base, y)
    selected = []
    trail = []
    remaining = candidate_terms()
    while remaining and len(selected) < MAX_TERMS:
        best = None
        for term in remaining:
            X = base.copy()
            for chosen in selected + [term]:
                X[chosen] = term_column(stand_rows, month, chosen)
            start = np.append(current.params.to_numpy(), 0.0)
            fit = fit_logit(X, y, start=start)
            lr = max(0.0, 2.0 * (fit.llf - current.llf))
            p = float(chi2.sf(lr, 1))
            if best is None or p < best[1]:
                best = (term, p, lr, fit)
        threshold = ALPHA / len(remaining)
        trail.append(
            {
                "step": len(selected) + 1,
                "term": best[0],
                "lr": best[2],
                "pvalue": best[1],
                "threshold": threshold,
                "accepted": bool(best[1] < threshold),
            }
        )
        if best[1] >= threshold:
            break
        selected.append(best[0])
        remaining.remove(best[0])
        current = best[3]
    return selected, pd.DataFrame(trail)


def hazards_logit(result, customers, stand, terms):
    panel = expand(customers, MAX_MONTH)
    stand_rows = stand.iloc[panel["cid"].to_numpy()].reset_index(drop=True)
    X = design(panel, stand_rows, terms).reindex(columns=result.params.index, fill_value=0.0)
    h = np.clip(np.asarray(result.predict(X), dtype=float), 1e-6, 1 - 1e-6)
    return h.reshape(len(customers), MAX_MONTH)


def gbm_features(person, stand):
    stand_rows = stand.iloc[person["cid"].to_numpy()].reset_index(drop=True)
    X = stand_rows.copy()
    X["month"] = person["month"].to_numpy(dtype=float)
    return X


def fit_gbm(person, stand, seed):
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=seed,
    )
    model.fit(gbm_features(person, stand), person["y"].to_numpy())
    return model


def hazards_gbm(model, customers, stand):
    panel = expand(customers, MAX_MONTH)
    h = model.predict_proba(gbm_features(panel, stand))[:, 1]
    return np.clip(h, 1e-6, 1 - 1e-6).reshape(len(customers), MAX_MONTH)


def fit_cox(customers, stand):
    df = stand.copy()
    df["time"] = customers["time"].to_numpy(dtype=float)
    df["event"] = customers["event"].to_numpy()
    for penalizer in (0.0, 0.01, 0.1):
        try:
            return CoxPHFitter(penalizer=penalizer).fit(df, "time", "event"), df
        except Exception:
            continue
    raise RuntimeError("Cox model failed")


def hazards_cox(model, stand):
    times = np.arange(1, MAX_MONTH + 1)
    surv = model.predict_survival_function(stand, times=times).to_numpy().T
    prev = np.concatenate([np.ones((len(surv), 1)), surv[:, :-1]], axis=1)
    return np.clip(1.0 - surv / np.clip(prev, 1e-9, None), 1e-6, 1 - 1e-6)


def km_at(kmf, t):
    return float(np.asarray(kmf.predict(t)).reshape(-1)[0])


def brier(times, events, s_t, t, cens):
    died = (times <= t) & (events == 1)
    alive = times > t
    out = np.zeros(len(times))
    if died.any():
        g = np.clip(np.asarray(cens.predict(np.clip(times[died] - 1e-6, 0, None)), dtype=float), 1e-3, 1)
        out[died] = s_t[died] ** 2 / g
    if alive.any():
        out[alive] = (1 - s_t[alive]) ** 2 / max(km_at(cens, t), 1e-3)
    return float(out.mean())


def window_calibration(customers, hazard):
    times = customers["time"].to_numpy()
    events = customers["event"].to_numpy()
    group = customers["PaymentMethod"].to_numpy() + " | " + customers["InternetService"].to_numpy()
    rows = []
    for g in np.unique(group):
        mask = group == g
        for w, (lo, hi) in CALIB_WINDOWS.items():
            at_risk = 0
            observed = 0
            predicted = 0.0
            for m in range(lo, hi + 1):
                risk = mask & (times >= m)
                at_risk += int(risk.sum())
                observed += int((risk & (times == m) & (events == 1)).sum())
                predicted += float(hazard[risk, m - 1].sum())
            if at_risk:
                rows.append({"group": g, "window": w, "at_risk": at_risk, "observed": observed, "predicted": predicted})
    frame = pd.DataFrame(rows)
    frame["abs_gap"] = (frame["observed"] - frame["predicted"]).abs() / frame["at_risk"]
    return frame


def score(customers, hazard, name, seed):
    surv = np.cumprod(1 - hazard, axis=1)
    times = customers["time"].to_numpy()
    events = customers["event"].to_numpy()
    cens = KaplanMeierFitter().fit(times, 1 - events)
    row = {"model": name, "seed": seed, "c_index": float(concordance_index(times, surv[:, MAX_MONTH - 1], events))}
    for t in BRIER_MONTHS:
        row[f"brier_{t}"] = brier(times, events, surv[:, t - 1], t, cens)
    row["ibs_1_24"] = float(np.mean([brier(times, events, surv[:, t - 1], t, cens) for t in range(1, MAX_MONTH + 1)]))
    calib = window_calibration(customers, hazard)
    for w in CALIB_WINDOWS:
        part = calib[calib["window"] == w]
        row[f"wce_{w}"] = float(np.average(part["abs_gap"], weights=part["at_risk"]))
    row["wce_all"] = float(np.average(calib["abs_gap"], weights=calib["at_risk"]))
    return row, calib


def coef_table(result, terms):
    conf = result.conf_int()
    rows = []
    for term in terms:
        b = float(result.params[term])
        rows.append(
            {
                "term": term,
                "odds_ratio": float(np.exp(b)),
                "or_low": float(np.exp(conf.loc[term, 0])),
                "or_high": float(np.exp(conf.loc[term, 1])),
                "pvalue": float(result.pvalues[term]),
            }
        )
    return pd.DataFrame(rows)


def joint_wald(result, terms):
    names = list(result.params.index)
    R = np.zeros((len(terms), len(names)))
    for i, term in enumerate(terms):
        R[i, names.index(term)] = 1.0
    test = result.wald_test(R, scalar=True)
    return float(np.asarray(test.statistic).ravel()[0]), float(np.asarray(test.pvalue).ravel()[0])


def pretty(term):
    if "@" not in term:
        return LABELS.get(term, term)
    var, w = term.split("@")
    return f"{LABELS[var]} x {WINDOW_LABEL[w]}"


def run_split(customers, seed):
    disc, conf = train_test_split(customers, test_size=0.5, random_state=seed, stratify=customers["event"])
    disc = disc.reset_index(drop=True)
    conf = conf.reset_index(drop=True)
    mean = float(disc["MonthlyCharges"].mean())
    sd = float(disc["MonthlyCharges"].std(ddof=0))
    s_disc = standing(disc, mean, sd)
    s_conf = standing(conf, mean, sd)
    p_disc = expand(disc)
    p_conf = expand(conf)
    y_disc = p_disc["y"].to_numpy()
    y_conf = p_conf["y"].to_numpy()

    selected, trail = screen(p_disc, s_disc, y_disc)

    rows_disc = s_disc.iloc[p_disc["cid"].to_numpy()].reset_index(drop=True)
    level = fit_logit(design(p_disc, rows_disc), y_disc)
    screened = fit_logit(design(p_disc, rows_disc, selected), y_disc)
    cox, _ = fit_cox(disc, s_disc)
    gbm = fit_gbm(p_disc, s_disc, seed)

    hazards = {
        "screened": hazards_logit(screened, conf, s_conf, selected),
        "level": hazards_logit(level, conf, s_conf, ()),
        "cox": hazards_cox(cox, s_conf),
        "gbm": hazards_gbm(gbm, conf, s_conf),
    }
    metrics = []
    calibs = {}
    for name, h in hazards.items():
        row, calib = score(conf, h, name, seed)
        row["n_selected"] = len(selected)
        metrics.append(row)
        calibs[name] = calib

    rows_conf = s_conf.iloc[p_conf["cid"].to_numpy()].reset_index(drop=True)
    groups = p_conf["cid"].to_numpy()
    confirm_level = fit_logit(design(p_conf, rows_conf), y_conf)
    confirm = fit_logit(design(p_conf, rows_conf, selected), y_conf, groups=groups)
    confirm_plain = fit_logit(design(p_conf, rows_conf, selected), y_conf)
    lr_conf = 2.0 * (confirm_plain.llf - confirm_level.llf)
    wald_stat, wald_p = joint_wald(confirm, selected) if selected else (np.nan, np.nan)
    confirmation = {
        "seed": seed,
        "selected": selected,
        "lr_stat": float(lr_conf),
        "lr_df": len(selected),
        "lr_p": float(chi2.sf(lr_conf, len(selected))) if selected else np.nan,
        "wald_stat": wald_stat,
        "wald_p": wald_p,
        "n_disc": int(len(disc)),
        "n_conf": int(len(conf)),
        "events_disc": int(disc["event"].sum()),
        "events_conf": int(conf["event"].sum()),
        "rows_disc": int(len(p_disc)),
        "rows_conf": int(len(p_conf)),
    }
    extra = None
    if seed == PRIMARY_SEED:
        renewal = fit_logit(design(p_conf, rows_conf, selected, renewal=True), y_conf, groups=groups)
        extra = {
            "trail": trail,
            "coefs": coef_table(confirm, selected),
            "renewal": coef_table(renewal, ["renewal_window"]),
            "level_coefs": coef_table(confirm_level, list(LABELS)),
            "hazards": hazards,
            "conf": conf,
            "calibs": calibs,
        }
    return metrics, confirmation, extra


def ph_diagnostics(customers):
    mean = float(customers["MonthlyCharges"].mean())
    sd = float(customers["MonthlyCharges"].std(ddof=0))
    s = standing(customers, mean, sd)
    model, df = fit_cox(customers, s)
    test = proportional_hazard_test(model, df, time_transform="rank")
    table = test.summary.reset_index().rename(columns={"index": "covariate"})
    table["label"] = table["covariate"].map(LABELS)
    hr = np.exp(model.params_)
    table["hazard_ratio"] = table["covariate"].map(hr)
    return table[["covariate", "label", "hazard_ratio", "test_statistic", "p"]]


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "font.family": "serif",
        }
    )


def plot_pipeline(path):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.axis("off")
    top = [
        "Customer records\n(n = 7,032)",
        "Person-month\nexpansion",
        "Stratified 50/50 split:\ndiscovery and\nconfirmation",
        f"Forward screen of\n{len(candidate_terms())} attribute-by-window\nterms (Bonferroni LR)",
    ]
    bottom = [
        "Repeat for\nfive splits;\nselection frequency",
        "Score on confirmation:\nC-index, Brier,\nwindow calibration",
        "Refit selected terms\non confirmation half;\nclustered Wald test",
    ]
    width, height, gap = 0.215, 0.32, 0.03
    xs = [i * (width + gap) + 0.01 for i in range(4)]
    for x, text in zip(xs, top):
        ax.add_patch(FancyBboxPatch((x, 0.58), width, height, boxstyle="round,pad=0.008", fc="#eff6ff", ec="#1d4ed8", lw=1.1))
        ax.text(x + width / 2, 0.58 + height / 2, text, ha="center", va="center", fontsize=7.5)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + width + gap, 0.74), xytext=(x + width, 0.74), arrowprops=dict(arrowstyle="->", color="#334155"))
    colors = [("#fff7ed", "#b45309"), ("#f0fdf4", "#0f766e"), ("#f0fdf4", "#0f766e")]
    bxs = [xs[1], xs[2], xs[3]]
    for x, text, (fc, ec) in zip(bxs, bottom, colors):
        ax.add_patch(FancyBboxPatch((x, 0.06), width, height, boxstyle="round,pad=0.008", fc=fc, ec=ec, lw=1.1))
        ax.text(x + width / 2, 0.06 + height / 2, text, ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(xs[3] + width / 2, 0.06 + height), xytext=(xs[3] + width / 2, 0.58), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.annotate("", xy=(xs[2] + width, 0.22), xytext=(xs[3], 0.22), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.annotate("", xy=(xs[1] + width, 0.22), xytext=(xs[2], 0.22), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def empirical(customers, column, mapping, horizon):
    times = customers["time"].to_numpy()
    events = customers["event"].to_numpy()
    out = []
    for label, mask in mapping.items():
        for m in range(1, horizon + 1):
            risk = mask & (times >= m)
            n = int(risk.sum())
            d = int((risk & (times == m) & (events == 1)).sum())
            out.append({"group": label, "month": m, "risk": n, "events": d, "hazard": d / n if n else np.nan})
    return pd.DataFrame(out)


def plot_empirical(customers, path):
    style()
    pay = customers["PaymentMethod"]
    quart = pd.qcut(customers["MonthlyCharges"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    a = empirical(customers, "MonthlyCharges", {q: (quart == q).to_numpy() for q in quart.cat.categories}, 12)
    b = empirical(
        customers,
        "PaymentMethod",
        {
            "Manual (check)": pay.isin(["Electronic check", "Mailed check"]).to_numpy(),
            "Automatic": (~pay.isin(["Electronic check", "Mailed check"])).to_numpy(),
        },
        12,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for label, color in zip(quart.cat.categories, ("#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a")):
        part = a[a["group"] == label]
        axes[0].plot(part["month"], part["hazard"], marker="o", ms=3, color=color, label=label)
    axes[0].set_title("(a) By monthly-charge quartile", fontsize=9)
    for label, color in (("Manual (check)", "#c2410c"), ("Automatic", "#0f766e")):
        part = b[b["group"] == label]
        axes[1].plot(part["month"], part["hazard"], marker="o", ms=3, color=color, label=label)
    axes[1].set_title("(b) By payment method", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Month of tenure")
        ax.set_xticks([1, 3, 6, 9, 12])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("Monthly exit hazard")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return a, b


def plot_forest(coefs, path):
    style()
    coefs = coefs.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.5, 0.45 * len(coefs) + 0.9))
    y = np.arange(len(coefs))
    ax.errorbar(
        coefs["odds_ratio"],
        y,
        xerr=[coefs["odds_ratio"] - coefs["or_low"], coefs["or_high"] - coefs["odds_ratio"]],
        fmt="o",
        color="#1d4ed8",
        capsize=3,
    )
    ax.axvline(1, color="#94a3b8", ls="--", lw=1)
    ax.set_xscale("log")
    ticks = [0.2, 0.5, 1, 2, 5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.12, 8)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(t).replace(" x ", "\nx ") for t in coefs["term"]], fontsize=8)
    ax.set_xlabel("Odds ratio (log scale), 95% CI")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_brier(summary_by_month, path):
    style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for name in ("screened", "level", "cox", "gbm"):
        part = summary_by_month[summary_by_month["model"] == name]
        ax.plot(part["month"], part["brier"], marker="o", ms=3, color=MODEL_COLORS[name], label=MODEL_NAMES[name])
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("IPCW Brier score")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_month1(conf, hazards, path):
    style()
    times = conf["time"].to_numpy()
    events = conf["event"].to_numpy()
    pay = conf["PaymentMethod"].to_numpy()
    groups = ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
    short = ["Bank\ntransfer", "Credit\ncard", "Electronic\ncheck", "Mailed\ncheck"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    width = 0.17
    x = np.arange(len(groups))
    obs = [float(((pay == g) & (times == 1) & (events == 1)).sum() / (pay == g).sum()) for g in groups]
    ax.bar(x - 2 * width, obs, width, color=MODEL_COLORS["observed"], label="Observed")
    for i, name in enumerate(("screened", "level", "cox", "gbm")):
        pred = [float(hazards[name][pay == g, 0].mean()) for g in groups]
        ax.bar(x + (i - 1) * width, pred, width, color=MODEL_COLORS[name], label=MODEL_NAMES[name])
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("Month-1 exit probability")
    ax.set_ylim(0, 0.14)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("interpreter:", sys.prefix)
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    customers, meta = load_customers(DATA_PATH)
    print(meta)

    ph = ph_diagnostics(customers)
    ph.to_csv(os.path.join(OUT_DIR, "ph_diagnostics.csv"), index=False)
    plot_pipeline(os.path.join(OUT_DIR, "fig_pipeline.png"))
    a, b = plot_empirical(customers, os.path.join(OUT_DIR, "fig_empirical.png"))
    pd.concat([a.assign(panel="charge_quartile"), b.assign(panel="payment")]).to_csv(
        os.path.join(OUT_DIR, "empirical_hazard.csv"), index=False
    )

    metrics = []
    confirmations = []
    extra = None
    for seed in SEEDS:
        print(f"split {seed} ...", flush=True)
        m, c, e = run_split(customers, seed)
        metrics += m
        confirmations.append(c)
        if e is not None:
            extra = e
        print("  selected:", c["selected"], "confirm LR p:", c["lr_p"], flush=True)

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(OUT_DIR, "metrics_by_split.csv"), index=False)
    cols = ["c_index", "ibs_1_24"] + [f"brier_{t}" for t in BRIER_MONTHS] + ["wce_m1", "wce_m2_6", "wce_m7_24", "wce_all"]
    summary = metrics.groupby("model")[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reindex(["screened", "level", "cox", "gbm"]).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

    paired = []
    for name in ("level", "cox", "gbm"):
        for col in ("ibs_1_24", "brier_1", "wce_m1", "wce_all", "c_index"):
            diff = (
                metrics[metrics["model"] == "screened"].set_index("seed")[col]
                - metrics[metrics["model"] == name].set_index("seed")[col]
            )
            paired.append(
                {
                    "versus": name,
                    "metric": col,
                    "mean_diff": float(diff.mean()),
                    "sd_diff": float(diff.std(ddof=1)),
                    "splits_screened_better": int((diff > 0).sum() if col == "c_index" else (diff < 0).sum()),
                }
            )
    pd.DataFrame(paired).to_csv(os.path.join(OUT_DIR, "paired_differences.csv"), index=False)

    by_month = []
    for name in ("screened", "level", "cox", "gbm"):
        part = metrics[metrics["model"] == name]
        for t in BRIER_MONTHS:
            by_month.append({"model": name, "month": t, "brier": float(part[f"brier_{t}"].mean())})
    by_month = pd.DataFrame(by_month)
    plot_brier(by_month, os.path.join(OUT_DIR, "fig_brier.png"))

    freq = {}
    for c in confirmations:
        for term in c["selected"]:
            freq[term] = freq.get(term, 0) + 1
    freq_table = (
        pd.DataFrame([{"term": t, "label": pretty(t), "splits_selected": n} for t, n in freq.items()])
        .sort_values("splits_selected", ascending=False)
        .reset_index(drop=True)
    )
    freq_table.to_csv(os.path.join(OUT_DIR, "selection_frequency.csv"), index=False)
    conf_table = pd.DataFrame(
        [{**{k: v for k, v in c.items() if k != "selected"}, "selected": "; ".join(c["selected"])} for c in confirmations]
    )
    conf_table.to_csv(os.path.join(OUT_DIR, "confirmation_by_split.csv"), index=False)

    extra["trail"].to_csv(os.path.join(OUT_DIR, "screen_trail_primary.csv"), index=False)
    coefs = extra["coefs"].copy()
    coefs["label"] = coefs["term"].map(pretty)
    coefs.to_csv(os.path.join(OUT_DIR, "confirmation_coefficients_primary.csv"), index=False)
    extra["renewal"].to_csv(os.path.join(OUT_DIR, "renewal_negative_control.csv"), index=False)
    lv = extra["level_coefs"].copy()
    lv["label"] = lv["term"].map(pretty)
    lv.to_csv(os.path.join(OUT_DIR, "level_coefficients_primary.csv"), index=False)
    for name, calib in extra["calibs"].items():
        calib.assign(model=name).to_csv(os.path.join(OUT_DIR, f"window_calibration_{name}.csv"), index=False)
    plot_forest(coefs, os.path.join(OUT_DIR, "fig_forest.png"))
    plot_month1(extra["conf"], extra["hazards"], os.path.join(OUT_DIR, "fig_month1.png"))

    payload = {
        "data": meta,
        "seeds": list(SEEDS),
        "primary_seed": PRIMARY_SEED,
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "candidates": len(candidate_terms()),
        "alpha": ALPHA,
        "confirmations": [
            {k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in c.items()} for c in confirmations
        ],
        "renewal_control": extra["renewal"].to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(freq_table.to_string(index=False))
    print(coefs.round(3).to_string(index=False))
    print(extra["renewal"].round(3).to_string(index=False))
    print(conf_table[["seed", "selected", "lr_stat", "lr_p", "wald_stat", "wald_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
