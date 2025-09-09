# scripts/evaluate_labels.py
import sys, re, warnings
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import krippendorff
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# Feature toggles (flags)
# -----------------------
TEXT_ONLY_JOIN = True                 # match by text only (permissive)
SAVE_GROUP_BREAKDOWNS = False        # per-investor / per-PDF tables & plots
SAVE_PR_CURVES = True               # per-investor PR curves/thresholds
MIN_N = 11                            # for group breakdowns

# Try fuzzy matching for investor synonyms (optional; pip install rapidfuzz)
TRY_FUZZY = True
try:
    from rapidfuzz import fuzz
except Exception:
    TRY_FUZZY = False

# =============================
# --- Config: file locations ---
# =============================
manual_path = Path(
    r"C:\Users\6559484\OneDrive - Universiteit Utrecht\Desktop\BiodiversityASSET_SODA\data\processed\manual_annotations\manual_annotation_investment_activity_classification.xlsx"
)
pred_path   = Path(
    r"C:\Users\6559484\OneDrive - Universiteit Utrecht\Desktop\BiodiversityASSET_SODA\data\processed\investment_activity_classification\combined_investment_activity_classification_dedup.csv"
)

assert manual_path.exists(), f"Missing manual file: {manual_path}"
assert pred_path.exists(),   f"Missing model file:  {pred_path}"

# Save EVERYTHING here (kept as in your snippet)
RESULTS_ROOT = Path(r"C:\Users\6559484\OneDrive - Universiteit Utrecht\Desktop\BiodiversityASSET_SODA\results")
OUT_DIR = RESULTS_ROOT / "accuracy_tests" / "investment_activity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Outputs will be saved to: {OUT_DIR}")

# =========================
# --- Load input tables ---
# =========================
def read_csv_flex(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")

# If Excel engine missing:  pip install openpyxl
df_gold_raw = pd.read_excel(manual_path)
df_pred_raw = read_csv_flex(pred_path)

# --- NEW: make column labels unique (fix duplicate header names like 'pdf_file_name') ---
df_gold_raw = df_gold_raw.loc[:, ~df_gold_raw.columns.duplicated(keep="first")]
df_pred_raw = df_pred_raw.loc[:, ~df_pred_raw.columns.duplicated(keep="first")]

# =========================================================
# --- Filter model rows that reference manual annotations ---
# =========================================================
manual_file_key = "manual_annotation_investment_activity_classification"  # substring to detect manual-origin rows
mask_cols = [c for c in ["csv_file_name", "source_csv_file"] if c in df_pred_raw.columns]
if mask_cols:
    origin_blob = df_pred_raw[mask_cols].astype(str).agg(" ".join, axis=1).str.lower()
    before = len(df_pred_raw)
    df_pred = df_pred_raw[~origin_blob.str.contains(manual_file_key, na=False)].copy()
    removed = before - len(df_pred)
    print(f"[INFO] Filtered {removed} model rows that referenced manual annotations ({mask_cols}).")
else:
    print("[WARN] No 'csv_file_name'/'source_csv_file' in model df; cannot origin-filter.")
    df_pred = df_pred_raw.copy()

# ===========================================
# --- Normalize text & choose join columns ---
# ===========================================
def norm_text(s: str) -> str:
    if pd.isna(s): return ""
    x = str(s)
    # robust typography normalization
    x = x.replace("\u00a0", " ")  # non-breaking space
    x = x.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes -> hyphen
    x = x.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes -> '
    x = x.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes -> "
    x = x.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(x.split())  # collapse whitespace

required = {"paragraph_text", "score"}
for name, dfc in [("manual", df_gold_raw), ("model", df_pred)]:
    missing = required - set(dfc.columns)
    if missing:
        raise ValueError(f"{name} file missing columns: {missing}. Columns present: {list(dfc.columns)}")

df_gold_raw["paragraph_text_norm"] = df_gold_raw["paragraph_text"].map(norm_text)
df_pred["paragraph_text_norm"]     = df_pred["paragraph_text"].map(norm_text)

# ---- FORCE permissive text-only join (your request) ----
if TEXT_ONLY_JOIN:
    join_keys = ["paragraph_text_norm"]
else:
    if "pdf_file_name" in df_gold_raw.columns and "pdf_file_name" in df_pred.columns:
        join_keys = ["pdf_file_name", "paragraph_text_norm"]
    else:
        join_keys = ["paragraph_text_norm"]

# ==========================================
# --- Resolve duplicates in manual (gold) ---
# ==========================================
gold = df_gold_raw.copy()
dup_mask = gold.duplicated(subset=join_keys, keep=False)

if dup_mask.any():
    print(f"[INFO] Found {dup_mask.sum()} manual rows in duplicate groups. Resolving…")

    def resolve_group(grp: pd.DataFrame):
        vals = pd.to_numeric(grp["score"], errors="coerce").dropna().astype(int)
        if vals.empty:
            return pd.Series({"resolved_score": pd.NA, "resolution": "no_label"})
        if vals.nunique() == 1:
            return pd.Series({"resolved_score": int(vals.iloc[0]), "resolution": "unanimous"})
        counts = vals.value_counts()
        if len(counts) > 1 and counts.iloc[0] > counts.iloc[1]:
            return pd.Series({"resolved_score": int(counts.idxmax()), "resolution": "majority"})
        else:
            return pd.Series({"resolved_score": pd.NA, "resolution": "tie"})

    res = (
        gold.loc[dup_mask]
        .groupby(join_keys, dropna=False, as_index=False)
        .apply(resolve_group)
        .reset_index(drop=True)
    )

    # Save duplicate details
    dup_detail = gold.loc[dup_mask].sort_values(join_keys).copy()
    dup_report = dup_detail.merge(res, on=join_keys, how="left")
    dup_report_path = OUT_DIR / "manual_duplicates_report.csv"
    dup_report.to_csv(dup_report_path, index=False)
    print(f"[INFO] Duplicate details saved to: {dup_report_path}")

    # Exclude unresolved groups (tie / no_label)
    unresolved = res[res["resolved_score"].isna()]
    if not unresolved.empty:
        conflict_rows = gold.merge(unresolved[join_keys], on=join_keys, how="inner")
        conflict_path = OUT_DIR / "manual_conflicting_duplicates.csv"
        conflict_rows.to_csv(conflict_path, index=False)
        print(f"[INFO] {unresolved.shape[0]} duplicate groups had ties/no labels. "
              f"Excluded from eval. Details: {conflict_path}")

    # Build deduplicated manual
    base = gold[~dup_mask].copy()
    resolved_ok = res[res["resolved_score"].notna()].copy()
    if not resolved_ok.empty:
        reps = (
            gold.merge(resolved_ok[join_keys], on=join_keys, how="inner")
                .drop_duplicates(subset=join_keys)
                .copy()
        )
        reps = reps.merge(resolved_ok[join_keys + ["resolved_score"]], on=join_keys, how="left")
        reps["score"] = reps["resolved_score"].astype("Int64")
        reps.drop(columns=["resolved_score"], inplace=True)
        df_gold = pd.concat([base, reps], ignore_index=True)
    else:
        df_gold = base
else:
    print("[INFO] No manual duplicates found by join keys.")
    df_gold = gold

# One row per join key (to satisfy validate='one_to_one')
df_gold = df_gold.drop_duplicates(subset=join_keys).copy()
df_pred = df_pred.drop_duplicates(subset=join_keys).copy()

# ============================
# --- Investor synonym builder (NEW) ---
# ============================
def _simple_clean(s: str) -> str:
    if pd.isna(s):
        return ""
    x = str(s).strip().lower()
    x = re.sub(r"[\s\-_]+", " ", x)
    x = re.sub(r"[.,;:()\\/]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

CORP_TOKENS = set("""
bank am asset assets management invest investments investment group nv sa plc llp llc ltd inc co ag se spa srl s.a.
partners capital advisors holdings services international national global
""".split())

def _base_key(s: str) -> str:
    x = _simple_clean(s)
    if not x:
        return ""
    toks = [t for t in re.split(r"\s+", x) if t and t not in CORP_TOKENS]
    toks = sorted(toks)
    return " ".join(toks)

def _pretty_title(s: str) -> str:
    if not s:
        return s
    words = s.split()
    out = []
    for w in words:
        if len(w) <= 6 and w.isupper():
            out.append(w)  # keep short acronyms
        else:
            out.append(w.title())
    return " ".join(out)

def _choose_canonical(variants: list[str]) -> str:
    counts = Counter(variants)
    most_common, _ = counts.most_common(1)[0]
    upper_short = [v for v in variants if v.isupper() and 2 <= len(v) <= 6]
    if upper_short:
        cand, _ = Counter(upper_short).most_common(1)[0]
        return cand
    return _pretty_title(_simple_clean(most_common))

# Collect raw names across BOTH original tables (before join)
raw_names = []
if "investor_name" in df_gold_raw.columns:
    raw_names += [n for n in df_gold_raw["investor_name"].dropna().astype(str) if n.strip()]
if "investor_name" in df_pred_raw.columns:
    raw_names += [n for n in df_pred_raw["investor_name"].dropna().astype(str) if n.strip()]
raw_names = list(dict.fromkeys(raw_names))

syn_csv = OUT_DIR / "investor_name_synonyms.csv"
syn_review_csv = OUT_DIR / "investor_name_synonyms_reviewed.csv"

mapping_clean_to_canonical = {}
if raw_names:
    # initial buckets
    buckets = defaultdict(list)
    for n in raw_names:
        buckets[_base_key(n)].append(n)

    # optional fuzzy merge of base keys
    if TRY_FUZZY:
        keys = list(buckets.keys())
        parent = {k: k for k in keys}

        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        FUZZY_THRESHOLD = 92  # 0..100 (higher = stricter)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                if not k1 or not k2:
                    continue
                if fuzz.token_sort_ratio(k1, k2) >= FUZZY_THRESHOLD:
                    union(k1, k2)

        merged = defaultdict(list)
        for k, variants in buckets.items():
            root = find(k)
            merged[root].extend(variants)
        buckets = merged

    # suggestions CSV + mapping
    suggestion_rows = []
    for k, variants in buckets.items():
        canonical = _choose_canonical(variants)
        v_counts = Counter(variants)
        for v, c in sorted(v_counts.items(), key=lambda x: (-x[1], x[0].lower())):
            suggestion_rows.append({
                "cluster_key": k,
                "canonical_suggestion": canonical,
                "variant": v,
                "variant_clean": _simple_clean(v),
                "count": c,
            })
            mapping_clean_to_canonical[_simple_clean(v)] = canonical

    pd.DataFrame(suggestion_rows).to_csv(syn_csv, index=False)
    print(f"[INFO] Investor synonym suggestions saved to: {syn_csv}")

    # Apply reviewed overrides if present
    if syn_review_csv.exists():
        try:
            df_rev = pd.read_csv(syn_review_csv, encoding="utf-8-sig")
            guess_var = "variant_clean" if "variant_clean" in df_rev.columns else "variant"
            guess_can = "canonical" if "canonical" in df_rev.columns else (
                        "canonical_suggestion" if "canonical_suggestion" in df_rev.columns else None)
            if guess_can is None:
                raise ValueError("Reviewed CSV missing a canonical column (expected 'canonical' or 'canonical_suggestion').")
            overrides = {}
            for _, r in df_rev.iterrows():
                v = str(r.get(guess_var, "") or "")
                c = str(r.get(guess_can, "") or "")
                v_clean = _simple_clean(v)
                if v_clean and c:
                    overrides[v_clean] = c
            mapping_clean_to_canonical.update(overrides)
            print(f"[INFO] Loaded reviewed investor canonical names from: {syn_review_csv}")
        except Exception as e:
            print(f"[WARN] Failed to load reviewed synonyms ({syn_review_csv}): {e}")


# ===================
# --- Inner join  ---
# ===================
# Build RHS columns from PRED without duplicating join keys
rhs_extra = [c for c in ["score", "investor_name", "csv_file_name", "pdf_file_name", "paragraph_text"]
             if c in df_pred.columns and c not in join_keys]
rhs_cols = list(dict.fromkeys(list(join_keys) + rhs_extra))

# Also keep the GOLD-side investor_name so we can fall back if PRED is empty
gold_keep = [c for c in ["investor_name", "csv_file_name", "pdf_file_name", "paragraph_text"] if c in df_gold.columns]

df = df_gold[join_keys + ["score"] + gold_keep].merge(
    df_pred[rhs_cols],
    on=join_keys,
    suffixes=("_gold", "_pred"),
    how="inner",
    validate="one_to_one"
)

if df.empty:
    raise ValueError(
        "After filtering and duplicate resolution, no overlap remains for comparison.\n"
        f"Join keys: {join_keys}\n"
        "Check normalization or whether the model df has predictions for these rows."
    )

# ===============================
# --- Ensure numeric labels    ---
# ===============================
df["score_gold"] = pd.to_numeric(df["score_gold"], errors="coerce").astype("Int64")
df["score_pred"] = pd.to_numeric(df["score_pred"], errors="coerce").astype("Int64")
df = df.dropna(subset=["score_gold", "score_pred"]).astype({"score_gold": "int32", "score_pred": "int32"})

# ===========================================
# --- Build a combined investor_name first ---
# ===========================================
def coalesce(*vals):
    for v in vals:
        if pd.notna(v) and str(v).strip():
            return str(v)
    return ""

pred_name_col = "investor_name_pred" if "investor_name_pred" in df.columns else ("investor_name" if "investor_name" in df.columns else None)
gold_name_col = "investor_name_gold" if "investor_name_gold" in df.columns else ("investor_name" if "investor_name" in df.columns else None)

# Create a unified investor_name_combined (prefer PRED, then GOLD)
if pred_name_col or gold_name_col:
    df["investor_name_combined"] = [
        coalesce(
            df.loc[i, pred_name_col] if pred_name_col else "",
            df.loc[i, gold_name_col] if gold_name_col else ""
        )
        for i in df.index
    ]
else:
    df["investor_name_combined"] = ""

# ===========================================
# --- Apply investor canonical mapping     ---
# ===========================================
def _apply_investor_norm(name: str) -> str:
    if pd.isna(name): return ""
    key = _simple_clean(name)
    canon = mapping_clean_to_canonical.get(key)
    if canon:
        return canon
    return _pretty_title(key)

df["investor_norm"] = df["investor_name_combined"].map(_apply_investor_norm)

# If everything is blank, skip per-investor outputs later
if df["investor_norm"].nunique() <= 1:
    print("[INFO] investor_norm has only one unique value; skipping per-investor breakdowns/plots.")
    SAVE_GROUP_BREAKDOWNS = False


# ======================================
# --- Overall class balance & baseline ---
# ======================================
p1 = df["score_gold"].mean()
print(f"\nN (after filtering/join): {len(df)}")
print(f"Share of class 1 (gold): {p1:.3%}")
counts = df["score_gold"].value_counts().reindex([0, 1], fill_value=0)
print("Counts gold (0,1):", counts.to_dict())
perc = df["score_gold"].value_counts(normalize=True).reindex([0, 1], fill_value=0).mul(100).round(2)
print("Perc gold (0,1):", (perc.astype(str) + "%").to_dict())
baseline_acc = max(p1, 1 - p1)
print(f"Majority-class baseline accuracy: {baseline_acc:.3%}")

# Save quick balance to CSV
(pd.DataFrame({"N":[len(df)], "share_class_1":[p1], "baseline_acc":[baseline_acc]})
   .to_csv(OUT_DIR / "overall_balance.csv", index=False))

# ====================================================
# --- Optional group breakdowns (default: OFF) ---
# ====================================================
if SAVE_GROUP_BREAKDOWNS:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # By investor (normalized)
    if "investor_norm" in df.columns:
        inv_bal = (
            df.dropna(subset=["investor_norm"])
              .groupby("investor_norm")["score_gold"]
              .agg(count="count", share_class_1="mean")
        )
        inv_over10 = inv_bal[inv_bal["count"] >= MIN_N].sort_values(
            ["share_class_1", "count"], ascending=[True, False]
        )
        print("\nClass balance by investor (>10 rows, normalized names):")
        print(inv_over10.head(100))
        inv_csv = OUT_DIR / "class_balance_by_investor_over10.csv"
        inv_over10.to_csv(inv_csv)
        print(f"Saved investor balance table to: {inv_csv}")

        if not inv_over10.empty:
            fig_h = max(4, 0.35 * len(inv_over10))
            fig, ax = plt.subplots(figsize=(10, fig_h))
            ax.bar(inv_over10.index.astype(str), inv_over10["share_class_1"].values)
            ax.set_ylabel("Share of class 1 (gold)")
            ax.set_xlabel("Investor (normalized)")
            ax.set_title(f"Class 1 share by investor (n ≥ {MIN_N}) — N={len(df)}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out_png = OUT_DIR / "class_balance_by_investor_over10.png"
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            print(f"Saved investor plot to: {out_png}")

    # By PDF
    if "pdf_file_name" in df.columns:
        pdf_bal = (
            df.dropna(subset=["pdf_file_name"])
              .groupby("pdf_file_name")["score_gold"]
              .agg(count="count", share_class_1="mean")
        )
        pdf_over10 = pdf_bal[pdf_bal["count"] >= MIN_N].sort_values(
            ["share_class_1", "count"], ascending=[True, False]
        )
        print("\nClass balance by PDF (>10 rows):")
        print(pdf_over10.head(100))
        pdf_csv = OUT_DIR / "class_balance_by_pdf_over10.csv"
        pdf_over10.to_csv(pdf_csv)
        print(f"Saved PDF balance table to: {pdf_csv}")

# ==========================================
# --- Per-investor metrics & macro averages ---
# ==========================================
def metrics_for_slice(y_true, y_pred) -> pd.Series:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.Series({
        "count": len(y_true),
        "share_class_1": float(pd.Series(y_true).mean()),
        "prec": p, "rec": r, "f1": f1, "mcc": mcc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })

per_inv = None
macro_stats = None
if "investor_norm" in df.columns and SAVE_GROUP_BREAKDOWNS:
    per_inv = df.groupby("investor_norm").apply(
        lambda g: metrics_for_slice(g["score_gold"], g["score_pred"])
    ).sort_values(["count", "f1"], ascending=[False, False])

    inv_metrics_csv = OUT_DIR / "per_investor_metrics.csv"
    per_inv.to_csv(inv_metrics_csv)
    print(f"\nSaved per-investor metrics to: {inv_metrics_csv}")

    eligible = per_inv[per_inv["count"] >= MIN_N].copy()
    eligible = eligible[(eligible["tp"] + eligible["fn"] > 0) & (eligible["tn"] + eligible["fp"] > 0)]
    if not eligible.empty:
        macro_prec = eligible["prec"].mean()
        macro_rec  = eligible["rec"].mean()
        macro_f1   = eligible["f1"].mean()
        macro_mcc  = eligible["mcc"].mean()

        TP = int(eligible["tp"].sum())
        FP = int(eligible["fp"].sum())
        FN = int(eligible["fn"].sum())
        micro_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        micro_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

        macro_stats = {
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "macro_mcc": macro_mcc,
            "micro_precision": micro_prec,
            "micro_recall": micro_rec,
            "micro_f1": micro_f1,
            "eligible_groups": len(eligible),
        }

        print("\nMacro-averaged per-investor scores (groups with ≥ "
              f"{MIN_N} and both classes present):")
        print(f"Macro Precision: {macro_prec:.4f}")
        print(f"Macro Recall   : {macro_rec:.4f}")
        print(f"Macro F1       : {macro_f1:.4f}")
        print(f"Macro MCC      : {macro_mcc:.4f}")

        print("\nMicro (via totals over the same groups):")
        print(f"Micro Precision: {micro_prec:.4f}")
        print(f"Micro Recall   : {micro_rec:.4f}")
        print(f"Micro F1       : {micro_f1:.4f}")
    else:
        print(f"\n[INFO] No eligible investors (≥ {MIN_N} rows and both classes present) for macro per-group averages.")

# =========================================================
# --- Optional: per-investor PR curves & best-F1 thresholds ---
# =========================================================
prob_candidates = [c for c in ["prob_1","score_prob","pred_prob","proba","model_score","prob"] if c in df.columns]
prob_col = prob_candidates[0] if prob_candidates else None

if prob_col and "investor_norm" in df.columns and SAVE_PR_CURVES:
    from matplotlib import pyplot as plt
    curves_dir = OUT_DIR / "per_investor_pr_curves"
    curves_dir.mkdir(exist_ok=True)

    thresh_rows = []
    for inv, g in df.groupby("investor_norm"):
        y_true = g["score_gold"].values
        y_score = pd.to_numeric(g[prob_col], errors="coerce").fillna(0.0).values
        if (y_true.sum() == 0) or (y_true.sum() == len(y_true)) or (len(g) < MIN_N):
            continue
        precision, recall, thresh = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        prec_, rec_, thr_ = precision[:-1], recall[:-1], thresh
        f1_points = (2 * prec_ * rec_) / (prec_ + rec_ + 1e-12)
        best_idx = f1_points.argmax()
        best_thr = float(thr_[best_idx])
        best_f1  = float(f1_points[best_idx])

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR curve — {inv}\n(AP={ap:.3f}, n={len(g)})")
        ax.grid(True, linestyle=":", linewidth=0.5)
        fig.tight_layout()
        safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", inv)[:80]
        fig.savefig(curves_dir / f"PR_{safe_name}.png", dpi=150)
        plt.close(fig)

        thresh_rows.append({
            "investor_norm": inv,
            "count": int(len(g)),
            "avg_precision": float(ap),
            "best_f1": best_f1,
            "best_threshold": best_thr
        })

    if thresh_rows:
        thr_df = pd.DataFrame(thresh_rows).sort_values(["best_f1", "count"], ascending=[False, False])
        thr_csv = OUT_DIR / "per_investor_best_thresholds.csv"
        thr_df.to_csv(thr_csv, index=False)
        print(f"\nSaved per-investor best-F1 thresholds to: {thr_csv}")
        print(f"Saved PR curves to folder: {curves_dir}")
    else:
        print(f"\n[INFO] No eligible investors (probs + both classes + n ≥ {MIN_N}). Skipped PR curves/threshold export.")
elif not prob_col:
    print("\n[INFO] No probability column found "
          "(looked for 'prob_1','score_prob','pred_prob','proba','model_score','prob'). "
          "Skipping PR curves & threshold tuning.")

# =====================
# --- Global metrics ---
# =====================
acc  = accuracy_score(df["score_gold"], df["score_pred"])
prec, rec, f1, _ = precision_recall_fscore_support(df["score_gold"], df["score_pred"], average="binary", zero_division=0)

print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 score : {f1:.4f}")

print("\nPer-class report:\n")
class_report = classification_report(df["score_gold"], df["score_pred"], labels=[0,1], zero_division=0)
print(class_report)

cm = confusion_matrix(df["score_gold"], df["score_pred"], labels=[0,1])
cm_df = pd.DataFrame(cm, index=["gold:0","gold:1"], columns=["pred:0","pred:1"])
print("\nConfusion matrix:")
print(cm_df)

# ============================
# --- Krippendorff’s alpha ---
# ============================
alpha = krippendorff.alpha(
    reliability_data=[df["score_gold"].tolist(), df["score_pred"].tolist()],
    level_of_measurement="nominal"
)
print(f"\nKrippendorff's alpha: {alpha:.4f}")

# ==========================
# --- Save artifacts      ---
# ==========================
# Save mismatches
keep_cols = [c for c in [
    "pdf_file_name","investor_name","investor_norm","csv_file_name",
    "paragraph_text","score_gold","score_pred"
] if c in df.columns]
mismatch_path = OUT_DIR / "label_mismatches.csv"
df[df["score_gold"] != df["score_pred"]][keep_cols].to_csv(mismatch_path, index=False)
print(f"\nSaved {df[df['score_gold'] != df['score_pred']].shape[0]} mismatches to: {mismatch_path}")

# Confusion matrix CSV
cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")

# Classification report to txt
with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as ftxt:
    ftxt.write(class_report)

# Concise summary
summary_lines = [
    f"N (after join/filter): {len(df)}",
    f"Share class 1 (gold): {p1:.4f}",
    f"Baseline accuracy   : {baseline_acc:.4f}",
    f"Accuracy            : {acc:.4f}",
    f"Precision (pos=1)   : {prec:.4f}",
    f"Recall    (pos=1)   : {rec:.4f}",
    f"F1        (pos=1)   : {f1:.4f}",
    f"Krippendorff alpha  : {alpha:.4f}",
    f"Fuzzy matching used : {TRY_FUZZY}",
    f"Synonyms CSV        : {syn_csv}",
    f"Reviewed CSV (opt.) : {syn_review_csv}",
    f"TEXT_ONLY_JOIN      : {TEXT_ONLY_JOIN}",
    f"SAVE_GROUP_BREAKDOWNS: {SAVE_GROUP_BREAKDOWNS}",
    f"SAVE_PR_CURVES      : {SAVE_PR_CURVES}",
]
with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as fs:
    fs.write("\n".join(summary_lines))

print(f"[INFO] Summary written to: {OUT_DIR / 'summary.txt'}")


print("unique investor_name_pred:", df.get("investor_name_pred", pd.Series(dtype=str)).nunique())
print("unique investor_name_gold:", df.get("investor_name_gold", pd.Series(dtype=str)).nunique())
print("unique investor_name_combined:", df["investor_name_combined"].nunique())
print("unique investor_norm:", df["investor_norm"].nunique())
