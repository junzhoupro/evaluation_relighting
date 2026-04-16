import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# -------------------------------------------------
# User config
# -------------------------------------------------
# Directory that contains your per-hand metrics CSV files,
# e.g. metric_results_aged-hand_choose.csv, metric_results_blackgirl_newnecklaces.csv, etc.
METRICS_DIR = "."
METRICS_GLOB = "metric_results_*.csv"

# Output directory
OUT_DIR = "./metrics_vs_human_outputs"

# Human crosstab files
CROSSTAB_FILES = {
    ("ring", "identity"): "./Export_Crosstab-jewelry_192-New Crosstab-054a49fb-ba2d-43fe-965f-9b8542d67405.xlsx",
    ("ring", "aesthetic"): "./Export_Crosstab-jewelry_attractive_1-New Crosstab-aef53b16-cfd9-4162-96dc-25bd83673869.xlsx",
    ("necklace", "identity"): "./Export_Crosstab-newnecklaces-New Crosstab-0ccb67e8-899a-430b-98f3-7a5979bbfbfe.xlsx",
    ("necklace", "aesthetic"): "./Export_Crosstab-newnecklaces_attract-New Crosstab-d7975062-5dfc-46d6-8d09-8adb79b207ab.xlsx",
}

# A/B/C mapping from the survey files
METHOD_MAP = {
    "A": "diffusion",
    "B": "hist",
    "C": "composite",
}

# Metric direction: higher_better or lower_better
# Add your own columns here when you extend the metric CSV.
METRIC_DIRECTIONS = {
    "masked_lpips": "lower_better",
    "masked_ssim": "higher_better",
    "masked_psnr": "higher_better",
    # Example custom metrics:
    "hist_chi2_fg_boundary": "lower_better",
    "hist_chi2_fg_fg": "lower_better",
}

# Columns that are numeric but should NOT be treated as evaluation metrics.
EXCLUDE_NUMERIC_COLS = {
    "mask_pixels",
    "crop_h",
    "crop_w",
}

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def normalize_hdr_key(text: str) -> str:
    text = str(text).lower()
    if "indoor_pool" in text or "indoor_p" in text or "indoor" in text:
        return "indoor_pool"
    if "cinema_hall" in text or "cinema_h" in text or "cinema" in text:
        return "cinema_hall"
    if "industrial_pipe" in text or "industrial_p" in text or "industri" in text:
        return "industrial_pipe"
    return text



def strip_prompt(question_text: str) -> str:
    return str(question_text).split(":")[0].strip()



def parse_question_prefix(question_text: str) -> Tuple[str, str]:
    """
    Returns:
        before_hdr: full prefix before the HDR token
        hdr_key: one of indoor_pool / cinema_hall / industrial_pipe

    Example:
        aged-hand_choose_aged-hand_golden-ring-test_gold_indoor_pool_4k
        -> (aged-hand_choose_aged-hand_golden-ring-test_gold, indoor_pool)
    """
    prefix = strip_prompt(question_text)
    hdr_key = normalize_hdr_key(prefix)

    if hdr_key == "indoor_pool":
        candidates = ["indoor_pool", "indoor_p", "indoor"]
    elif hdr_key == "cinema_hall":
        candidates = ["cinema_hall", "cinema_h", "cinema"]
    elif hdr_key == "industrial_pipe":
        candidates = ["industrial_pipe", "industrial_p", "industri"]
    else:
        raise ValueError(f"Could not identify HDR type from question: {question_text}")

    cut_idx = -1
    for token in candidates:
        idx = prefix.find(token)
        if idx != -1:
            cut_idx = idx
            break
    if cut_idx == -1:
        raise ValueError(f"Could not split HDR from question: {question_text}")

    before_hdr = prefix[:cut_idx].rstrip("_")
    return before_hdr, hdr_key



def load_all_metrics(metrics_dir: str, pattern: str) -> pd.DataFrame:
    paths = sorted(Path(metrics_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No metric CSV files found under {metrics_dir!r} matching {pattern!r}."
        )

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["source_csv"] = str(p)
        frames.append(df)

    if not frames:
        raise ValueError("Metric CSV files were found, but all of them are empty.")

    metrics = pd.concat(frames, ignore_index=True)

    required = {"hand_name", "hdr_name", "jewelry_name", "method"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metric CSV is missing required columns: {sorted(missing)}")

    metrics["hand_name"] = metrics["hand_name"].astype(str)
    metrics["jewelry_name"] = metrics["jewelry_name"].astype(str)
    metrics["method"] = metrics["method"].astype(str)
    metrics["hdr_key"] = metrics["hdr_name"].map(normalize_hdr_key)

    # Longest-first prefix matching is safer when some hand names share prefixes.
    hand_names = sorted(metrics["hand_name"].dropna().unique().tolist(), key=len, reverse=True)
    metrics.attrs["known_hand_names"] = hand_names

    return metrics



def split_before_hdr(before_hdr: str, known_hand_names: List[str]) -> Tuple[str, str]:
    """
    Split 'aged-hand_choose_aged-hand_golden-ring-test_gold' into:
        hand_name   = aged-hand_choose
        jewelry     = aged-hand_golden-ring-test_gold
    using known hand_name values from the metrics CSV files.
    """
    before_hdr = str(before_hdr)
    for hand_name in known_hand_names:
        prefix = hand_name + "_"
        if before_hdr.startswith(prefix):
            jewelry_name = before_hdr[len(prefix):]
            return hand_name, jewelry_name
    raise ValueError(
        f"Could not split question key {before_hdr!r} using known hand names: {known_hand_names}"
    )



def parse_crosstab_file(
    xlsx_path: str,
    category: str,
    task: str,
    known_hand_names: List[str],
) -> pd.DataFrame:
    """
    Parse the per-question A/B/C counts from the Qualtrics-style crosstab.

    Assumptions from your files:
      - row 5 (0-based row 4) contains the question text
      - row 9 (0-based row 8) contains Total / A / B / C counts
      - each question block spans 5 columns: question, Total, A, B, C
    """
    df = pd.read_excel(xlsx_path, header=None)
    rows = []

    for c in range(2, df.shape[1] - 3, 5):
        q = df.iat[4, c]
        if pd.isna(q):
            continue

        before_hdr, hdr_key = parse_question_prefix(q)
        hand_name, jewelry_name = split_before_hdr(before_hdr, known_hand_names)

        total = int(df.iat[8, c])
        a = int(df.iat[8, c + 1])
        b = int(df.iat[8, c + 2])
        cc = int(df.iat[8, c + 3])

        counts = {"diffusion": a, "hist": b, "composite": cc}
        question_key = f"{hand_name}|{hdr_key}|{jewelry_name}"

        for method_name, votes in counts.items():
            rows.append(
                {
                    "category": category,
                    "task": task,
                    "question_text": q,
                    "question_key": question_key,
                    "hand_name": hand_name,
                    "hdr_key": hdr_key,
                    "jewelry_name": jewelry_name,
                    "method": method_name,
                    "votes": int(votes),
                    "total_votes": int(total),
                    "vote_share": float(votes) / float(total) if total > 0 else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No question rows parsed from {xlsx_path}")
    return out



def build_human_votes_df(known_hand_names: List[str]) -> pd.DataFrame:
    frames = []
    for (category, task), path in CROSSTAB_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing crosstab file: {path}")
        frames.append(parse_crosstab_file(path, category, task, known_hand_names))
    human = pd.concat(frames, ignore_index=True)
    return human



def detect_metric_columns(metrics: pd.DataFrame) -> List[str]:
    numeric_cols = metrics.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = []
    for col in numeric_cols:
        if col in EXCLUDE_NUMERIC_COLS:
            continue
        if col in {"total_votes", "votes", "vote_share"}:
            continue
        metric_cols.append(col)

    # Keep only metrics we know how to interpret, unless user added them to METRIC_DIRECTIONS.
    filtered = [c for c in metric_cols if c in METRIC_DIRECTIONS]
    return filtered



def metric_score_for_correlation(df: pd.DataFrame, metric: str) -> pd.Series:
    direction = METRIC_DIRECTIONS[metric]
    if direction == "higher_better":
        return df[metric]
    if direction == "lower_better":
        return -df[metric]
    raise ValueError(f"Unknown direction for metric {metric}: {direction}")



def summarize_spearman(merged: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []

    for (category, task), g in merged.groupby(["category", "task"], dropna=False):
        valid = g[[metric, "vote_share"]].dropna()
        if len(valid) < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(metric_score_for_correlation(valid, metric), valid["vote_share"])
        rows.append(
            {
                "level": "overall",
                "category": category,
                "task": task,
                "metric": metric,
                "method": "ALL",
                "n_rows": len(valid),
                "spearman_rho": rho,
                "p_value": pval,
            }
        )

        for method_name, gm in g.groupby("method", dropna=False):
            valid_m = gm[[metric, "vote_share"]].dropna()
            if len(valid_m) < 3:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(metric_score_for_correlation(valid_m, metric), valid_m["vote_share"])
            rows.append(
                {
                    "level": "per_method",
                    "category": category,
                    "task": task,
                    "metric": metric,
                    "method": method_name,
                    "n_rows": len(valid_m),
                    "spearman_rho": rho,
                    "p_value": pval,
                }
            )

    return pd.DataFrame(rows)


def summarize_spearman_by_lighting(merged: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []

    # Overall correlation within each category x task x lighting
    for (category, task, hdr_key), g in merged.groupby(["category", "task", "hdr_key"], dropna=False):
        valid = g[[metric, "vote_share"]].dropna()
        if len(valid) < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(metric_score_for_correlation(valid, metric), valid["vote_share"])

        rows.append(
            {
                "level": "overall_by_lighting",
                "category": category,
                "task": task,
                "hdr_key": hdr_key,
                "metric": metric,
                "method": "ALL",
                "n_rows": len(valid),
                "spearman_rho": rho,
                "p_value": pval,
            }
        )

    # Optional: per-method correlation within each category x task x lighting
    for (category, task, hdr_key, method_name), g in merged.groupby(
        ["category", "task", "hdr_key", "method"], dropna=False
    ):
        valid = g[[metric, "vote_share"]].dropna()
        if len(valid) < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(metric_score_for_correlation(valid, metric), valid["vote_share"])

        rows.append(
            {
                "level": "per_method_by_lighting",
                "category": category,
                "task": task,
                "hdr_key": hdr_key,
                "metric": metric,
                "method": method_name,
                "n_rows": len(valid),
                "spearman_rho": rho,
                "p_value": pval,
            }
        )

    return pd.DataFrame(rows)

def choose_metric_winner(group: pd.DataFrame, metric: str) -> str:
    direction = METRIC_DIRECTIONS[metric]
    sub = group[["method", metric]].dropna()
    if sub.empty:
        return np.nan
    if direction == "higher_better":
        best_val = sub[metric].max()
    else:
        best_val = sub[metric].min()
    winners = sub.loc[sub[metric] == best_val, "method"].tolist()
    # If metric ties, mark as tie so we can skip that sample in winner-accuracy stats.
    return winners[0] if len(winners) == 1 else "TIE"



def choose_human_winner(group: pd.DataFrame) -> str:
    sub = group[["method", "votes"]].dropna()
    if sub.empty:
        return np.nan
    best_val = sub["votes"].max()
    winners = sub.loc[sub["votes"] == best_val, "method"].tolist()
    return winners[0] if len(winners) == 1 else "TIE"



def summarize_winner_accuracy(merged: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []

    grouped = merged.groupby(["category", "task", "question_key"], dropna=False)
    sample_rows = []
    for (category, task, question_key), g in grouped:
        human_winner = choose_human_winner(g)
        metric_winner = choose_metric_winner(g, metric)
        sample_rows.append(
            {
                "category": category,
                "task": task,
                "question_key": question_key,
                "human_winner": human_winner,
                "metric_winner": metric_winner,
            }
        )
    sample_df = pd.DataFrame(sample_rows)

    for (category, task), g in sample_df.groupby(["category", "task"], dropna=False):
        non_tied = g[(g["human_winner"] != "TIE") & (g["metric_winner"] != "TIE")].copy()
        correct = (non_tied["human_winner"] == non_tied["metric_winner"]).sum()
        n_eval = len(non_tied)
        acc = (correct / n_eval) if n_eval > 0 else np.nan
        rows.append(
            {
                "category": category,
                "task": task,
                "metric": metric,
                "n_total_samples": len(g),
                "n_evaluable_samples": n_eval,
                "n_correct": int(correct),
                "winner_accuracy": acc,
                "n_human_ties_or_metric_ties": len(g) - n_eval,
            }
        )

    return pd.DataFrame(rows), sample_df



def write_simple_report(path: str, summary_corr: pd.DataFrame, summary_acc: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("Metrics vs. Human Judgment Summary\n")
    lines.append("================================\n")

    lines.append("Spearman correlation with human vote share\n")
    lines.append("-----------------------------------------\n")
    for _, row in summary_corr[summary_corr["level"] == "overall"].sort_values(["metric", "category", "task"]).iterrows():
        lines.append(
            f"{row['metric']} | {row['category']} | {row['task']}: "
            f"rho={row['spearman_rho']:.4f} | p={row['p_value']:.4g} | n={int(row['n_rows'])}\n"
        )

    lines.append("\nWinner prediction accuracy\n")
    lines.append("--------------------------\n")
    for _, row in summary_acc.sort_values(["metric", "category", "task"]).iterrows():
        acc = row["winner_accuracy"]
        acc_str = "nan" if pd.isna(acc) else f"{acc:.4f}"
        lines.append(
            f"{row['metric']} | {row['category']} | {row['task']}: "
            f"acc={acc_str} | correct={int(row['n_correct'])}/{int(row['n_evaluable_samples'])} "
            f"| ties_skipped={int(row['n_human_ties_or_metric_ties'])}\n"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)



def main() -> None:
    ensure_dir(OUT_DIR)

    # 1) Load all metrics CSV files
    metrics = load_all_metrics(METRICS_DIR, METRICS_GLOB)
    known_hand_names = metrics.attrs["known_hand_names"]

    # 2) Parse all human crosstab files into one long dataframe
    human = build_human_votes_df(known_hand_names)

    # 3) Merge on hand_name + jewelry_name + hdr_key + method
    merge_cols = ["hand_name", "jewelry_name", "hdr_key", "method"]
    merged = human.merge(metrics, on=merge_cols, how="inner", validate="many_to_one")

    # Safety check: make sure every question-method row found a metric row
    missing_human = human.merge(metrics[merge_cols], on=merge_cols, how="left", indicator=True)
    missing_human = missing_human[missing_human["_merge"] == "left_only"].drop(columns=["_merge"])

    human_out = os.path.join(OUT_DIR, "human_votes_long.csv")
    human.to_csv(human_out, index=False)

    missing_out = os.path.join(OUT_DIR, "human_rows_without_metrics.csv")
    missing_human.to_csv(missing_out, index=False)

    merged_out = os.path.join(OUT_DIR, "metrics_vs_human_merged.csv")
    merged.to_csv(merged_out, index=False)

    metric_cols = detect_metric_columns(metrics)
    if not metric_cols:
        raise ValueError(
            "No usable metric columns were found. Check METRIC_DIRECTIONS and your metric CSV columns."
        )

    all_corr = []
    all_acc = []
    all_sample_winners = []

    for metric in metric_cols:
        corr_df = summarize_spearman(merged, metric)
        corr_lighting_df = summarize_spearman_by_lighting(merged, metric)

        acc_df, sample_winners = summarize_winner_accuracy(merged, metric)
        sample_winners["metric"] = metric

        all_corr.append(corr_df)
        all_corr.append(corr_lighting_df)
        all_acc.append(acc_df)
        all_sample_winners.append(sample_winners)

    corr_all = pd.concat(all_corr, ignore_index=True)
    acc_all = pd.concat(all_acc, ignore_index=True)
    winners_all = pd.concat(all_sample_winners, ignore_index=True)

    corr_all.to_csv(os.path.join(OUT_DIR, "spearman_summary.csv"), index=False)
    acc_all.to_csv(os.path.join(OUT_DIR, "winner_accuracy_summary.csv"), index=False)
    winners_all.to_csv(os.path.join(OUT_DIR, "sample_level_winners.csv"), index=False)

    corr_lighting_only = corr_all[corr_all["level"].isin(["overall_by_lighting", "per_method_by_lighting"])].copy()
    corr_lighting_only.to_csv(os.path.join(OUT_DIR, "spearman_summary_by_lighting.csv"), index=False)

    write_simple_report(
        os.path.join(OUT_DIR, "metrics_vs_human_report.txt"),
        corr_all,
        acc_all,
    )

    print("Done.")
    print(f"Saved human long-format votes to: {human_out}")
    print(f"Saved merged human+metrics table to: {merged_out}")
    print(f"Saved missing human rows (no metric match) to: {missing_out}")
    print(f"Saved Spearman summary to: {os.path.join(OUT_DIR, 'spearman_summary.csv')}")
    print(f"Saved winner accuracy summary to: {os.path.join(OUT_DIR, 'winner_accuracy_summary.csv')}")
    print(f"Saved text report to: {os.path.join(OUT_DIR, 'metrics_vs_human_report.txt')}")


if __name__ == "__main__":
    main()
