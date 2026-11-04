"""Main training pipeline orchestration for MLForge.

Trains baselines and gradient boosting candidate models, logs runs to MLflow,
selects validation decision thresholds, records holdout metrics, and promotes models.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.data.download import compute_sha256, download_dataset
from src.data.validate import run_validation
from src.features.build import (
    build_pipeline,
    save_reference_snapshot,
    split_dataset,
)
from src.models.evaluate import (
    create_error_analysis_dataframe,
    evaluate_split,
    optimize_threshold,
)
from src.models.promote import evaluate_and_promote_candidate

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "train_config.yaml"
)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config


def run_training_pipeline(
    config_path: Path = DEFAULT_CONFIG_PATH,
    force_promote: bool = False,
    override_reason: str = "",
) -> dict[str, Any]:
    """Executes the full end-to-end training and tracking pipeline."""
    config = load_config(config_path)
    base_dir = config_path.parent.parent

    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{base_dir / 'mlflow.db'}")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", config["project"]["experiment_name"])
    model_name = os.environ.get("MODEL_NAME", config["project"]["name"])
    seed = int(config["project"]["seed"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[Train] MLflow Tracking URI: {tracking_uri}")
    print(f"[Train] MLflow Experiment: {experiment_name}")

    # Ensure validated data exists
    processed_dir = base_dir / str(config["data"]["processed_dir"])
    data_file = processed_dir / "validated_churn.parquet"
    if not data_file.exists():
        raw_file = base_dir / str(config["data"]["raw_dir"]) / str(config["data"]["raw_filename"])
        if not raw_file.exists():
            download_dataset(config_path=config_path)
        run_validation(config_path=config_path)

    # Compute data hash for experiment lineage
    raw_csv = base_dir / str(config["data"]["raw_dir"]) / str(config["data"]["raw_filename"])
    data_sha256 = compute_sha256(raw_csv) if raw_csv.exists() else "unknown"

    df = pd.read_parquet(data_file)
    target_col = str(config["data"]["target_column"])
    split_cfg = config["data"]["split"]

    # Stratified train/val/holdout splits
    df_train, df_val, df_holdout = split_dataset(
        df,
        target_column=target_col,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=seed,
    )
    print(
        f"[Train] Dataset split sizes: Train={len(df_train)}, Val={len(df_val)}, Holdout={len(df_holdout)}"
    )

    # Save holdout reference snapshot for drift detection
    ref_dir = base_dir / str(config["data"]["reference_dir"])
    save_reference_snapshot(df_holdout, ref_dir, target_column=target_col)

    feature_cols = [c for c in df.columns if c != target_col]
    x_train, y_train = df_train[feature_cols], df_train[target_col].to_numpy()

    runs_summary: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Baseline Model (Logistic Regression)
    # -------------------------------------------------------------
    print("[Train] Training Baseline: Logistic Regression...")
    with mlflow.start_run(run_name="baseline-logistic-regression") as run:
        baseline_model = LogisticRegression(
            max_iter=int(config["model"]["baseline"]["params"]["max_iter"]),
            C=float(config["model"]["baseline"]["params"]["C"]),
            solver=str(config["model"]["baseline"]["params"]["solver"]),
            random_state=seed,
        )
        baseline_pipeline = build_pipeline(baseline_model)
        baseline_pipeline.fit(x_train, y_train)

        # Optimize threshold on validation split
        opt_threshold = optimize_threshold(baseline_pipeline, df_val, target_column=target_col)
        val_metrics = evaluate_split(
            baseline_pipeline,
            df_val,
            threshold=opt_threshold,
            target_column=target_col,
        )
        holdout_metrics = evaluate_split(
            baseline_pipeline,
            df_holdout,
            threshold=opt_threshold,
            target_column=target_col,
        )

        # Log parameters, metrics, tags
        mlflow.log_params(config["model"]["baseline"]["params"])
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("optimal_threshold", opt_threshold)

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in holdout_metrics.items():
            mlflow.log_metric(k, v)

        mlflow.set_tags(
            {
                "data_sha256": data_sha256,
                "seed": seed,
                "pipeline_version": str(config["project"]["pipeline_version"]),
                "model_family": "baseline",
            }
        )

        mlflow.sklearn.log_model(baseline_pipeline, artifact_path="model")

        runs_summary.append(
            {
                "run_id": run.info.run_id,
                "name": "baseline-logistic-regression",
                "val_pr_auc": val_metrics["pr_auc"],
                "holdout_pr_auc": holdout_metrics["pr_auc"],
                "holdout_roc_auc": holdout_metrics["roc_auc"],
                "holdout_f1": holdout_metrics["f1"],
                "threshold": opt_threshold,
                "pipeline": baseline_pipeline,
            }
        )

    # -------------------------------------------------------------
    # 2. Main Candidate Grid (HistGradientBoostingClassifier)
    # -------------------------------------------------------------
    print("[Train] Training Candidates: HistGradientBoosting grid...")
    lr_grid = [0.05, 0.1]
    leaf_grid = [15, 31]
    min_samples_grid = [20, 40]

    for lr in lr_grid:
        for leaf in leaf_grid:
            for min_leaf in min_samples_grid:
                run_name = f"hgb-lr{lr}-leaf{leaf}-min{min_leaf}"
                with mlflow.start_run(run_name=run_name) as run:
                    hgb_model = HistGradientBoostingClassifier(
                        learning_rate=lr,
                        max_leaf_nodes=leaf,
                        min_samples_leaf=min_leaf,
                        random_state=seed,
                    )
                    hgb_pipeline = build_pipeline(hgb_model)
                    hgb_pipeline.fit(x_train, y_train)

                    opt_threshold = optimize_threshold(
                        hgb_pipeline,
                        df_val,
                        target_column=target_col,
                    )
                    val_metrics = evaluate_split(
                        hgb_pipeline,
                        df_val,
                        threshold=opt_threshold,
                        target_column=target_col,
                    )
                    holdout_metrics = evaluate_split(
                        hgb_pipeline,
                        df_holdout,
                        threshold=opt_threshold,
                        target_column=target_col,
                    )

                    # Log to MLflow
                    params = {
                        "model_type": "hist_gradient_boosting",
                        "learning_rate": lr,
                        "max_leaf_nodes": leaf,
                        "min_samples_leaf": min_leaf,
                        "optimal_threshold": opt_threshold,
                    }
                    mlflow.log_params(params)

                    for k, v in val_metrics.items():
                        mlflow.log_metric(f"val_{k}", v)
                    for k, v in holdout_metrics.items():
                        mlflow.log_metric(k, v)

                    mlflow.set_tags(
                        {
                            "data_sha256": data_sha256,
                            "seed": seed,
                            "pipeline_version": str(config["project"]["pipeline_version"]),
                            "model_family": "candidate",
                        }
                    )

                    # Save error analysis artifact for the run
                    error_df = create_error_analysis_dataframe(
                        hgb_pipeline,
                        df_holdout,
                        threshold=opt_threshold,
                        target_column=target_col,
                    )
                    err_file = (
                        base_dir / "results" / f"holdout_predictions_{run.info.run_id[:8]}.parquet"
                    )
                    error_df.to_parquet(err_file, index=False)
                    mlflow.log_artifact(str(err_file), artifact_path="diagnostics")

                    mlflow.sklearn.log_model(hgb_pipeline, artifact_path="model")

                    runs_summary.append(
                        {
                            "run_id": run.info.run_id,
                            "name": run_name,
                            "val_pr_auc": val_metrics["pr_auc"],
                            "holdout_pr_auc": holdout_metrics["pr_auc"],
                            "holdout_roc_auc": holdout_metrics["roc_auc"],
                            "holdout_f1": holdout_metrics["f1"],
                            "threshold": opt_threshold,
                            "pipeline": hgb_pipeline,
                        }
                    )

    # -------------------------------------------------------------
    # 3. Best Model Selection & Promotion Gate
    # -------------------------------------------------------------
    # Best candidate selected based on validation PR-AUC
    candidates = [r for r in runs_summary if r["name"] != "baseline-logistic-regression"]
    best_candidate = max(candidates, key=lambda x: x["val_pr_auc"])
    print(
        f"[Train] Best Candidate: {best_candidate['name']} (Val PR-AUC: {best_candidate['val_pr_auc']:.4f}, Holdout PR-AUC: {best_candidate['holdout_pr_auc']:.4f})"
    )

    delta_thresh = float(config["promotion"]["pr_auc_delta_threshold"])
    assigned_alias, promo_msg = evaluate_and_promote_candidate(
        run_id=best_candidate["run_id"],
        candidate_pr_auc=best_candidate["holdout_pr_auc"],
        model_name=model_name,
        delta_threshold=delta_thresh,
        force=force_promote,
        override_reason=override_reason,
    )

    # -------------------------------------------------------------
    # 4. Write Honest Results Metrics
    # -------------------------------------------------------------
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_file = results_dir / "metrics_summary.json"
    clean_summary = [{k: v for k, v in r.items() if k != "pipeline"} for r in runs_summary]
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_sha256": data_sha256,
                "best_candidate_run_id": best_candidate["run_id"],
                "assigned_alias": assigned_alias,
                "promotion_message": promo_msg,
                "runs": clean_summary,
            },
            f,
            indent=2,
        )

    results_readme = results_dir / "README.md"
    readme_rows = "\n".join(
        [
            f"| `{r['name']}` | `{r['run_id'][:8]}` | {r['val_pr_auc']:.4f} | {r['holdout_pr_auc']:.4f} | {r['holdout_roc_auc']:.4f} | {r['holdout_f1']:.4f} | {r['threshold']:.2f} |"
            for r in runs_summary
        ]
    )

    results_readme_content = f"""# MLForge — Measured Holdout Metrics

All numbers below were measured directly on the held-out test split (15% stratified, seed={seed}) from dataset hash `{data_sha256}`.

| Model / Run | Run ID | Val PR-AUC | Holdout PR-AUC | Holdout ROC-AUC | Holdout F1 | Decision Threshold |
|---|---|---|---|---|---|---|
{readme_rows}

## Registry Status
- **Model Name:** `{model_name}`
- **Active Champion Run:** `{best_candidate['run_id']}`
- **Assigned Alias:** `{assigned_alias}`
- **Promotion Audit:** {promo_msg}
"""
    results_readme.write_text(results_readme_content, encoding="utf-8")
    print(f"[Train] Results successfully recorded to {results_readme}")

    return {
        "best_candidate": best_candidate,
        "assigned_alias": assigned_alias,
        "runs_summary": clean_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute MLForge training pipeline.")
    parser.add_argument("--force-promote", action="store_true", help="Force candidate promotion.")
    parser.add_argument("--reason", default="", help="Override reason if force promoting.")
    args = parser.parse_args()

    run_training_pipeline(force_promote=args.force_promote, override_reason=args.reason)
