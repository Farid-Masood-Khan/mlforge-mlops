"""Model promotion gate enforcing champion/challenger rules in the MLflow registry.

Promotion rule: candidate holdout PR-AUC >= champion holdout PR-AUC + 0.002.
Modern MLflow 2.x alias management (champion / challenger).
"""

from __future__ import annotations

import argparse

from mlflow.tracking import MlflowClient


def evaluate_and_promote_candidate(
    run_id: str,
    candidate_pr_auc: float,
    model_name: str = "mlforge-churn",
    delta_threshold: float = 0.002,
    force: bool = False,
    override_reason: str = "",
) -> tuple[str, str]:
    """Evaluates candidate against champion and assigns appropriate alias in registry.

    Returns:
        (assigned_alias, explanation_message)
    """
    client = MlflowClient()

    # Ensure registered model exists
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(
            model_name,
            description="MLForge production churn prediction model registry.",
        )

    # Create model version from the run's logged artifact
    model_uri = f"runs:/{run_id}/model"
    model_version_obj = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=f"Automated registration from run {run_id} with PR-AUC {candidate_pr_auc:.4f}",
    )
    version = str(model_version_obj.version)

    # Check if a champion currently exists
    champion_version = None
    champion_pr_auc = -1.0
    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
        champion_version = champ.version
        champ_run = client.get_run(champ.run_id)
        champion_pr_auc = float(champ_run.data.metrics.get("pr_auc", -1.0))
    except Exception:
        # No champion alias set yet
        champion_version = None

    if champion_version is None:
        # First model registered automatically becomes champion
        client.set_registered_model_alias(model_name, "champion", version)
        msg = f"Version {version} promoted to 'champion' (initial registry baseline, PR-AUC: {candidate_pr_auc:.4f})."
        print(f"[Promote] {msg}")
        return "champion", msg

    delta = candidate_pr_auc - champion_pr_auc
    promote_to_champion = False

    if force:
        promote_to_champion = True
        reason = override_reason or "Manual administrative override"
        client.set_model_version_tag(
            model_name,
            version,
            "promotion_override_reason",
            reason,
        )
        msg = f"Version {version} forced to 'champion' via override: '{reason}' (PR-AUC: {candidate_pr_auc:.4f} vs champion {champion_pr_auc:.4f})."
    elif delta >= delta_threshold:
        promote_to_champion = True
        msg = f"Version {version} promoted to 'champion'! Beat champion v{champion_version} by {delta:+.4f} (PR-AUC: {candidate_pr_auc:.4f} vs {champion_pr_auc:.4f}, required delta: +{delta_threshold})."
    else:
        promote_to_champion = False
        msg = f"Version {version} assigned to 'challenger'. PR-AUC {candidate_pr_auc:.4f} did not beat champion v{champion_version} ({champion_pr_auc:.4f} + {delta_threshold})."

    if promote_to_champion:
        # Demote previous champion to challenger
        client.set_registered_model_alias(model_name, "challenger", champion_version)
        client.set_registered_model_alias(model_name, "champion", version)
        assigned_alias = "champion"
    else:
        client.set_registered_model_alias(model_name, "challenger", version)
        assigned_alias = "challenger"

    print(f"[Promote] {msg}")
    return assigned_alias, msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote or alias registered models.")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to evaluate.")
    parser.add_argument("--pr-auc", type=float, required=True, help="Holdout PR-AUC.")
    parser.add_argument("--model-name", default="mlforge-churn", help="Registered model name.")
    parser.add_argument("--delta", type=float, default=0.002, help="PR-AUC improvement margin.")
    parser.add_argument("--force", action="store_true", help="Force promotion to champion.")
    parser.add_argument("--reason", default="", help="Override reason when forcing.")

    args = parser.parse_args()
    evaluate_and_promote_candidate(
        run_id=args.run_id,
        candidate_pr_auc=args.pr_auc,
        model_name=args.model_name,
        delta_threshold=args.delta,
        force=args.force,
        override_reason=args.reason,
    )
