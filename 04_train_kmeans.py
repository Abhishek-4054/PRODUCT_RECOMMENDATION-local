"""
04_train_kmeans.py
KMeans clustering for user segmentation.
Reads data/processed/kmeans_input.csv
Writes models/kmeans_artifacts/: kmeans_model.pkl, metrics.json, user_clusters.csv
Optionally plots cluster profiles (--plot flag).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import argparse
import os
import pickle
import json

FEATURES = [
    "total_spend", "purchase_frequency", "avg_basket_value",
    "unique_products", "category_diversity", "price_sensitivity",
]

MODEL_DIR = "models/kmeans_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train KMeans user segmentation model")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--plot", action="store_true", help="Save cluster profile plot")
    return parser.parse_args()


def train():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv("data/processed/kmeans_input.csv")
    user_ids = df["user_id"].values
    X = df[FEATURES].values
    print(f"Training data shape: {X.shape}")

    # ── Elbow method ───────────────────────────────────────────────────────────
    print("\n── Elbow Method ──")
    inertias = []
    sil_scores = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        print(f"  K={k} | Inertia={km.inertia_:.1f} | Silhouette={silhouette_score(X, labels):.4f}")

    # ── Train final model ──────────────────────────────────────────────────────
    print(f"\n── Training final model with K={args.n_clusters} ──")
    final_km = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_init=10,
    )
    final_labels = final_km.fit_predict(X)

    # ── Evaluation ────────────────────────────────────────────────────────────
    sil = silhouette_score(X, final_labels)
    db = davies_bouldin_score(X, final_labels)
    inertia = final_km.inertia_

    print(f"\nSilhouette Score    : {sil:.4f}  (higher is better, max=1.0)")
    print(f"Davies-Bouldin Score: {db:.4f}  (lower is better)")
    print(f"Inertia             : {inertia:.1f}")

    # ── Cluster profiles & persona labels ─────────────────────────────────────
    df["cluster"] = final_labels
    profile = df.groupby("cluster")[FEATURES].mean()
    print("\n── Cluster Profiles ──")
    print(profile.round(2))

    persona_map = {}
    spend_med = profile["total_spend"].median()
    freq_med = profile["purchase_frequency"].median()

    for c in range(args.n_clusters):
        high_spend = profile.loc[c, "total_spend"] > spend_med
        high_freq = profile.loc[c, "purchase_frequency"] > freq_med
        if high_spend and high_freq:
            persona_map[c] = "Champion"
        elif high_spend:
            persona_map[c] = "High-Spender"
        elif high_freq:
            persona_map[c] = "Frequent-Buyer"
        else:
            persona_map[c] = "Occasional"

    print(f"\nPersona map: {persona_map}")

    # ── Optional cluster profile plot ─────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            colors = ["#2E86C1", "#1E8449", "#D35400", "#8E44AD"]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for ax, feat in zip(axes.flatten(), FEATURES):
                clusters = list(profile.index)
                vals = [profile.loc[c, feat] for c in clusters]
                bars = ax.bar([str(c) for c in clusters], vals, color=colors[:len(clusters)])
                ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
                ax.set_xlabel("Cluster")
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8,
                    )
            plt.suptitle("KMeans Cluster Profiles", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{MODEL_DIR}/cluster_profiles.png", dpi=150, bbox_inches="tight")
            print(f"\nPlot saved: {MODEL_DIR}/cluster_profiles.png")
            plt.show()
        except ImportError:
            print("matplotlib not installed, skipping plot.")

    # ── Save artifacts ────────────────────────────────────────────────────────
    metrics = {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_score": round(db, 4),
        "inertia": round(inertia, 1),
        "n_clusters": args.n_clusters,
        "persona_map": {str(k): v for k, v in persona_map.items()},
        "cluster_profiles": profile.round(3).to_dict(),
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    user_cluster_df = pd.DataFrame({"user_id": user_ids, "cluster": final_labels})
    user_cluster_df["persona"] = user_cluster_df["cluster"].map(persona_map)
    user_cluster_df.to_csv(f"{MODEL_DIR}/user_clusters.csv", index=False)

    with open(f"{MODEL_DIR}/kmeans_model.pkl", "wb") as f:
        pickle.dump(final_km, f)

    print(f"\nAll artifacts saved to {MODEL_DIR}/")
    print(f"  kmeans_model.pkl")
    print(f"  metrics.json")
    print(f"  user_clusters.csv")


if __name__ == "__main__":
    train()
