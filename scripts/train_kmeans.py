"""
scripts/train_kmeans.py
Runs as a SageMaker Training Job.
SageMaker automatically mounts:
  /opt/ml/input/data/train  — your training data from S3
  /opt/ml/model             — where you save model artifacts
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import argparse, os, pickle, json


FEATURES = [
    'total_spend', 'purchase_frequency', 'avg_basket_value',
    'unique_products', 'category_diversity', 'price_sensitivity'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters',    type=int, default=4)
    parser.add_argument('--random_state',  type=int, default=42)
    parser.add_argument('--max_iter',      type=int, default=300)
    # SageMaker injects these environment variables automatically
    parser.add_argument('--model-dir',  type=str, default=os.environ.get('SM_MODEL_DIR',  '/opt/ml/model'))
    parser.add_argument('--input-dir',  type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    return parser.parse_args()


def train():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    df       = pd.read_csv(f'{args.input_dir}/kmeans_input.csv')
    user_ids = df['user_id'].values
    X        = df[FEATURES].values

    print(f'Training data shape: {X.shape}')

    # ── Elbow method — find optimal K ─────────────────────────────────────────
    print('\n── Elbow Method ──')
    inertias   = []
    sil_scores = []
    for k in range(2, 8):
        km     = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        print(f'  K={k} | Inertia={km.inertia_:.1f} | Silhouette={silhouette_score(X, labels):.4f}')

    # ── Train final model ──────────────────────────────────────────────────────
    print(f'\n── Training final model with K={args.n_clusters} ──')
    final_km     = KMeans(n_clusters=args.n_clusters, random_state=args.random_state,
                          max_iter=args.max_iter, n_init=10)
    final_labels = final_km.fit_predict(X)

    # ── Evaluation metrics ─────────────────────────────────────────────────────
    sil     = silhouette_score(X, final_labels)
    db      = davies_bouldin_score(X, final_labels)
    inertia = final_km.inertia_

    print(f'\nSilhouette Score    : {sil:.4f}  (higher is better, max=1.0)')
    print(f'Davies-Bouldin Score: {db:.4f}  (lower is better)')
    print(f'Inertia             : {inertia:.1f}')

    # ── Cluster profiles & persona labels ─────────────────────────────────────
    df['cluster'] = final_labels
    profile = df.groupby('cluster')[FEATURES].mean()
    print('\n── Cluster Profiles ──')
    print(profile.round(2))

    persona_map = {}
    spend_med = profile['total_spend'].median()
    freq_med  = profile['purchase_frequency'].median()

    for c in range(args.n_clusters):
        high_spend = profile.loc[c, 'total_spend']       > spend_med
        high_freq  = profile.loc[c, 'purchase_frequency'] > freq_med
        if high_spend and high_freq:
            persona_map[c] = 'Champion'
        elif high_spend:
            persona_map[c] = 'High-Spender'
        elif high_freq:
            persona_map[c] = 'Frequent-Buyer'
        else:
            persona_map[c] = 'Occasional'

    print(f'\nPersona map: {persona_map}')

    # ── Save artifacts ─────────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Metrics JSON
    metrics = {
        'silhouette_score':      round(sil,     4),
        'davies_bouldin_score':  round(db,      4),
        'inertia':               round(inertia, 1),
        'n_clusters':            args.n_clusters,
        'persona_map':           {str(k): v for k, v in persona_map.items()},
        'cluster_profiles':      profile.round(3).to_dict()
    }
    with open(f'{args.model_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # 2. User → cluster mapping
    user_cluster_df = pd.DataFrame({'user_id': user_ids, 'cluster': final_labels})
    user_cluster_df['persona'] = user_cluster_df['cluster'].map(persona_map)
    user_cluster_df.to_csv(f'{args.model_dir}/user_clusters.csv', index=False)

    # 3. Trained KMeans model
    with open(f'{args.model_dir}/kmeans_model.pkl', 'wb') as f:
        pickle.dump(final_km, f)

    print(f'\nAll artifacts saved to {args.model_dir}')


if __name__ == '__main__':
    train()
