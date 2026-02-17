"""
02_preprocessing.py
Feature engineering: reads data/raw/, writes data/processed/ and models/scaler.pkl
No AWS required — runs 100% locally.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from mlxtend.preprocessing import TransactionEncoder
import os
import pickle

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


def run():
    # ── Load raw data ──────────────────────────────────────────────────────────
    txn = pd.read_csv(f"{INPUT_DIR}/transactions.csv")
    users = pd.read_csv(f"{INPUT_DIR}/users.csv")
    prods = pd.read_csv(f"{INPUT_DIR}/products.csv")
    print(f"Loaded: {len(txn)} transactions | {len(users)} users | {len(prods)} products")

    # ── 1. User-level aggregations for KMeans ──────────────────────────────────
    txn["total_price"] = txn["price"] * txn["quantity"]

    user_stats = txn.groupby("user_id").agg(
        total_spend=("total_price", "sum"),
        purchase_frequency=("transaction_id", "count"),
        avg_basket_value=("total_price", "mean"),
        unique_products=("product_id", "nunique"),
    ).reset_index()

    # Category diversity per user
    txn_prods = txn.merge(prods[["product_id", "category"]], on="product_id")
    cat_div = txn_prods.groupby("user_id")["category"].nunique().reset_index()
    cat_div.columns = ["user_id", "category_diversity"]

    user_features = users.merge(user_stats, on="user_id", how="left")
    user_features = user_features.merge(cat_div, on="user_id", how="left")
    user_features.fillna(0, inplace=True)

    # Price sensitivity
    max_price = prods["price"].max()
    user_features["price_sensitivity"] = (
        user_features["avg_order_value"] / max_price
    ).clip(0, 1)

    # ── 2. Basket matrix for Market Basket Analysis ────────────────────────────
    baskets = txn.groupby("transaction_id")["product_id"].apply(list).reset_index()
    te = TransactionEncoder()
    te_array = te.fit_transform(baskets["product_id"].tolist())
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    # ── 3. Scale KMeans features ───────────────────────────────────────────────
    KMEANS_FEATURES = [
        "total_spend", "purchase_frequency", "avg_basket_value",
        "unique_products", "category_diversity", "price_sensitivity",
    ]
    X = user_features[KMEANS_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=KMEANS_FEATURES)
    scaled_df["user_id"] = user_features["user_id"].values

    # ── Save outputs ───────────────────────────────────────────────────────────
    user_features.to_csv(f"{OUTPUT_DIR}/user_features.csv", index=False)
    scaled_df.to_csv(f"{OUTPUT_DIR}/kmeans_input.csv", index=False)
    basket_df.to_csv(f"{OUTPUT_DIR}/basket_matrix.csv", index=False)
    prods.to_csv(f"{OUTPUT_DIR}/products_enriched.csv", index=False)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nFeature engineering complete.")
    print(f"  user_features  : {user_features.shape}")
    print(f"  basket_matrix  : {basket_df.shape}")
    print(f"  kmeans_input   : {scaled_df.shape}")
    print("\nOutputs saved to data/processed/ and models/scaler.pkl")


if __name__ == "__main__":
    run()
