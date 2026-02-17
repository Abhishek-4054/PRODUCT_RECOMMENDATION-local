"""
01_generate_data.py
Generates synthetic data: products, users, and transactions.
Run this first before any other script.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

os.makedirs("data/raw", exist_ok=True)

# ── PRODUCTS ──────────────────────────────────────────────────────────────────
categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
price_ranges = {
    "Electronics": (50, 800),
    "Clothing": (15, 150),
    "Books": (8, 60),
    "Home": (20, 300),
    "Sports": (25, 400),
}

products = []
for i in range(1, 51):
    cat = random.choice(categories)
    lo, hi = price_ranges[cat]
    products.append({
        "product_id": f"P{i:03d}",
        "name": f"{cat}_Item_{i}",
        "category": cat,
        "price": round(random.uniform(lo, hi), 2),
        "avg_rating": round(random.uniform(3.0, 5.0), 1),
    })
products_df = pd.DataFrame(products)
print(f"Products: {products_df.shape}")

# ── USERS ─────────────────────────────────────────────────────────────────────
regions = ["North", "South", "East", "West"]
users = []
for i in range(1, 1001):
    persona = random.choice(["budget", "regular", "premium"])
    spend_map = {"budget": (15, 80), "regular": (60, 250), "premium": (200, 1200)}
    order_map = {"budget": (1, 5), "regular": (5, 20), "premium": (15, 50)}
    lo, hi = spend_map[persona]
    ol, oh = order_map[persona]
    users.append({
        "user_id": f"U{i:04d}",
        "age": random.randint(18, 65),
        "avg_order_value": round(random.uniform(lo, hi), 2),
        "total_orders": random.randint(ol, oh),
        "region": random.choice(regions),
        "preferred_category": random.choice(categories),
        "persona": persona,
    })
users_df = pd.DataFrame(users)
print(f"Users: {users_df.shape}")

# ── TRANSACTIONS ──────────────────────────────────────────────────────────────
COMBOS = [
    ["P001", "P002", "P003"],
    ["P010", "P011"],
    ["P020", "P021", "P022"],
    ["P030", "P031"],
    ["P040", "P041", "P042"],
]

transactions = []
txn_id = 1
start_date = datetime(2024, 1, 1)

for _ in range(10000):
    user = users_df.sample(1).iloc[0]
    txn_date = start_date + timedelta(days=random.randint(0, 364))
    if random.random() < 0.40:
        product_id = random.choice(random.choice(COMBOS))
    else:
        product_id = random.choice(products_df["product_id"].tolist())
    prod = products_df[products_df["product_id"] == product_id].iloc[0]
    transactions.append({
        "transaction_id": f"T{txn_id:06d}",
        "user_id": user["user_id"],
        "product_id": product_id,
        "quantity": random.randint(1, 5),
        "price": prod["price"],
        "region": user["region"],
        "timestamp": txn_date.strftime("%Y-%m-%d"),
    })
    txn_id += 1

transactions_df = pd.DataFrame(transactions)
print(f"Transactions: {transactions_df.shape}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
products_df.to_csv("data/raw/products.csv", index=False)
users_df.to_csv("data/raw/users.csv", index=False)
transactions_df.to_csv("data/raw/transactions.csv", index=False)
print("\nSaved to data/raw/")
print("  data/raw/products.csv")
print("  data/raw/users.csv")
print("  data/raw/transactions.csv")
