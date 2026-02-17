"""
03_market_basket.py
Market Basket Analysis using FP-Growth algorithm.
Reads data/processed/basket_matrix.csv
Writes models/mba_rules.json and models/association_rules.csv
"""

import pandas as pd
import json
import os
from mlxtend.frequent_patterns import fpgrowth, association_rules

os.makedirs("models", exist_ok=True)


def run():
    # ── Load basket matrix ─────────────────────────────────────────────────────
    basket_df = pd.read_csv("data/processed/basket_matrix.csv")
    print(f"Basket matrix: {basket_df.shape[0]} transactions x {basket_df.shape[1]} products")

    # ── Run FP-Growth ──────────────────────────────────────────────────────────
    frequent_itemsets = fpgrowth(basket_df, min_support=0.02, use_colnames=True)
    print(f"\nFrequent itemsets found: {len(frequent_itemsets)}")
    print("\nTop 10 by support:")
    print(frequent_itemsets.sort_values("support", ascending=False).head(10).to_string())

    # ── Generate Association Rules ─────────────────────────────────────────────
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules[rules["confidence"] >= 0.30].sort_values("lift", ascending=False)
    print(f"\nRules generated: {len(rules)}")
    print("\nTop 15 rules:")
    print(
        rules[["antecedents", "consequents", "support", "confidence", "lift"]]
        .head(15)
        .to_string()
    )

    # ── Build recommendation dictionary ───────────────────────────────────────
    rec_dict = {}
    for _, row in rules.iterrows():
        for ant in row["antecedents"]:
            for con in row["consequents"]:
                score = float(row["lift"]) * float(row["confidence"])
                if ant not in rec_dict:
                    rec_dict[ant] = []
                rec_dict[ant].append({
                    "product": con,
                    "score": round(score, 4),
                    "confidence": round(float(row["confidence"]), 4),
                    "lift": round(float(row["lift"]), 4),
                })

    for prod in rec_dict:
        rec_dict[prod] = sorted(rec_dict[prod], key=lambda x: x["score"], reverse=True)

    print(f"\nProducts with MBA recommendations: {len(rec_dict)}")

    # Show an example
    if rec_dict:
        example_prod = list(rec_dict.keys())[0]
        print(f"\nSample recs for {example_prod}:")
        for r in rec_dict[example_prod][:5]:
            print(f"  -> {r['product']} (lift={r['lift']}, conf={r['confidence']})")

    # ── Save ───────────────────────────────────────────────────────────────────
    with open("models/mba_rules.json", "w") as f:
        json.dump(rec_dict, f, indent=2)

    rules.to_csv("models/association_rules.csv", index=False)
    print("\nSaved: models/mba_rules.json")
    print("Saved: models/association_rules.csv")


if __name__ == "__main__":
    run()
