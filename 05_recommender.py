"""
05_recommender.py
Hybrid Recommendation Engine — local version.
Hybrid Score = 0.40*MBA + 0.25*ClusterPopularity + 0.20*Rating + 0.15*PriceMatch

Usage:
    python 05_recommender.py                    # demo all scenarios
    python 05_recommender.py --user U0001 --product P001 --top_n 5
    python 05_recommender.py --interactive      # interactive prompt loop
"""

import pandas as pd
import numpy as np
import json
import os
import argparse


class HybridRecommender:
    """
    Hybrid recommendation engine combining:
      - Market Basket Analysis (MBA) rules
      - KMeans cluster-based popularity
      - Product average rating
      - User price sensitivity match
    """

    def __init__(
        self,
        mba_rules_path="models/mba_rules.json",
        user_clusters_path="models/kmeans_artifacts/user_clusters.csv",
        user_features_path="data/processed/user_features.csv",
        products_path="data/processed/products_enriched.csv",
    ):
        with open(mba_rules_path) as f:
            self.mba_rules = json.load(f)
        self.user_clusters = pd.read_csv(user_clusters_path)
        self.user_features = pd.read_csv(user_features_path)
        self.products = pd.read_csv(products_path)
        self._build_cluster_popularity()
        print("HybridRecommender ready.")
        print(f"  MBA rules for {len(self.mba_rules)} products")
        print(f"  {len(self.user_clusters)} users with cluster assignments")
        print(f"  {len(self.products)} products in catalog")

    def _build_cluster_popularity(self):
        """Build top-rated products per cluster as fallback."""
        self.cluster_popular = {}
        top_products = self.products.nlargest(15, "avg_rating")["product_id"].tolist()
        for c in self.user_clusters["cluster"].unique():
            self.cluster_popular[int(c)] = top_products

    def _normalize(self, values):
        arr = np.array(values, dtype=float)
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    def recommend(self, user_id: str, viewed_product_id: str, top_n: int = 5) -> dict:
        """
        Generate hybrid recommendations.

        Args:
            user_id: e.g. 'U0001'
            viewed_product_id: e.g. 'P001'
            top_n: number of recommendations to return

        Returns:
            dict with user info and ranked recommendations list
        """
        user_row = self.user_features[self.user_features["user_id"] == user_id]
        if user_row.empty:
            # Cold-start: return globally popular products
            return self._cold_start_response(user_id, viewed_product_id, top_n)

        user_region = user_row.iloc[0]["region"]
        price_sens = float(user_row.iloc[0]["price_sensitivity"])

        cluster_row = self.user_clusters[self.user_clusters["user_id"] == user_id]
        user_cluster = int(cluster_row.iloc[0]["cluster"]) if not cluster_row.empty else 0
        user_persona = cluster_row.iloc[0]["persona"] if not cluster_row.empty else "Unknown"

        # Get MBA candidates or fall back to cluster-popular
        mba_candidates = self.mba_rules.get(viewed_product_id, [])
        if not mba_candidates:
            fallback_ids = self.cluster_popular.get(user_cluster, [])
            mba_candidates = [
                {"product": p, "score": 1.0, "confidence": 0.5, "lift": 1.0}
                for p in fallback_ids
            ]

        max_price = self.products["price"].max()
        avg_spend = float(user_row.iloc[0]["avg_order_value"])
        filtered = []

        for cand in mba_candidates:
            prod_row = self.products[self.products["product_id"] == cand["product"]]
            if prod_row.empty:
                continue
            prod = prod_row.iloc[0]
            # Filter out products far too expensive for user
            if prod["price"] > avg_spend * 3:
                continue
            norm_price = prod["price"] / max_price
            price_match = 1 - abs(price_sens - norm_price)
            filtered.append({
                **cand,
                "product_name": prod["name"],
                "category": prod["category"],
                "price": prod["price"],
                "avg_rating": prod["avg_rating"],
                "price_match": price_match,
            })

        if not filtered:
            return self._cold_start_response(user_id, viewed_product_id, top_n)

        # Cluster popularity flag
        cluster_pop_list = self.cluster_popular.get(user_cluster, [])
        for c in filtered:
            c["cluster_pop"] = 1.0 if c["product"] in cluster_pop_list else 0.5

        # Normalize signal arrays
        mba_norm = self._normalize([c["score"] for c in filtered])
        rating_norm = self._normalize([c["avg_rating"] for c in filtered])
        price_arr = np.array([c["price_match"] for c in filtered])
        cluster_arr = np.array([c["cluster_pop"] for c in filtered])

        # Hybrid scoring formula
        final_scores = 0.40 * mba_norm + 0.25 * cluster_arr + 0.20 * rating_norm + 0.15 * price_arr

        for i, c in enumerate(filtered):
            c["final_score"] = round(float(final_scores[i]), 4)

        results = sorted(filtered, key=lambda x: x["final_score"], reverse=True)

        return {
            "user_id": user_id,
            "persona": user_persona,
            "region": user_region,
            "cluster": user_cluster,
            "viewed_product": viewed_product_id,
            "recommendations": [
                {
                    "rank": i + 1,
                    "product_id": r["product"],
                    "product_name": r["product_name"],
                    "category": r["category"],
                    "price": r["price"],
                    "avg_rating": r["avg_rating"],
                    "final_score": r["final_score"],
                    "score_breakdown": {
                        "mba": round(float(mba_norm[i]), 4),
                        "cluster_pop": r["cluster_pop"],
                        "rating": round(float(rating_norm[i]), 4),
                        "price_match": round(r["price_match"], 4),
                    },
                }
                for i, r in enumerate(results[:top_n])
            ],
        }

    def _cold_start_response(self, user_id, viewed_product_id, top_n):
        """Return globally popular products for unknown users."""
        top_prods = self.products.nlargest(top_n, "avg_rating")[
            ["product_id", "name", "price", "avg_rating"]
        ].to_dict("records")
        return {
            "user_id": user_id,
            "persona": "Unknown (cold-start)",
            "region": "N/A",
            "cluster": -1,
            "viewed_product": viewed_product_id,
            "note": "User not found — returning globally popular products",
            "recommendations": [
                {
                    "rank": i + 1,
                    "product_id": r["product_id"],
                    "product_name": r["name"],
                    "category": "N/A",
                    "price": r["price"],
                    "avg_rating": r["avg_rating"],
                    "final_score": None,
                    "score_breakdown": None,
                }
                for i, r in enumerate(top_prods)
            ],
        }

    def print_result(self, result: dict):
        """Pretty-print recommendation result."""
        print(f"\n{'='*65}")
        print(
            f"User: {result['user_id']}  |  Persona: {result['persona']}  "
            f"|  Cluster: {result['cluster']}"
        )
        print(f"Region: {result['region']}  |  Viewed Product: {result['viewed_product']}")
        if "note" in result:
            print(f"Note: {result['note']}")
        print(f"\nTop {len(result['recommendations'])} Recommendations:")
        print(f"{'─'*65}")
        for r in result["recommendations"]:
            print(
                f"  {r['rank']}. {r['product_name']:30s} | "
                f"${r['price']:7.2f} | ⭐{r['avg_rating']} | "
                f"Score: {r['final_score']}"
            )
            if r["score_breakdown"]:
                s = r["score_breakdown"]
                print(
                    f"     MBA={s['mba']:.2f}  Cluster={s['cluster_pop']:.2f}  "
                    f"Rating={s['rating']:.2f}  Price={s['price_match']:.2f}"
                )
        print(f"{'='*65}")


def run_demo(rec):
    """Run 3 built-in demo scenarios."""
    print("\n" + "=" * 65)
    print("SCENARIO 1: Champion User viewing Electronics (P001)")
    print("=" * 65)
    result = rec.recommend(user_id="U0001", viewed_product_id="P001", top_n=5)
    rec.print_result(result)

    print("\n" + "=" * 65)
    print("SCENARIO 2: Budget User viewing Books area (P010)")
    print("=" * 65)
    result2 = rec.recommend(user_id="U0050", viewed_product_id="P010", top_n=5)
    rec.print_result(result2)

    print("\n" + "=" * 65)
    print("SCENARIO 3: Cold-Start — Unknown User")
    print("=" * 65)
    result3 = rec.recommend(user_id="U9999", viewed_product_id="P020", top_n=5)
    rec.print_result(result3)


def interactive_mode(rec):
    """Loop for manual testing."""
    print("\n=== Interactive Mode (type 'quit' to exit) ===")
    while True:
        user_id = input("\nEnter user_id (e.g. U0001): ").strip()
        if user_id.lower() == "quit":
            break
        product_id = input("Enter product_id (e.g. P001): ").strip()
        if product_id.lower() == "quit":
            break
        try:
            top_n = int(input("How many recommendations? [5]: ").strip() or "5")
        except ValueError:
            top_n = 5
        result = rec.recommend(user_id=user_id, viewed_product_id=product_id, top_n=top_n)
        rec.print_result(result)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Recommendation Engine")
    parser.add_argument("--user", type=str, help="User ID, e.g. U0001")
    parser.add_argument("--product", type=str, help="Viewed product ID, e.g. P001")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    rec = HybridRecommender()

    if args.interactive:
        interactive_mode(rec)
    elif args.user and args.product:
        result = rec.recommend(args.user, args.product, args.top_n)
        rec.print_result(result)
    else:
        run_demo(rec)


if __name__ == "__main__":
    main()
