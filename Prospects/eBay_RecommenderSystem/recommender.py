import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, ndcg_score
import time
import random

# ========================
# 1. System Architecture
# ========================

class ProductionRecommender:
    def __init__(self):
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.feature_store = {}
        self.ranking_model = None
        self.popular_items = []
        self.latency_slo = 200  # ms

    # ========================
    # 2. Offline Training
    # ========================
    def train_offline(self, user_data, item_data, interactions):
        """Train embeddings and ranking model offline"""
        # Generate user embeddings (simulated)
        self.user_embeddings = {user: np.random.rand(32) for user in user_data['user_id']}

        # Generate item embeddings (simulated)
        self.item_embeddings = {item: np.random.rand(32) for item in item_data['item_id']}

        # Train ranking model (simulated)
        X = np.random.rand(1000, 10)  # Simulated features
        y = np.random.randint(0, 2, 1000)  # Simulated labels
        self.ranking_model = RandomForestClassifier()
        self.ranking_model.fit(X, y)

        # Identify popular items for cold start
        self.popular_items = interactions['item_id'].value_counts().index.tolist()[:100]

        print("Offline training complete")

    # ========================
    # 3. Online Retrieval (ANN)
    # ========================
    def retrieve_candidates(self, user_id, context, k=100):
        """Retrieve candidates using approximate nearest neighbors"""
        start_time = time.time()

        # Cold start handling
        if user_id not in self.user_embeddings:
            return self.handle_cold_start(context)

        # ANN search (simulated)
        user_embedding = self.user_embeddings[user_id]
        all_items = list(self.item_embeddings.keys())
        item_embeddings = np.array([self.item_embeddings[i] for i in all_items])

        # Simulated ANN with brute-force for demo (use FAISS/HNSW in production)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(item_embeddings)
        distances, indices = nn.kneighbors([user_embedding])

        candidates = [all_items[i] for i in indices[0]]

        # Latency control
        elapsed = (time.time() - start_time) * 1000
        if elapsed > self.latency_slo * 0.3:  # Use 30% of budget for retrieval
            print(f"ANN latency warning: {elapsed:.2f}ms")

        return candidates

    # ========================
    # 4. Cold Start Handling
    # ========================
    def handle_cold_start(self, context):
        """Strategies for new users/items"""
        # Popular items bootstrap
        if 'category' in context:
            # Semantic bootstrap using context
            return [i for i in self.popular_items if i.startswith(context['category'])][:100]
        return self.popular_items[:100]

    # ========================
    # 5. Feature Assembly
    # ========================
    def get_features(self, user_id, candidates):
        """Fetch features from feature store (simulated)"""
        start_time = time.time()
        features = []

        for item in candidates:
            # Simulated feature assembly
            user_feat = self.user_embeddings.get(user_id, np.zeros(32))
            item_feat = self.item_embeddings.get(item, np.zeros(32))
            context_feat = np.random.rand(10)  # Simulated context features
            features.append(np.concatenate([user_feat, item_feat, context_feat]))

        # Latency simulation
        time.sleep(0.02 * len(candidates)/100)  # 2ms per item

        elapsed = (time.time() - start_time) * 1000
        if elapsed > self.latency_slo * 0.4:  # Use 40% of budget for features
            print(f"Feature latency warning: {elapsed:.2f}ms")

        return np.array(features)

    # ========================
    # 6. Ranking & Re-ranking
    # ========================
    def rank_items(self, features, candidates):
        """Rank candidates using ML model"""
        start_time = time.time()

        # Model prediction
        scores = self.ranking_model.predict_proba(features)[:, 1]
        ranked_indices = np.argsort(scores)[::-1]
        ranked_candidates = [candidates[i] for i in ranked_indices]

        # Apply business rules (diversity)
        final_results = self.apply_diversity(ranked_candidates)

        elapsed = (time.time() - start_time) * 1000
        if elapsed > self.latency_slo * 0.3:  # Use 30% of budget for ranking
            print(f"Ranking latency warning: {elapsed:.2f}ms")

        return final_results

    def apply_diversity(self, items, max_per_category=3):
        """Re-ranking with diversity constraints"""
        diversified = []
        category_counts = {}

        for item in items:
            # Simulated category extraction
            category = item.split('_')[0] if '_' in item else 'default'

            if category not in category_counts:
                category_counts[category] = 0

            if category_counts[category] < max_per_category:
                diversified.append(item)
                category_counts[category] += 1

            if len(diversified) >= 20:  # Final recommendation size
                break

        return diversified

    # ========================
    # 7. End-to-End Service
    # ========================
    def recommend(self, user_id, context):
        """End-to-end recommendation workflow"""
        start_time = time.time()

        try:
            # Candidate retrieval
            candidates = self.retrieve_candidates(user_id, context, k=100)

            # Feature assembly
            features = self.get_features(user_id, candidates)

            # Ranking and re-ranking
            recommendations = self.rank_items(features, candidates)

            # Real-time signal incorporation
            if 'recent_view' in context:
                recommendations = self.boost_recent(recommendations, context['recent_view'])

        except Exception as e:
            print(f"Error in recommendation: {e}")
            recommendations = self.fallback_recommendation(context)

        # Latency check
        elapsed = (time.time() - start_time) * 1000
        print(f"Total latency: {elapsed:.2f}ms {'(SLO Exceeded!)' if elapsed > self.latency_slo else ''}")

        return recommendations[:10]  # Return top 10

    def boost_recent(self, items, recent_view):
        """Boost recently viewed items in recommendations"""
        return [item for item in items if item not in recent_view][:15] + recent_view[:5]

    # ========================
    # 8. Reliability & Fallbacks
    # ========================
    def fallback_recommendation(self, context):
        """Fallback strategy for system failures"""
        if 'category' in context:
            return [item for item in self.popular_items if item.startswith(context['category'])][:10]
        return self.popular_items[:10]

    # ========================
    # 9. Evaluation
    # ========================
    def evaluate_offline(self, test_data):
        """Offline evaluation metrics"""
        # Simulated evaluation
        y_true = test_data['label']
        y_pred = np.random.rand(len(y_true))  # Simulated predictions

        auc = roc_auc_score(y_true, y_pred)
        ndcg = ndcg_score([y_true], [y_pred])

        print(f"Offline Metrics - AUC: {auc:.4f}, NDCG: {ndcg:.4f}")
        return auc, ndcg

    def a_b_test(self, user_group):
        """Simulate online A/B testing"""
        # Simulated business metrics
        ctr = random.uniform(0.05, 0.15)
        cvr = random.uniform(0.01, 0.05)
        gmv = random.randint(1000, 5000)

        print(f"A/B Test Results - CTR: {ctr:.4f}, CVR: {cvr:.4f}, GMV: ${gmv}")
        return ctr, cvr, gmv

# ========================
# 10. Simulation
# ========================
# ========================
# 10. Simulation
# ========================
if __name__ == "__main__":
    # Initialize system
    recommender = ProductionRecommender()

    # Generate simulated data
    users = pd.DataFrame({'user_id': [f'user_{i}' for i in range(1000)]})
    items = pd.DataFrame({'item_id': [f'cat_{i//100}_item_{i%100}' for i in range(1000)]})
    interactions = pd.DataFrame({
        'user_id': np.random.choice(users['user_id'], 5000),
        'item_id': np.random.choice(items['item_id'], 5000)
    })

    # Offline training
    recommender.train_offline(users, items, interactions)

    # Offline evaluation
    test_data = pd.DataFrame({
        'user_id': np.random.choice(users['user_id'], 1000),
        'item_id': np.random.choice(items['item_id'], 1000),
        'label': np.random.randint(0, 2, 1000)
    })
    recommender.evaluate_offline(test_data)

    # Online recommendation
    context = {
        'category': 'cat_5',
        'recent_view': ['cat_5_item_123', 'cat_5_item_456']
    }
    print("\nRecommendations for known user:")
    print(recommender.recommend('user_123', context))

    print("\nRecommendations for new user (cold start):")
    print(recommender.recommend('new_user', context))

    # A/B testing
    print("\nA/B Testing Results:")
    recommender.a_b_test("treatment_group")
