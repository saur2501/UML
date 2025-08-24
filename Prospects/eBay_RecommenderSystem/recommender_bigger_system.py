"""
Production Recommender System Demonstration
A comprehensive program demonstrating production-grade recommender system design
with all the qualities specified for real-world deployment.
"""

import numpy as np
import pandas as pd
import time
import random
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta

# Mock external dependencies
class MockFeatureStore:
    """Mock feature store for production simulation"""
    def __init__(self):
        self.features = {}
        self.latency_ms = 5  # Simulated 5ms latency

    def get_user_features(self, user_id: str) -> Dict[str, float]:
        time.sleep(self.latency_ms / 1000)
        return {
            'user_age': random.randint(18, 65),
            'user_tenure_days': random.randint(1, 1000),
            'purchase_frequency': random.random(),
            'avg_order_value': random.uniform(10, 200),
            'last_activity_days': random.randint(0, 30)
        }

    def get_item_features(self, item_id: str) -> Dict[str, float]:
        time.sleep(self.latency_ms / 1000)
        return {
            'item_price': random.uniform(5, 500),
            'item_category': hash(item_id.split('_')[0]) % 10,
            'item_popularity': random.random(),
            'item_rating': random.uniform(1, 5),
            'inventory_level': random.randint(0, 1000)
        }

class MockANNIndex:
    """Mock Approximate Nearest Neighbor index"""
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.item_embeddings = {}
        self.latency_ms = 15  # Simulated 15ms latency

    def add_items(self, items: List[str], embeddings: np.ndarray):
        for item, embedding in zip(items, embeddings):
            self.item_embeddings[item] = embedding

    def query(self, query_embedding: np.ndarray, k: int = 100) -> List[Tuple[str, float]]:
        time.sleep(self.latency_ms / 1000)

        # Simulate ANN search with some randomness
        items = list(self.item_embeddings.keys())
        random.shuffle(items)

        # Calculate distances (simplified)
        results = []
        for item in items[:k*2]:  # Consider more items for simulation
            item_embedding = self.item_embeddings[item]
            distance = np.linalg.norm(query_embedding - item_embedding)
            results.append((item, distance))

        # Sort by distance and return top k
        results.sort(key=lambda x: x[1])
        return results[:k]

class MockRedis:
    """Mock Redis for real-time features"""
    def __init__(self):
        self.data = {}
        self.latency_ms = 2  # Simulated 2ms latency

    def get(self, key: str) -> Optional[Any]:
        time.sleep(self.latency_ms / 1000)
        return self.data.get(key)

    def set(self, key: str, value: Any, ttl: int = 3600):
        time.sleep(self.latency_ms / 1000)
        self.data[key] = value

@dataclass
class RecommendationRequest:
    """Structured request for recommendations"""
    user_id: str
    context: Dict[str, Any]
    session_id: str
    timestamp: datetime
    request_id: str

@dataclass
class RecommendationResponse:
    """Structured response from recommender"""
    recommendations: List[str]
    scores: List[float]
    latency_ms: float
    fallback_used: bool
    error_message: Optional[str] = None

class MetricsCollector:
    """Collect and track system metrics"""
    def __init__(self):
        self.latencies = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self.business_metrics = defaultdict(list)

    def record_latency(self, latency_ms: float, endpoint: str):
        self.latencies.append((datetime.now(), latency_ms, endpoint))

    def record_error(self, error_type: str, endpoint: str):
        self.error_counts[(error_type, endpoint)] += 1

    def record_request(self, endpoint: str):
        self.request_counts[endpoint] += 1

    def record_business_metric(self, metric_name: str, value: float):
        self.business_metrics[metric_name].append((datetime.now(), value))

    def get_p99_latency(self, endpoint: str = None) -> float:
        relevant_latencies = [lat for _, lat, ep in self.latencies
                             if endpoint is None or ep == endpoint]
        if not relevant_latencies:
            return 0
        return np.percentile(relevant_latencies, 99)

    def get_error_rate(self, endpoint: str = None) -> float:
        total_requests = sum(count for ep, count in self.request_counts.items()
                           if endpoint is None or ep == endpoint)
        total_errors = sum(count for (err, ep), count in self.error_counts.items()
                         if endpoint is None or ep == endpoint)
        return total_errors / total_requests if total_requests > 0 else 0

class EmbeddingGenerator:
    """Generate embeddings for users and items"""
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def generate_user_embedding(self, user_features: Dict[str, float]) -> np.ndarray:
        """Generate user embedding from features"""
        # Simulate neural network embedding generation
        base_embedding = np.random.randn(self.embedding_dim) * 0.1

        # Incorporate features
        feature_vector = np.array(list(user_features.values()))
        feature_embedding = np.random.randn(self.embedding_dim, len(feature_vector)) @ feature_vector

        return base_embedding + 0.1 * feature_embedding

    def generate_item_embedding(self, item_features: Dict[str, float]) -> np.ndarray:
        """Generate item embedding from features"""
        # Simulate content-based embedding
        base_embedding = np.random.randn(self.embedding_dim) * 0.1

        # Incorporate features
        feature_vector = np.array(list(item_features.values()))
        feature_embedding = np.random.randn(self.embedding_dim, len(feature_vector)) @ feature_vector

        return base_embedding + 0.1 * feature_embedding

class RankingModel:
    """Machine learning model for ranking candidates"""
    def __init__(self):
        self.model_weights = np.random.randn(50)  # Simulated model weights
        self.feature_names = [f'feature_{i}' for i in range(50)]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability of interaction"""
        # Simulate model prediction
        scores = features @ self.model_weights[:features.shape[1]]
        probabilities = 1 / (1 + np.exp(-scores))  # Sigmoid
        return probabilities

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary outcome"""
        probabilities = self.predict_proba(features)
        return (probabilities > 0.5).astype(int)

class ColdStartHandler:
    """Handle cold start scenarios for new users/items"""
    def __init__(self, popular_items: List[str]):
        self.popular_items = popular_items
        self.category_popularity = defaultdict(list)

        # Build category-based popularity
        for item in popular_items:
            category = item.split('_')[0] if '_' in item else 'default'
            self.category_popularity[category].append(item)

    def get_bootstrap_candidates(self, context: Dict[str, Any], k: int = 100) -> List[str]:
        """Get candidates for cold start scenarios"""
        candidates = []

        # Strategy 1: Popular items
        candidates.extend(self.popular_items[:k//3])

        # Strategy 2: Category-based popularity
        if 'category' in context:
            category_items = self.category_popularity.get(context['category'], [])
            candidates.extend(category_items[:k//3])

        # Strategy 3: Semantic similarity (simulated)
        if 'query' in context:
            semantic_items = self._semantic_search(context['query'], k//3)
            candidates.extend(semantic_items)

        # Strategy 4: Neighborhood-based (simulated)
        if 'location' in context:
            neighborhood_items = self._neighborhood_items(context['location'], k//3)
            candidates.extend(neighborhood_items)

        return list(set(candidates))[:k]  # Remove duplicates and limit

    def _semantic_search(self, query: str, k: int) -> List[str]:
        """Simulate semantic search based on query"""
        # Simple keyword matching simulation
        relevant_items = [item for item in self.popular_items
                         if any(keyword in item.lower() for keyword in query.lower().split())]
        return relevant_items[:k]

    def _neighborhood_items(self, location: str, k: int) -> List[str]:
        """Simulate location-based recommendations"""
        # Random selection based on location (simplified)
        location_hash = hash(location) % len(self.popular_items)
        return self.popular_items[location_hash:location_hash + k]

class DiversityFilter:
    """Apply diversity constraints to recommendations"""
    def __init__(self, max_per_category: int = 3, min_categories: int = 5):
        self.max_per_category = max_per_category
        self.min_categories = min_categories

    def apply_diversity(self, items_with_scores: List[Tuple[str, float]]) -> List[str]:
        """Apply diversity rules to ranked items"""
        diversified = []
        category_counts = defaultdict(int)
        categories_represented = set()

        for item, score in items_with_scores:
            category = item.split('_')[0] if '_' in item else 'default'

            # Check diversity constraints
            if (category_counts[category] < self.max_per_category or
                len(categories_represented) < self.min_categories):

                diversified.append(item)
                category_counts[category] += 1
                categories_represented.add(category)

                if len(diversified) >= 20:  # Final recommendation size
                    break

        return diversified

class BusinessRulesEngine:
    """Apply business rules to recommendations"""
    def __init__(self):
        self.rules = {
            'exclude_out_of_stock': True,
            'boost_high_margin': True,
            'demote_competitors': True,
            'seasonal_boost': True
        }

    def apply_rules(self, items_with_scores: List[Tuple[str, float]],
                   context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Apply business rules to adjust scores"""
        adjusted_items = []

        for item, score in items_with_scores:
            adjusted_score = score

            # Rule 1: Exclude out of stock items
            if self.rules['exclude_out_of_stock'] and random.random() < 0.05:  # 5% out of stock
                continue

            # Rule 2: Boost high margin items
            if self.rules['boost_high_margin'] and random.random() < 0.3:  # 30% high margin
                adjusted_score *= 1.2

            # Rule 3: Demote competitor items
            if self.rules['demote_competitors'] and 'competitor' in item:
                adjusted_score *= 0.5

            # Rule 4: Seasonal boost
            if self.rules['seasonal_boost'] and context.get('season') == 'holiday':
                adjusted_score *= 1.1

            adjusted_items.append((item, adjusted_score))

        return adjusted_items

class ProductionRecommenderSystem:
    """
    Production-grade recommender system with all specified qualities
    """

    def __init__(self, latency_slo_ms: int = 200):
        # Core components
        self.feature_store = MockFeatureStore()
        self.ann_index = MockANNIndex()
        self.redis = MockRedis()
        self.embedding_generator = EmbeddingGenerator()
        self.ranking_model = RankingModel()
        self.cold_start_handler = None
        self.diversity_filter = DiversityFilter()
        self.business_rules = BusinessRulesEngine()
        self.metrics = MetricsCollector()

        # System configuration
        self.latency_slo_ms = latency_slo_ms
        self.max_candidates = 100
        self.final_recommendations = 20

        # Real-time signals
        self.user_sessions = defaultdict(lambda: {'recent_views': deque(maxlen=10),
                                               'recent_purchases': deque(maxlen=5)})

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the system with training data"""
        # Generate sample items and embeddings
        items = [f'cat_{i%10}_item_{j}' for i in range(10) for j in range(100)]
        item_embeddings = np.random.randn(len(items), 128)

        self.ann_index.add_items(items, item_embeddings)

        # Initialize cold start handler
        popular_items = items[:200]  # Top 200 popular items
        self.cold_start_handler = ColdStartHandler(popular_items)

        self.logger.info("Production recommender system initialized")

    def train_offline(self, user_data: pd.DataFrame, item_data: pd.DataFrame,
                     interactions: pd.DataFrame):
        """
        Offline training phase
        Generate embeddings and train ranking models
        """
        start_time = time.time()

        # Generate user embeddings
        self.logger.info("Generating user embeddings...")
        for _, user_row in user_data.iterrows():
            user_id = user_row['user_id']
            user_features = self.feature_store.get_user_features(user_id)
            user_embedding = self.embedding_generator.generate_user_embedding(user_features)
            self.redis.set(f"user_embedding:{user_id}", user_embedding.tolist())

        # Generate item embeddings
        self.logger.info("Generating item embeddings...")
        item_embeddings_list = []
        for _, item_row in item_data.iterrows():
            item_id = item_row['item_id']
            item_features = self.feature_store.get_item_features(item_id)
            item_embedding = self.embedding_generator.generate_item_embedding(item_features)
            item_embeddings_list.append(item_embedding)

        # Update ANN index
        self.ann_index.add_items(item_data['item_id'].tolist(),
                                np.array(item_embeddings_list))

        # Train ranking model (simulated)
        self.logger.info("Training ranking model...")
        self._train_ranking_model(interactions)

        training_time = time.time() - start_time
        self.logger.info(f"Offline training completed in {training_time:.2f} seconds")

    def _train_ranking_model(self, interactions: pd.DataFrame):
        """Train the ranking model (simulated)"""
        # In production, this would use actual interaction data
        # For demo, we simulate the training process
        features = np.random.randn(10000, 50)
        labels = np.random.randint(0, 2, 10000)

        # Simulate training by updating model weights
        self.ranking_model.model_weights = np.random.randn(50)

        self.logger.info("Ranking model training completed")

    def retrieve_candidates(self, user_id: str, context: Dict[str, Any],
                          k: int = None) -> List[str]:
        """
        Online retrieval phase using ANN
        """
        if k is None:
            k = self.max_candidates

        start_time = time.time()

        try:
            # Get user embedding
            user_embedding = self._get_user_embedding(user_id)

            # ANN search
            candidate_results = self.ann_index.query(user_embedding, k)
            candidates = [item for item, _ in candidate_results]

            # Fallback if ANN fails
            if not candidates:
                candidates = self.cold_start_handler.get_bootstrap_candidates(context, k)

            latency = (time.time() - start_time) * 1000
            self.metrics.record_latency(latency, 'retrieve_candidates')

            if latency > self.latency_slo_ms * 0.3:
                self.logger.warning(f"ANN retrieval latency: {latency:.2f}ms")

            return candidates

        except Exception as e:
            self.logger.error(f"Candidate retrieval failed: {e}")
            self.metrics.record_error('retrieval_failure', 'retrieve_candidates')
            return self.cold_start_handler.get_bootstrap_candidates(context, k)

    def _get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get user embedding with cold start handling"""
        # Try cache first
        cached_embedding = self.redis.get(f"user_embedding:{user_id}")
        if cached_embedding:
            return np.array(cached_embedding)

        # Cold start: generate embedding on the fly
        user_features = self.feature_store.get_user_features(user_id)
        user_embedding = self.embedding_generator.generate_user_embedding(user_features)

        # Cache for future use
        self.redis.set(f"user_embedding:{user_id}", user_embedding.tolist())

        return user_embedding

    def assemble_features(self, user_id: str, candidates: List[str]) -> np.ndarray:
        """
        Feature assembly phase
        Combine user, item, and context features
        """
        start_time = time.time()

        try:
            features = []

            # Get user features
            user_features = self.feature_store.get_user_features(user_id)

            # Get real-time session features
            session_features = self._get_session_features(user_id)

            for item in candidates:
                # Get item features
                item_features = self.feature_store.get_item_features(item_id=item)

                # Combine all features
                combined_features = (
                    list(user_features.values()) +
                    list(item_features.values()) +
                    list(session_features.values()) +
                    list(self._get_context_features().values())
                )

                # Pad or truncate to fixed size
                combined_features = combined_features[:50]  # Limit to 50 features
                combined_features.extend([0] * (50 - len(combined_features)))  # Pad zeros

                features.append(combined_features)

            latency = (time.time() - start_time) * 1000
            self.metrics.record_latency(latency, 'assemble_features')

            if latency > self.latency_slo_ms * 0.4:
                self.logger.warning(f"Feature assembly latency: {latency:.2f}ms")

            return np.array(features)

        except Exception as e:
            self.logger.error(f"Feature assembly failed: {e}")
            self.metrics.record_error('feature_assembly_failure', 'assemble_features')
            # Return random features as fallback
            return np.random.randn(len(candidates), 50)

    def _get_session_features(self, user_id: str) -> Dict[str, float]:
        """Get real-time session features"""
        session = self.user_sessions[user_id]

        return {
            'session_length': len(session['recent_views']),
            'recent_views_count': len(session['recent_views']),
            'recent_purchases_count': len(session['recent_purchases']),
            'session_recency': 1.0 if session['recent_views'] else 0.0
        }

    def _get_context_features(self) -> Dict[str, float]:
        """Get context features"""
        return {
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': 1.0 if datetime.now().weekday() >= 5 else 0.0,
            'is_holiday_season': 1.0 if datetime.now().month in [11, 12] else 0.0
        }

    def rank_candidates(self, features: np.ndarray, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Ranking phase using ML model
        """
        start_time = time.time()

        try:
            # Get model predictions
            scores = self.ranking_model.predict_proba(features)

            # Create item-score pairs
            item_scores = list(zip(candidates, scores))

            # Sort by score
            item_scores.sort(key=lambda x: x[1], reverse=True)

            latency = (time.time() - start_time) * 1000
            self.metrics.record_latency(latency, 'rank_candidates')

            if latency > self.latency_slo_ms * 0.3:
                self.logger.warning(f"Ranking latency: {latency:.2f}ms")

            return item_scores

        except Exception as e:
            self.logger.error(f"Ranking failed: {e}")
            self.metrics.record_error('ranking_failure', 'rank_candidates')
            # Return random scores as fallback
            random_scores = np.random.rand(len(candidates))
            return list(zip(candidates, random_scores))

    def re_rank_and_apply_rules(self, item_scores: List[Tuple[str, float]],
                               context: Dict[str, Any]) -> List[str]:
        """
        Re-ranking with diversity and business rules
        """
        start_time = time.time()

        try:
            # Apply business rules
            rule_adjusted_scores = self.business_rules.apply_rules(item_scores, context)

            # Apply diversity filter
            final_recommendations = self.diversity_filter.apply_diversity(rule_adjusted_scores)

            latency = (time.time() - start_time) * 1000
            self.metrics.record_latency(latency, 're_rank')

            return final_recommendations

        except Exception as e:
            self.logger.error(f"Re-ranking failed: {e}")
            self.metrics.record_error('reranking_failure', 're_rank')
            # Return top items without re-ranking
            return [item for item, _ in item_scores[:self.final_recommendations]]

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        End-to-end recommendation workflow
        """
        start_time = time.time()
        fallback_used = False

        try:
            self.metrics.record_request('recommend')

            # Update session with real-time signals
            self._update_session(request.user_id, request.context)

            # Step 1: Candidate retrieval
            candidates = self.retrieve_candidates(request.user_id, request.context)

            if not candidates:
                # Fallback to popular items
                candidates = self.cold_start_handler.get_bootstrap_candidates(request.context)
                fallback_used = True

            # Step 2: Feature assembly
            features = self.assemble_features(request.user_id, candidates)

            # Step 3: Ranking
            item_scores = self.rank_candidates(features, candidates)

            # Step 4: Re-ranking and business rules
            recommendations = self.re_rank_and_apply_rules(item_scores, request.context)

            # Extract scores for final recommendations
            final_scores = [score for item, score in item_scores
                           if item in recommendations[:self.final_recommendations]]

            total_latency = (time.time() - start_time) * 1000

            # Check SLO compliance
            if total_latency > self.latency_slo_ms:
                self.logger.warning(f"SLO exceeded: {total_latency:.2f}ms > {self.latency_slo_ms}ms")
                self.metrics.record_error('slo_exceeded', 'recommend')

            self.metrics.record_latency(total_latency, 'recommend')

            return RecommendationResponse(
                recommendations=recommendations[:self.final_recommendations],
                scores=final_scores[:self.final_recommendations],
                latency_ms=total_latency,
                fallback_used=fallback_used
            )

        except Exception as e:
            self.logger.error(f"Recommendation failed: {e}")
            self.metrics.record_error('recommendation_failure', 'recommend')

            # Ultimate fallback
            fallback_items = self.cold_start_handler.get_bootstrap_candidates(
                request.context, self.final_recommendations
            )

            return RecommendationResponse(
                recommendations=fallback_items[:self.final_recommendations],
                scores=[0.5] * len(fallback_items[:self.final_recommendations]),
                latency_ms=(time.time() - start_time) * 1000,
                fallback_used=True,
                error_message=str(e)
            )

    def _update_session(self, user_id: str, context: Dict[str, Any]):
        """Update user session with real-time signals"""
        session = self.user_sessions[user_id]

        if 'recent_view' in context:
            session['recent_views'].append(context['recent_view'])

        if 'recent_purchase' in context:
            session['recent_purchases'].append(context['recent_purchase'])

        # Update recency
        session['last_activity'] = datetime.now()

    def evaluate_offline(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Offline evaluation metrics
        """
        from sklearn.metrics import roc_auc_score, ndcg_score

        # Simulate predictions
        y_true = test_data['label'].values
        y_pred = np.random.rand(len(y_true))

        # Calculate metrics
        auc = roc_auc_score(y_true, y_pred)
        ndcg = ndcg_score([y_true], [y_pred])

        # Recall@K
        recall_at_10 = np.mean([1 if random.random() < 0.3 else 0 for _ in range(len(y_true))])

        metrics = {
            'auc': auc,
            'ndcg': ndcg,
            'recall_at_10': recall_at_10
        }

        self.logger.info(f"Offline metrics: {metrics}")
        return metrics

    def run_ab_test(self, test_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Online A/B testing simulation
        """
        # Simulate business metrics
        base_ctr = 0.08  # 8% base click-through rate
        base_cvr = 0.02  # 2% base conversion rate
        base_gmv_per_user = 50  # $50 GMV per user

        # Apply test variations
        if test_config.get('treatment') == 'new_model':
            ctr_multiplier = 1.15  # 15% improvement
            cvr_multiplier = 1.10  # 10% improvement
            gmv_multiplier = 1.12  # 12% improvement
        else:
            ctr_multiplier = 1.0
            cvr_multiplier = 1.0
            gmv_multiplier = 1.0

        # Add some randomness
        ctr = base_ctr * ctr_multiplier * random.uniform(0.95, 1.05)
        cvr = base_cvr * cvr_multiplier * random.uniform(0.95, 1.05)
        gmv_per_user = base_gmv_per_user * gmv_multiplier * random.uniform(0.95, 1.05)

        # Calculate derived metrics
        rpm = gmv_per_user * ctr * cvr  # Revenue per mille

        results = {
            'ctr': ctr,
            'cvr': cvr,
            'gmv_per_user': gmv_per_user,
            'rpm': rpm
        }

        # Record metrics
        for metric_name, value in results.items():
            self.metrics.record_business_metric(metric_name, value)

        self.logger.info(f"A/B test results: {results}")
        return results

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health and performance metrics"""
        return {
            'p99_latency_ms': self.metrics.get_p99_latency(),
            'error_rate': self.metrics.get_error_rate(),
            'total_requests': sum(self.metrics.request_counts.values()),
            'uptime_hours': 24.0,  # Simulated
            'memory_usage_mb': 512,  # Simulated
            'cpu_usage_percent': 45.0,  # Simulated
            'active_sessions': len(self.user_sessions)
        }

def demonstrate_production_recommender():
    """Demonstrate the production recommender system"""
    print("=" * 80)
    print("PRODUCTION RECOMMENDER SYSTEM DEMONSTRATION")
    print("=" * 80)

    # Initialize system
    print("1. Initializing Production Recommender System...")
    recommender = ProductionRecommenderSystem(latency_slo_ms=200)

    # Generate sample data
    print("\n2. Generating Sample Data...")
    users = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(1000)]
    })

    items = pd.DataFrame({
        'item_id': [f'cat_{i%10}_item_{j}' for i in range(10) for j in range(100)]
    })

    interactions = pd.DataFrame({
        'user_id': np.random.choice(users['user_id'], 10000),
        'item_id': np.random.choice(items['item_id'], 10000),
        'label': np.random.randint(0, 2, 10000)
    })

    # Offline training
    print("\n3. Offline Training Phase...")
    recommender.train_offline(users, items, interactions)

    # Offline evaluation
    print("\n4. Offline Evaluation...")
    test_data = pd.DataFrame({
        'user_id': np.random.choice(users['user_id'], 1000),
        'item_id': np.random.choice(items['item_id'], 1000),
        'label': np.random.randint(0, 2, 1000)
    })
    offline_metrics = recommender.evaluate_offline(test_data)
    print(f"   AUC: {offline_metrics['auc']:.4f}")
    print(f"   NDCG: {offline_metrics['ndcg']:.4f}")
    print(f"   Recall@10: {offline_metrics['recall_at_10']:.4f}")

    # Online recommendations
    print("\n5. Online Recommendation Phase...")

    # Test scenarios
    test_scenarios = [
        {
            'name': 'Known User with Context',
            'user_id': 'user_123',
            'context': {
                'category': 'cat_5',
                'recent_view': 'cat_5_item_123',
                'season': 'holiday',
                'query': 'electronics'
            }
        },
        {
            'name': 'Cold Start User',
            'user_id': 'new_user_999',
            'context': {
                'category': 'cat_3',
                'location': 'new_york',
                'query': 'clothing'
            }
        },
        {
            'name': 'Session with Recent Activity',
            'user_id': 'user_456',
            'context': {
                'recent_view': 'cat_7_item_456',
                'recent_purchase': 'cat_7_item_789',
                'session_id': 'session_123'
            }
        }
    ]

    for scenario in test_scenarios:
        print(f"\n   {scenario['name']}:")

        request = RecommendationRequest(
            user_id=scenario['user_id'],
            context=scenario['context'],
            session_id=scenario['context'].get('session_id', 'default_session'),
            timestamp=datetime.now(),
            request_id=f"req_{random.randint(1000, 9999)}"
        )

        response = recommender.recommend(request)

        print(f"     Recommendations: {response.recommendations[:5]}")
        print(f"     Latency: {response.latency_ms:.2f}ms")
        print(f"     Fallback Used: {response.fallback_used}")
        if response.error_message:
            print(f"     Error: {response.error_message}")

    # A/B Testing
    print("\n6. A/B Testing Phase...")

    ab_test_configs = [
        {'name': 'Control Group', 'treatment': 'control'},
        {'name': 'New Model', 'treatment': 'new_model'}
    ]

    for config in ab_test_configs:
        print(f"\n   {config['name']}:")
        results = recommender.run_ab_test(config)
        print(f"     CTR: {results['ctr']:.4f} ({results['ctr']*100:.2f}%)")
        print(f"     CVR: {results['cvr']:.4f} ({results['cvr']*100:.2f}%)")
        print(f"     GMV per User: ${results['gmv_per_user']:.2f}")
        print(f"     RPM: ${results['rpm']:.2f}")

    # System Health
    print("\n7. System Health Check...")
    health = recommender.get_system_health()
    print(f"   P99 Latency: {health['p99_latency_ms']:.2f}ms")
    print(f"   Error Rate: {health['error_rate']*100:.2f}%")
    print(f"   Total Requests: {health['total_requests']}")
    print(f"   Active Sessions: {health['active_sessions']}")
    print(f"   CPU Usage: {health['cpu_usage_percent']:.1f}%")
    print(f"   Memory Usage: {health['memory_usage_mb']}MB")

    # Performance under load
    print("\n8. Load Testing Simulation...")
    load_test_results = []

    for i in range(100):  # Simulate 100 requests
        user_id = f'user_{random.randint(0, 999)}'
        context = {
            'category': f'cat_{random.randint(0, 9)}',
            'recent_view': f'cat_{random.randint(0, 9)}_item_{random.randint(0, 99)}'
        }

        request = RecommendationRequest(
            user_id=user_id,
            context=context,
            session_id=f'session_{random.randint(0, 999)}',
            timestamp=datetime.now(),
            request_id=f'load_req_{i}'
        )

        response = recommender.recommend(request)
        load_test_results.append(response.latency_ms)

    load_p99 = np.percentile(load_test_results, 99)
    load_avg = np.mean(load_test_results)
    slo_compliance_rate = sum(1 for latency in load_test_results if latency <= 200) / len(load_test_results)

    print(f"   Load Test Results (100 requests):")
    print(f"     Average Latency: {load_avg:.2f}ms")
    print(f"     P99 Latency: {load_p99:.2f}ms")
    print(f"     SLO Compliance Rate: {slo_compliance_rate*100:.1f}%")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("✓ Production-grade architecture with proper SLOs")
    print("✓ Offline training with embeddings and ranking models")
    print("✓ Online retrieval with ANN (Approximate Nearest Neighbors)")
    print("✓ Feature assembly with mock feature store and Redis")
    print("✓ ML-based ranking with real-time features")
    print("✓ Re-ranking with diversity and business rules")
    print("✓ Cold start handling with multiple strategies")
    print("✓ Real-time signals and session context")
    print("✓ Comprehensive evaluation (offline + online A/B testing)")
    print("✓ Reliability with fallbacks and graceful degradation")
    print("✓ Performance monitoring and health checks")
    print("✓ Load testing and SLO compliance")

if __name__ == "__main__":
    demonstrate_production_recommender()
