import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.base import BaseEstimator, ClassifierMixin

class EvolutionaryActiveLearner:
    def __init__(self, evolutionary_learner, query_strategy='uncertainty', 
                 query_every_n=5, initial_pool_size=1000):
        self.evolutionary_learner = evolutionary_learner
        self.query_strategy = query_strategy
        self.query_every_n = query_every_n
        self.initial_pool_size = initial_pool_size
        self.query_history = []
        
    def get_query_strategy(self):
        """Get the query strategy function"""
        strategies = {
            'uncertainty': uncertainty_sampling,
            'margin': margin_sampling,
            'entropy': entropy_sampling
        }
        return strategies.get(self.query_strategy, uncertainty_sampling)
    
    def create_proxy_classifier(self, population):
        """Create a proxy classifier from evolutionary population for Active Learning"""
        class EvolutionaryProxyClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self, evolutionary_learner, population):
                self.evolutionary_learner = evolutionary_learner
                self.population = population
                self.best_individual = None
                
            def fit(self, X, y):
                # Use the best individual from current population
                self.best_individual = max(self.population, 
                                         key=lambda ind: ind.fitness.values[0])
                return self
                
            def predict(self, X):
                return self.evolutionary_learner.predict(self.best_individual, X)
                
            def predict_proba(self, X):
                # For uncertainty sampling, we need probability estimates
                predictions = self.predict(X)
                proba = np.zeros((len(predictions), 2))
                proba[:, 1] = predictions
                proba[:, 0] = 1 - predictions
                return proba
        
        return EvolutionaryProxyClassifier(self.evolutionary_learner, population)
    
    def run_evolution_with_active_learning(self, X_pool, y_pool, X_test, y_test):
        """Run evolutionary learning with active learning"""
        # Initial training set
        initial_idx = np.random.choice(len(X_pool), 
                                     self.initial_pool_size, 
                                     replace=False)
        X_train = X_pool[initial_idx]
        y_train = y_pool[initial_idx]
        
        # Remove initial samples from pool
        mask = np.ones(len(X_pool), bool)
        mask[initial_idx] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        
        performance_history = []
        query_strategy_func = self.get_query_strategy()
        
        # Initial evolution
        print("Initial evolutionary learning...")
        population, logbook = self.evolutionary_learner.run_evolution(
            X_train, y_train
        )
        
        # Create active learner
        proxy_classifier = self.create_proxy_classifier(population)
        learner = ActiveLearner(
            estimator=proxy_classifier,
            query_strategy=query_strategy_func,
            X_training=X_train, y_training=y_train
        )
        
        # Active learning loop
        n_queries = min(10, len(X_pool) // 100)  # Limit queries for demo
        
        for query_idx in range(n_queries):
            if len(X_pool) == 0:
                break
                
            # Query new samples
            query_idx, query_instance = learner.query(X_pool, n_instances=100)
            
            # Teach with new samples
            learner.teach(
                X=X_pool[query_idx], 
                y=y_pool[query_idx]
            )
            
            # Update training set
            X_train = np.vstack([X_train, X_pool[query_idx]])
            y_train = np.concatenate([y_train, y_pool[query_idx]])
            
            # Remove queried instances from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            
            # Retrain evolutionary learner with expanded dataset
            if query_idx % self.query_every_n == 0:
                print(f"Active Learning Query {query_idx}: Retraining evolution...")
                population, _ = self.evolutionary_learner.run_evolution(
                    X_train, y_train
                )
                
                # Update proxy classifier
                proxy_classifier = self.create_proxy_classifier(population)
                learner.estimator = proxy_classifier
            
            # Evaluate performance
            accuracy = learner.score(X_test, y_test)
            performance_history.append(accuracy)
            self.query_history.append(len(X_train))
            
            print(f"Query {query_idx}: Test Accuracy = {accuracy:.4f}, "
                  f"Training size = {len(X_train)}")
        
        return learner, population, performance_history

if __name__ == "__main__":
    print("Active Learning module created successfully.")
