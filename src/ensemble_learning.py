import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class EvolutionaryEnsemble:
    def __init__(self, evolutionary_learner, voting='soft', top_k=10):
        self.evolutionary_learner = evolutionary_learner
        self.voting = voting
        self.top_k = top_k
        self.ensemble = []
        self.weights = []
        
    def create_ensemble(self, population):
        """Create ensemble from final population"""
        # Select top individuals
        sorted_population = sorted(population, 
                                 key=lambda ind: ind.fitness.values[0], 
                                 reverse=True)
        self.ensemble = sorted_population[:self.top_k]
        
        # Calculate weights based on fitness
        fitnesses = [ind.fitness.values[0] for ind in self.ensemble]
        self.weights = np.array(fitnesses) / sum(fitnesses)
        
        print(f"Created ensemble with {len(self.ensemble)} members")
        
    def predict(self, X):
        """Make predictions using ensemble"""
        if self.voting == 'hard':
            return self._hard_voting(X)
        else:
            return self._soft_voting(X)
    
    def _hard_voting(self, X):
        """Hard voting ensemble"""
        predictions = []
        for individual in self.ensemble:
            pred = self.evolutionary_learner.predict(individual, X)
            predictions.append(pred)
        
        # Majority vote
        predictions = np.array(predictions)
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        return ensemble_pred.astype(int)
    
    def _soft_voting(self, X):
        """Soft voting ensemble with weights"""
        predictions = []
        for individual, weight in zip(self.ensemble, self.weights):
            # Get continuous outputs
            func = self.evolutionary_learner.toolbox.compile(expr=individual)
            outputs = []
            for sample in X:
                try:
                    output = func(*sample)
                    outputs.append(output)
                except:
                    outputs.append(0)
            
            # Apply weight to outputs
            weighted_outputs = np.array(outputs) * weight
            predictions.append(weighted_outputs)
        
        # Weighted average
        ensemble_output = np.sum(predictions, axis=0)
        return (ensemble_output > 0).astype(int)
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

if __name__ == "__main__":
    print("Ensemble Learning module created successfully.")
