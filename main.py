import numpy as np
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from src.data_preprocessing import HiggsDataPreprocessor
from src.evolutionary_learning import EvolutionaryLearner
from src.active_learning import EvolutionaryActiveLearner
from src.ensemble_learning import EvolutionaryEnsemble

class HiggsBosonExperiment:
    def __init__(self, data_path='data/higgs_boson.csv'):
        self.data_path = data_path
        self.preprocessor = HiggsDataPreprocessor()
        self.results = {}
        
    def run_experiments(self):
        """Run all experiments"""
        print("=== Higgs Boson Detection Experiment ===")
        
        # 1. Data Preprocessing
        print("\n1. Loading and preprocessing data...")
        data = self.preprocessor.load_data(self.data_path)
        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.preprocess(data)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # Store test data for report generation
        self.X_test = X_test
        self.y_test = y_test
        
        # 2. Baseline Evolutionary Learning
        print("\n2. Running Baseline Evolutionary Learning...")
        start_time = time.time()
        
        evol_learner = EvolutionaryLearner(
            n_features=len(feature_names),
            population_size=30,  # Smaller for faster testing
            generations=10
        )
        
        population_ga, logbook_ga = evol_learner.run_evolution(X_train, y_train)
        
        # Get best individual
        best_individual_ga = max(population_ga, key=lambda ind: ind.fitness.values[0])
        y_pred_ga = evol_learner.predict(best_individual_ga, X_test)
        accuracy_ga = np.mean(y_pred_ga == y_test)
        
        ga_time = time.time() - start_time
        
        self.results['GA'] = {
            'accuracy': accuracy_ga,
            'predictions': y_pred_ga,
            'time': ga_time,
            'population': population_ga,
            'logbook': logbook_ga
        }
        
        print(f"GA - Accuracy: {accuracy_ga:.4f}, Time: {ga_time:.2f}s")
        
        # 3. Ensemble Learning on GA
        print("\n3. Running Ensemble Learning on GA population...")
        ensemble_ga = EvolutionaryEnsemble(evol_learner, voting='soft', top_k=5)
        ensemble_ga.create_ensemble(population_ga)
        accuracy_ensemble_ga = ensemble_ga.evaluate_ensemble(X_test, y_test)
        
        self.results['GA+EL'] = {
            'accuracy': accuracy_ensemble_ga,
            'predictions': ensemble_ga.predict(X_test)
        }
        
        print(f"GA+EL - Accuracy: {accuracy_ensemble_ga:.4f}")
        
        # 4. Evolutionary Learning + Active Learning (Simplified for testing)
        print("\n4. Running Evolutionary Learning + Active Learning...")
        start_time = time.time()
        
        # Use smaller pool for testing
        pool_size = min(1000, len(X_train))
        pool_idx = np.random.choice(len(X_train), pool_size, replace=False)
        X_pool = X_train[pool_idx]
        y_pool = y_train[pool_idx]
        
        evol_learner_al = EvolutionaryLearner(
            n_features=len(feature_names),
            population_size=30,
            generations=10
        )
        
        active_learner = EvolutionaryActiveLearner(
            evolutionary_learner=evol_learner_al,
            query_strategy='uncertainty',
            query_every_n=2,
            initial_pool_size=300
        )
        
        # Run simplified active learning
        learner_al, population_ga_al, performance_history = active_learner.run_evolution_with_active_learning(
            X_pool, y_pool, X_test, y_test
        )
        
        accuracy_ga_al = learner_al.score(X_test, y_test)
        ga_al_time = time.time() - start_time
        
        self.results['GA+AL'] = {
            'accuracy': accuracy_ga_al,
            'predictions': learner_al.predict(X_test),
            'time': ga_al_time,
            'population': population_ga_al,
            'performance_history': performance_history
        }
        
        print(f"GA+AL - Accuracy: {accuracy_ga_al:.4f}, Time: {ga_al_time:.2f}s")
        
        # 5. Ensemble Learning on GA+AL
        print("\n5. Running Ensemble Learning on GA+AL population...")
        ensemble_ga_al = EvolutionaryEnsemble(evol_learner_al, voting='soft', top_k=5)
        ensemble_ga_al.create_ensemble(population_ga_al)
        accuracy_ensemble_ga_al = ensemble_ga_al.evaluate_ensemble(X_test, y_test)
        
        self.results['GA+AL+EL'] = {
            'accuracy': accuracy_ensemble_ga_al,
            'predictions': ensemble_ga_al.predict(X_test)
        }
        
        print(f"GA+AL+EL - Accuracy: {accuracy_ensemble_ga_al:.4f}")
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS")
        print("="*50)
        
        # Comparative table
        methods = ['GA', 'GA+AL', 'GA+EL', 'GA+AL+EL']
        
        print(f"{'Method':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Time (s)':<10}")
        print("-" * 65)
        
        for method in methods:
            if method in self.results:
                pred = self.results[method]['predictions']
                accuracy = self.results[method]['accuracy']
                
                # Calculate metrics
                precision = precision_score(self.y_test, pred, zero_division=0)
                recall = recall_score(self.y_test, pred, zero_division=0)
                f1 = f1_score(self.y_test, pred, zero_division=0)
                time_taken = self.results[method].get('time', 'N/A')
                
                print(f"{method:<12} {accuracy:.4f}    {precision:.4f}    {recall:.4f}    {f1:.4f}    {time_taken if time_taken == 'N/A' else f'{time_taken:.2f}':<10}")
        
        # Analysis
        self._write_analysis()
    
    def _write_analysis(self):
        """Write analysis of results"""
        print("\n" + "="*50)
        print("ANALYSIS")
        print("="*50)
        
        best_method = max([(method, self.results[method]['accuracy']) 
                          for method in self.results], 
                         key=lambda x: x[1])
        
        print(f"Best Performing Method: {best_method[0]} (Accuracy: {best_method[1]:.4f})")
        
        print("\nKey Observations:")
        print("- GA: Baseline evolutionary approach")
        print("- GA+AL: Active learning for sample efficiency") 
        print("- GA+EL: Ensemble learning for robustness")
        print("- GA+AL+EL: Combined approach")
        
        print("\nNext steps for improvement:")
        print("- Increase population size and generations")
        print("- Try different active learning strategies")
        print("- Optimize hyperparameters")
        print("- Use real Higgs Boson dataset")

if __name__ == "__main__":
    print("Starting Higgs Boson Detection Experiment...")
    experiment = HiggsBosonExperiment()
    
    try:
        results = experiment.run_experiments()
        experiment.generate_report()
    except Exception as e:
        print(f"Error during experiment: {e}")
        print("This might be due to missing dependencies.")
        print("Please run: pip install -r requirements.txt")
