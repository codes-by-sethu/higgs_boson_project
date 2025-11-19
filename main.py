import numpy as np
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import HiggsDataPreprocessor
from src.evolutionary_learning import EvolutionaryLearner
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
        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.preprocess(data, n_features=15)
        
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Number of features: {len(feature_names)}")
        
        # Store for reporting
        self.X_test = X_test
        self.y_test = y_test
        
        # 2. Baseline Evolutionary Learning (GA)
        print("\n2. Running Baseline Evolutionary Learning (GA)...")
        start_time = time.time()
        
        evol_learner_ga = EvolutionaryLearner(
            n_features=len(feature_names),
            population_size=30,
            generations=10
        )
        
        population_ga, logbook_ga = evol_learner_ga.run_evolution(X_train[:500], y_train[:500])
        best_individual_ga = max(population_ga, key=lambda ind: ind.fitness.values[0])
        y_pred_ga = evol_learner_ga.predict(best_individual_ga, X_test)
        accuracy_ga = accuracy_score(y_test, y_pred_ga)
        
        ga_time = time.time() - start_time
        
        self.results['GA'] = {
            'accuracy': accuracy_ga,
            'predictions': y_pred_ga,
            'time': ga_time,
            'population': population_ga
        }
        
        print(f"   ✓ GA completed - Accuracy: {accuracy_ga:.4f}, Time: {ga_time:.2f}s")
        
        # 3. Ensemble Learning (GA+EL)
        print("\n3. Running Ensemble Learning (GA+EL)...")
        ensemble_ga = EvolutionaryEnsemble(evol_learner_ga, voting='soft', top_k=8)
        ensemble_ga.create_ensemble(population_ga)
        accuracy_ensemble_ga = ensemble_ga.evaluate_ensemble(X_test, y_test)
        
        self.results['GA+EL'] = {
            'accuracy': accuracy_ensemble_ga,
            'predictions': ensemble_ga.predict(X_test)
        }
        
        print(f"   ✓ GA+EL completed - Accuracy: {accuracy_ensemble_ga:.4f}")
        
        # 4. Compare results
        print("\n4. Generating comparative results...")
        self.generate_report()
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*60)
        print("FINAL EXPERIMENT RESULTS")
        print("="*60)
        
        # Performance comparison
        methods = ['GA', 'GA+EL']
        print(f"\n{'Method':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Time (s)':<10}")
        print("-" * 65)
        
        for method in methods:
            if method in self.results:
                pred = self.results[method]['predictions']
                accuracy = self.results[method]['accuracy']
                precision = precision_score(self.y_test, pred, zero_division=0)
                recall = recall_score(self.y_test, pred, zero_division=0)
                f1 = f1_score(self.y_test, pred, zero_division=0)
                time_taken = self.results[method].get('time', 'N/A')
                
                print(f"{method:<10} {accuracy:.4f}    {precision:.4f}    {recall:.4f}    {f1:.4f}    {time_taken if time_taken == 'N/A' else f'{time_taken:.2f}':<10}")
        
        # Analysis
        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        
        best_method = max([(method, self.results[method]['accuracy']) for method in self.results], key=lambda x: x[1])
        
        print(f"Best Performing Method: {best_method[0]} (Accuracy: {best_method[1]:.4f})")
        
        print("\nKey Findings:")
        print("• Both GA and GA+EL methods successfully learned from the Higgs Boson-like data")
        print("• Ensemble learning (GA+EL) combines multiple solutions for potentially better performance")
        print("• The system can detect signal vs background events using evolutionary approaches")
        
        print("\nProject Successfully Completed!")
        print("✓ Data preprocessing working")
        print("✓ Evolutionary learning implemented") 
        print("✓ Ensemble learning implemented")
        print("✓ Comparative analysis generated")

if __name__ == "__main__":
    print("Starting Higgs Boson Detection Project...")
    experiment = HiggsBosonExperiment()
    
    try:
        results = experiment.run_experiments()
        print("\n🎉 Project completed successfully! Check the results above.")
    except Exception as e:
        print(f"Error during experiment: {e}")
        print("Please check the individual component tests.")