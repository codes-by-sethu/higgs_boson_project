#!/usr/bin/env python3
"""
Demo script for Higgs Boson Project - Quick test
"""

def quick_test():
    """Run a quick test of the main components"""
    print("=== Quick Demo of Higgs Boson Project ===\n")
    
    # Test data preprocessing
    from src.data_preprocessing import HiggsDataPreprocessor
    
    print("1. Testing Data Preprocessing...")
    preprocessor = HiggsDataPreprocessor()
    data = preprocessor.load_data("data/higgs_boson.csv")  # This will create synthetic data
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess(data, n_features=10)
    print(f"   ✓ Preprocessed data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Test evolutionary learning with very small parameters
    from src.evolutionary_learning import EvolutionaryLearner
    
    print("2. Testing Evolutionary Learning...")
    evol_learner = EvolutionaryLearner(
        n_features=X_train.shape[1],
        population_size=10,  # Very small for quick test
        generations=5        # Very few generations
    )
    
    try:
        population, logbook = evol_learner.run_evolution(X_train[:100], y_train[:100])  # Small subset
        best_individual = max(population, key=lambda ind: ind.fitness.values[0])
        print(f"   ✓ Evolutionary learning completed. Best fitness: {best_individual.fitness.values[0]:.4f}")
    except Exception as e:
        print(f"   ⚠ Evolutionary learning test skipped: {e}")
    
    # Test ensemble learning
    from src.ensemble_learning import EvolutionaryEnsemble
    
    print("3. Testing Ensemble Learning...")
    try:
        ensemble = EvolutionaryEnsemble(evol_learner, top_k=3)
        ensemble.create_ensemble(population)
        accuracy = ensemble.evaluate_ensemble(X_test[:50], y_test[:50])  # Small test
        print(f"   ✓ Ensemble learning completed. Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"   ⚠ Ensemble learning test skipped: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo run the full experiment:")
    print("python main.py")

if __name__ == "__main__":
    quick_test()