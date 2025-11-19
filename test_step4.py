#!/usr/bin/env python3
print("=== STEP 4: Testing Evolutionary Learning ===")

try:
    from src.data_preprocessing import HiggsDataPreprocessor
    from src.evolutionary_learning import EvolutionaryLearner
    
    print("1. Setting up data...")
    preprocessor = HiggsDataPreprocessor()
    data = preprocessor.load_data("data/higgs_boson.csv")
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess(data, n_features=8)
    
    print("2. Creating evolutionary learner...")
    evol_learner = EvolutionaryLearner(
        n_features=len(features),
        population_size=20,  # Small for testing
        generations=5        # Few generations for quick test
    )
    
    print("3. Running evolutionary learning (this may take a moment)...")
    print("   Using small subset for quick testing...")
    population, logbook = evol_learner.run_evolution(X_train[:100], y_train[:100])
    
    print("4. Checking results...")
    best_individual = max(population, key=lambda ind: ind.fitness.values[0])
    best_fitness = best_individual.fitness.values[0]
    
    print(f"   ✓ Evolutionary learning completed!")
    print(f"   - Best fitness: {best_fitness:.4f}")
    print(f"   - Population size: {len(population)}")
    print(f"   - Number of generations completed: {len(logbook)}")
    
    # Test prediction
    print("5. Testing predictions...")
    y_pred = evol_learner.predict(best_individual, X_test[:10])
    accuracy = (y_pred == y_test[:10]).mean()
    print(f"   - Prediction accuracy on 10 test samples: {accuracy:.2f}")
    
except Exception as e:
    print(f"✗ Evolutionary learning failed: {e}")
    import traceback
    traceback.print_exc()