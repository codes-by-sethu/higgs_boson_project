#!/usr/bin/env python3
print("=== STEP 5: Testing Ensemble Learning ===")

try:
    from src.data_preprocessing import HiggsDataPreprocessor
    from src.evolutionary_learning import EvolutionaryLearner
    from src.ensemble_learning import EvolutionaryEnsemble
    
    print("1. Setting up data...")
    preprocessor = HiggsDataPreprocessor()
    data = preprocessor.load_data("data/higgs_boson.csv")
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess(data, n_features=8)
    
    print("2. Creating evolutionary population...")
    evol_learner = EvolutionaryLearner(
        n_features=len(features),
        population_size=15,  # Small for testing
        generations=4        # Few generations for quick test
    )
    
    print("3. Running evolution to create population...")
    population, _ = evol_learner.run_evolution(X_train[:80], y_train[:80])
    
    print("4. Creating ensemble from population...")
    ensemble = EvolutionaryEnsemble(evol_learner, voting='soft', top_k=5)
    ensemble.create_ensemble(population)
    
    print("5. Testing ensemble performance...")
    ensemble_accuracy = ensemble.evaluate_ensemble(X_test[:20], y_test[:20])
    
    print(f"   ✓ Ensemble learning completed!")
    print(f"   - Ensemble size: {len(ensemble.ensemble)} individuals")
    print(f"   - Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Compare with best individual
    best_individual = max(population, key=lambda ind: ind.fitness.values[0])
    best_accuracy = (evol_learner.predict(best_individual, X_test[:20]) == y_test[:20]).mean()
    print(f"   - Best individual accuracy: {best_accuracy:.4f}")
    print(f"   - Ensemble improvement: {ensemble_accuracy - best_accuracy:+.4f}")
    
except Exception as e:
    print(f"✗ Ensemble learning failed: {e}")
    import traceback
    traceback.print_exc()