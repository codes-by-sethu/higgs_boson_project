#!/usr/bin/env python3
print("=== STEP 3: Testing Data Preprocessing ===")

try:
    from src.data_preprocessing import HiggsDataPreprocessor
    
    print("1. Creating preprocessor...")
    preprocessor = HiggsDataPreprocessor()
    
    print("2. Loading data...")
    data = preprocessor.load_data("data/higgs_boson.csv")
    print(f"   ✓ Data loaded with shape: {data.shape}")
    print(f"   ✓ Columns: {list(data.columns)}")
    
    print("3. Preprocessing data...")
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess(data, n_features=10)
    
    print(f"   ✓ Preprocessing completed!")
    print(f"   - Training set: {X_train.shape}")
    print(f"   - Test set: {X_test.shape}")
    print(f"   - Number of features: {len(features)}")
    print(f"   - First 5 features: {features[:5]}")
    
except Exception as e:
    print(f"✗ Data preprocessing failed: {e}")
    import traceback
    traceback.print_exc()