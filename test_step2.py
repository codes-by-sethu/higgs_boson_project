#!/usr/bin/env python3
print("=== STEP 2: Testing Custom Modules ===")

modules_to_test = [
    "src.data_preprocessing",
    "src.evolutionary_learning", 
    "src.ensemble_learning",
    "src.active_learning"
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✓ {module}")
    except ImportError as e:
        print(f"✗ {module}: {e}")

print("\nIf all modules imported successfully, we can proceed to Step 3.")