#!/usr/bin/env python3

packages_to_test = [
    "numpy",
    "pandas", 
    "sklearn",
    "deap",
    "matplotlib",
    "seaborn"
]

for package in packages_to_test:
    try:
        if package == "sklearn":
            __import__("sklearn")
        else:
            __import__(package)
        print(f"✓ {package}")
    except ImportError as e:
        print(f"✗ {package}: {e}")

print("\nIf you see any '✗' above, install missing packages with:")
print("pip install numpy pandas scikit-learn deap matplotlib seaborn")