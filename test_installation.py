#!/usr/bin/env python3
"""
Simple test script to verify all dependencies are installed correctly
"""

def test_imports():
    """Test if all required packages can be imported"""
    packages = [
        'numpy', 'pandas', 'sklearn', 'deap', 'modAL', 
        'matplotlib', 'seaborn'
    ]
    
    print("Testing package imports...")
    
    for package in packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            return False
    
    print("\nAll packages imported successfully!")
    return True

def test_modules():
    """Test if our custom modules can be imported"""
    modules = [
        'src.data_preprocessing',
        'src.evolutionary_learning', 
        'src.active_learning',
        'src.ensemble_learning'
    ]
    
    print("\nTesting custom modules...")
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
    
    print("All custom modules imported successfully!")
    return True

if __name__ == "__main__":
    print("=== Testing Higgs Boson Project Installation ===\n")
    
    success1 = test_imports()
    success2 = test_modules()
    
    if success1 and success2:
        print("\n🎉 All tests passed! You're ready to run the project.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the main experiment: python main.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")