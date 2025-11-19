import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class HiggsDataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.selector = None
        
    def load_data(self, file_path):
        """Load Higgs Boson dataset"""
        try:
            data = pd.read_csv(file_path)
            print(f"Dataset shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data if real dataset is not available"""
        n_samples = 10000
        n_features = 28
        
        np.random.seed(self.random_state)
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        y = (X[:, 0] + X[:, 1]**2 + np.sin(X[:, 2]) + 
             np.random.normal(0, 0.5, n_samples)) > 0
        y = y.astype(int)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data['label'] = y
        
        print("Using synthetic dataset for demonstration")
        return data
    
    def preprocess(self, data, target_column='label', n_features=20):
        """Main preprocessing pipeline"""
        X = data.drop(columns=[target_column], errors='ignore')
        y = data[target_column]
        
        X = self._handle_missing_values(X)
        X_selected = self._select_features(X, y, n_features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (X_train_scaled, X_test_scaled, 
                y_train.values, y_test.values, 
                X_selected.columns.tolist())
    
    def _handle_missing_values(self, X):
        """Handle missing values in the dataset"""
        if X.isnull().sum().sum() > 0:
            print(f"Found {X.isnull().sum().sum()} missing values")
            X = X.fillna(X.median())
        return X
    
    def _select_features(self, X, y, n_features):
        """Select top k features using ANOVA F-value"""
        if len(X.columns) > n_features:
            self.selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = self.selector.fit_transform(X, y)
            selected_features = X.columns[self.selector.get_support()].tolist()
            print(f"Selected {len(selected_features)} features")
            return pd.DataFrame(X_selected, columns=selected_features)
        return X

if __name__ == "__main__":
    preprocessor = HiggsDataPreprocessor()
    data = preprocessor.load_data("data/higgs_boson.csv")
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess(data)
    print(f"Preprocessing completed. Training set: {X_train.shape}")
