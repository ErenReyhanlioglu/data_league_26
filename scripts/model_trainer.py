import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from scripts.config import MODELS_DIR

class ModelTrainer:
    """
    Modular trainer designed for external parameter injection and manual persistence.
    Supports LightGBM, CatBoost, XGBoost, and Random Forest.
    """
    
    def __init__(self, model_type='lgbm', params=None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the selected model using the parameters passed during initialization.
        No automatic saving occurs here to allow for manual result evaluation.
        """
        if self.model_type == 'lgbm':
            self.model = lgb.LGBMClassifier(**self.params)
            # Use early stopping if validation data is provided
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                callbacks=[lgb.early_stopping(100, verbose=False)] if X_val is not None else None
            )

        elif self.model_type == 'catboost':
            # CatBoost handles early stopping internally via early_stopping_rounds
            self.model = CatBoostClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val) if X_val is not None else None,
                verbose=False
            )

        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                verbose=False
            )

        elif self.model_type == 'rf':
            # Random Forest does not support eval_sets in the same way as GBDTs
            self.model = RandomForestClassifier(**self.params)
            self.model.fit(X_train, y_train)

        return self.model

    def predict_proba(self, X):
        """Returns the probability estimates for the positive class (1)."""
        if self.model == None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def manual_save(self, filename):
        """
        Saves the current model state only when explicitly called by the user.
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"{self.model_type}_{filename}.joblib")
        joblib.dump(self.model, path)
        print(f"--- Persistence Success ---")
        print(f"Model manually saved at: {path}")