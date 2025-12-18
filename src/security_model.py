from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from src.logger import get_logger

logger = get_logger(__name__)

class SecurityModel:
    """Modèle de détection d'anomalies de sécurité."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
        self.model_path = model_path
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Entraîner le modèle sur les données fournies."""
        logger.info(f"Entraînement du modèle sur {len(X)} échantillons")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Entraînement
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": self.model.score(X_test, y_test),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Modèle entraîné - ROC AUC: {metrics['roc_auc']:.3f}")
        
        # Sauvegarder si un chemin est fourni
        if self.model_path:
            self.save(self.model_path)
        
        return metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prédire si une journée est anormale."""
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné")
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Retourner les probabilités de prédiction."""
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné")
        
        return self.model.predict_proba(features)
    
    def save(self, path: Path) -> None:
        """Sauvegarder le modèle."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Modèle sauvegardé: {path}")
    
    def load(self, path: Path) -> None:
        """Charger un modèle pré-entraîné."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Modèle chargé: {path}")

def load_security_model(model_path: Path) -> SecurityModel:
    """Charger le modèle de sécurité."""
    model = SecurityModel()
    
    if model_path.exists():
        model.load(model_path)
    else:
        logger.warning(f"Modèle non trouvé à {model_path}, création d'un nouveau")
    
    return model