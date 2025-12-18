from __future__ import annotations
import time
from datetime import datetime
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.config import Settings
from src.logger import get_logger
from src.model import predict as model_predict, train_model
from numpy import np

app = FastAPI(title="DS API", version="1.0.0")
logger = get_logger(__name__)
settings = Settings.from_env()

# Charger le modèle (version simplifiée)
try:
    # Ici, vous chargeriez votre modèle réel
    model = None  # À remplacer par load_model()
    logger.info("Modèle chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    model = None

class PredictionRequest(BaseModel):
    """Payload générique pour un modèle tabulaire."""
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    timestamp: datetime

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware pour logguer chaque requête HTTP."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    
    logger.info(
        "HTTP request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
        client_ip=request.client.host if request.client else "unknown",
    )
    return response

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ds-api",
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check() -> JSONResponse:
    """Readiness check endpoint."""
    try:
        # Vérifier que le modèle est chargé
        if model is None:
            raise ValueError("Modèle non chargé")
        
        # Vérifier la connexion à la base de données (exemple)
        # db_check = check_database_connection()
        
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "checks": ["model", "api"]}
        )
    except Exception as exc:
        logger.error("Readiness check failed", error=str(exc))
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(exc)},
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Prédiction générique à partir d'un vecteur de features."""
    try:
        logger.info("Prediction requested", features=request.features)
        
        # Simulation de prédiction (à remplacer)
        if model is None:
            # Pour l'exemple, on retourne une valeur aléatoire
            import random
            prediction = random.random()
        else:
            # prediction = model_predict(model, [request.features])[0]
            prediction = 0.5
        
        logger.info("Prediction successful", prediction=float(prediction))
        
        return PredictionResponse(
            prediction=float(prediction),
            timestamp=datetime.utcnow(),
        )
    except Exception as exc:
        logger.error("Prediction failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

@app.get("/metrics")
async def metrics():
    """Endpoint pour les métriques Prometheus."""
    # À implémenter avec prometheus-client
    return {"message": "Metrics endpoint - à implémenter"}


# Ajoutez ces imports en haut du fichier
from src.security_model import load_security_model
from src.security_features import build_daily_security_features, detect_anomalies_in_realtime
import pandas as pd

# Ajoutez ce code après l'initialisation de l'app
security_model = load_security_model(settings.MODEL_PATH)

# Ajoutez ces classes Pydantic
class SecurityFeatures(BaseModel):
    nb_requests: int
    nb_errors_5xx: int
    nb_errors_4xx: int
    latency_mean_ms: float
    latency_p95_ms: float
    latency_p99_ms: Optional[float] = None
    unique_ips: Optional[int] = None

class SecurityPredictionResponse(BaseModel):
    prob_attack: float
    is_anomalous: bool
    level: str
    confidence: float
    features_used: List[str]
    timestamp: datetime

# Ajoutez cet endpoint
@app.post("/predict_security", response_model=SecurityPredictionResponse)
async def predict_security(features: SecurityFeatures) -> SecurityPredictionResponse:
    """Prédire si un jour est normal ou anormal côté sécurité."""
    try:
        logger.info("Security prediction requested", **features.dict())
        
        # Préparer les features pour le modèle
        features_dict = features.dict(exclude_none=True)
        feature_vector = np.array([
            features_dict["nb_requests"],
            features_dict["nb_errors_5xx"],
            features_dict["nb_errors_4xx"],
            features_dict["latency_mean_ms"],
            features_dict["latency_p95_ms"]
        ]).reshape(1, -1)
        
        # Prédiction
        scores = security_model.predict_proba(feature_vector)[0]
        prob_attack = float(scores[1])
        
        # Déterminer le niveau de sévérité
        is_anomalous = prob_attack >= settings.SECURITY_THRESHOLD
        
        if prob_attack >= settings.CRITICAL_THRESHOLD:
            level = "critique"
        elif prob_attack >= settings.SECURITY_THRESHOLD:
            level = "suspect"
        else:
            level = "normal"
        
        # Log et réponse
        logger.info(
            "Security prediction done",
            prob_attack=prob_attack,
            is_anomalous=is_anomalous,
            level=level
        )
        
        return SecurityPredictionResponse(
            prob_attack=prob_attack,
            is_anomalous=is_anomalous,
            level=level,
            confidence=prob_attack,
            features_used=["nb_requests", "nb_errors_5xx", "nb_errors_4xx", 
                          "latency_mean_ms", "latency_p95_ms"],
            timestamp=datetime.utcnow(),
        )
        
    except Exception as exc:
        logger.error("Security prediction failed", error=str(exc))
        raise HTTPException(status_code=500, 
                          detail=f"Security prediction failed: {str(exc)}") from exc

# Ajoutez cet endpoint pour analyser des logs bruts
@app.post("/analyze_logs")
async def analyze_logs(logs: List[dict]):
    """Analyser des logs bruts pour détecter des anomalies."""
    try:
        df = pd.DataFrame(logs)
        
        # Construire les features
        daily_features = build_daily_security_features(df)
        
        # Prédire pour chaque jour
        predictions = []
        for idx, row in daily_features.iterrows():
            pred = security_model.predict_proba(row.values.reshape(1, -1))[0]
            predictions.append({
                "date": idx.date().isoformat(),
                "prob_attack": float(pred[1]),
                "features": row.to_dict()
            })
        
        return {
            "days_analyzed": len(predictions),
            "anomalous_days": sum(1 for p in predictions 
                                 if p["prob_attack"] >= settings.SECURITY_THRESHOLD),
            "predictions": predictions
        }
        
    except Exception as exc:
        logger.error("Log analysis failed", error=str(exc))
        raise HTTPException(status_code=500, 
                          detail=f"Log analysis failed: {str(exc)}") from exc


@app.get("/")
async def root():
    """Page d'accueil."""
    return {
        "message": "DS API DevSecOps",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/ready",
            "/predict",
            "/docs",
            "/metrics"
        ]
    }
