from __future__ import annotations
from dataclasses import dataclass
from typing import Final, Optional
import pandas as pd
import numpy as np

@dataclass(slots=True)
class SecurityFeatureConfig:
    """Configuration simple pour les règles 'anormal'."""
    max_mean_latency_ms: int = 500
    max_5xx_errors: int = 10
    min_requests_per_day: int = 1000
    max_p95_latency_ms: int = 1000

CONFIG: Final[SecurityFeatureConfig] = SecurityFeatureConfig()

def build_daily_security_features(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Agréger les logs au niveau journalier.
    
    Args:
        raw_logs: DataFrame contenant au moins les colonnes
            - timestamp (datetime)
            - status_code (int)
            - latency_ms (float)
    
    Returns:
        DataFrame indexé par date avec des indicateurs de sécurité.
    """
    if raw_logs.empty:
        raise ValueError("Le DataFrame de logs est vide")
    
    df = raw_logs.copy()
    
    # Assurer les types de colonnes
    required_columns = ["timestamp", "status_code", "latency_ms"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante: {col}")
    
    # Convertir en datetime si ce n'est pas déjà fait
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    
    # Agrégations journalières
    grouped = (
        df.groupby("date")
        .agg(
            nb_requests=("status_code", "count"),
            nb_errors_5xx=("status_code", lambda s: (s >= 500).sum()),
            nb_errors_4xx=("status_code", lambda s: ((s >= 400) & (s < 500)).sum()),
            latency_mean_ms=("latency_ms", "mean"),
            latency_p50_ms=("latency_ms", lambda s: s.quantile(0.50)),
            latency_p95_ms=("latency_ms", lambda s: s.quantile(0.95)),
            latency_p99_ms=("latency_ms", lambda s: s.quantile(0.99)),
            unique_ips=("client_ip", lambda s: s.nunique() if "client_ip" in df.columns else 0),
        )
        .reset_index()
    )
    
    # Calculer des ratios
    grouped["error_5xx_rate"] = grouped["nb_errors_5xx"] / grouped["nb_requests"]
    grouped["error_4xx_rate"] = grouped["nb_errors_4xx"] / grouped["nb_requests"]
    grouped["requests_per_ip"] = grouped["nb_requests"] / grouped["unique_ips"].replace(0, 1)
    
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped.set_index("date")

def add_security_label(
    features: pd.DataFrame,
    config: Optional[SecurityFeatureConfig] = None,
) -> pd.DataFrame:
    """Ajouter une colonne label (0 = normal, 1 = anormal) selon des règles métier."""
    cfg = config or CONFIG
    df = features.copy()
    
    # Règles métier pour détecter les anomalies
    conditions_anormal = (
        (df["latency_mean_ms"] > cfg.max_mean_latency_ms) |
        (df["nb_errors_5xx"] > cfg.max_5xx_errors) |
        (df["nb_requests"] < cfg.min_requests_per_day) |
        (df["latency_p95_ms"] > cfg.max_p95_latency_ms)
    )
    
    df["label"] = conditions_anormal.astype(int)
    df["confidence"] = np.where(
        conditions_anormal,
        (df["latency_mean_ms"] / cfg.max_mean_latency_ms).clip(0, 1),
        0
    )
    
    return df

def detect_anomalies_in_realtime(
    current_stats: dict,
    historical_mean: pd.DataFrame,
    threshold_std: float = 3.0
) -> dict:
    """Détecter des anomalies en temps réel par rapport à l'historique.
    
    Args:
        current_stats: Statistiques courantes
        historical_mean: Moyennes historiques
        threshold_std: Seuil en écarts-types
    
    Returns:
        Dictionnaire avec les anomalies détectées
    """
    anomalies = {}
    
    for metric in ["latency_mean_ms", "nb_errors_5xx", "nb_requests"]:
        if metric in current_stats and metric in historical_mean.columns:
            current_val = current_stats[metric]
            hist_mean = historical_mean[metric].mean()
            hist_std = historical_mean[metric].std()
            
            if hist_std > 0:
                z_score = abs((current_val - hist_mean) / hist_std)
                if z_score > threshold_std:
                    anomalies[metric] = {
                        "value": current_val,
                        "z_score": z_score,
                        "threshold": threshold_std
                    }
    
    return anomalies