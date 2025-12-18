from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional
from dotenv import load_dotenv

# Nom du fichier d'environnement
ENV_FILE: Final[str] = ".env"

@dataclass(slots=True)
class Settings:
    """Configuration centralisée de l'application."""
    
    # Variables d'environnement principales
    DATABASE_URL: str
    API_KEY: Optional[str]
    MODEL_PATH: Path
    LOG_LEVEL: str
    
    # Seuils pour la détection d'anomalies
    SECURITY_THRESHOLD: float
    CRITICAL_THRESHOLD: float
    
    # Configuration API
    API_HOST: str
    API_PORT: int
    API_WORKERS: int
    
    # Configuration base de données
    DB_POOL_SIZE: int
    DB_MAX_OVERFLOW: int
    DB_POOL_RECYCLE: int
    
    # Configuration logging
    LOG_FORMAT: str
    LOG_FILE: Optional[Path]
    
    # Configuration monitoring
    METRICS_ENABLED: bool
    HEALTH_CHECK_INTERVAL: int
    
    # Configuration sécurité
    ALLOWED_ORIGINS: list[str]
    RATE_LIMIT_REQUESTS: int
    RATE_LIMIT_PERIOD: int
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Construire un objet Settings à partir des variables d'environnement."""
        # Charger les variables d'environnement depuis le fichier .env
        load_dotenv(ENV_FILE)
        
        # Chemin du modèle avec valeur par défaut
        model_path = Path(os.getenv("MODEL_PATH", "./models/model_latest.pkl"))
        
        # Liste des origines autorisées (CORS)
        allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000")
        allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]
        
        # Création de l'instance Settings
        settings = cls(
            DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///./app.db"),
            API_KEY=os.getenv("API_KEY"),
            MODEL_PATH=model_path,
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO").upper(),
            SECURITY_THRESHOLD=float(os.getenv("SECURITY_THRESHOLD", "0.5")),
            CRITICAL_THRESHOLD=float(os.getenv("CRITICAL_THRESHOLD", "0.8")),
            API_HOST=os.getenv("API_HOST", "0.0.0.0"),
            API_PORT=int(os.getenv("API_PORT", "8000")),
            API_WORKERS=int(os.getenv("API_WORKERS", "1")),
            DB_POOL_SIZE=int(os.getenv("DB_POOL_SIZE", "5")),
            DB_MAX_OVERFLOW=int(os.getenv("DB_MAX_OVERFLOW", "10")),
            DB_POOL_RECYCLE=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            LOG_FORMAT=os.getenv("LOG_FORMAT", "JSON"),
            LOG_FILE=Path(os.getenv("LOG_FILE")) if os.getenv("LOG_FILE") else None,
            METRICS_ENABLED=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            HEALTH_CHECK_INTERVAL=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            ALLOWED_ORIGINS=allowed_origins,
            RATE_LIMIT_REQUESTS=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            RATE_LIMIT_PERIOD=int(os.getenv("RATE_LIMIT_PERIOD", "60")),
        )
        
        # Validation de la configuration au démarrage
        settings._validate()
        
        return settings
    
    def _validate(self) -> None:
        """Valider la configuration au démarrage (fail fast)."""
        
        # Validation des chemins de fichiers
        if self.MODEL_PATH.suffix not in [".pkl", ".h5", ".joblib"]:
            raise ValueError(f"Format de modèle non supporté: {self.MODEL_PATH.suffix}")
        
        if self.LOG_FILE and not self.LOG_FILE.parent.exists():
            raise FileNotFoundError(f"Répertoire du fichier de log introuvable: {self.LOG_FILE.parent}")
        
        # Validation des seuils
        if not 0.0 <= self.SECURITY_THRESHOLD <= 1.0:
            raise ValueError(f"SECURITY_THRESHOLD doit être entre 0 et 1: {self.SECURITY_THRESHOLD}")
        
        if not 0.0 <= self.CRITICAL_THRESHOLD <= 1.0:
            raise ValueError(f"CRITICAL_THRESHOLD doit être entre 0 et 1: {self.CRITICAL_THRESHOLD}")
        
        if self.CRITICAL_THRESHOLD <= self.SECURITY_THRESHOLD:
            raise ValueError(
                f"CRITICAL_THRESHOLD ({self.CRITICAL_THRESHOLD}) doit être "
                f"supérieur à SECURITY_THRESHOLD ({self.SECURITY_THRESHOLD})"
            )
        
        # Validation des ports
        if not 1 <= self.API_PORT <= 65535:
            raise ValueError(f"Port API invalide: {self.API_PORT}")
        
        # Validation des limites
        if self.RATE_LIMIT_REQUESTS <= 0:
            raise ValueError(f"RATE_LIMIT_REQUESTS doit être positif: {self.RATE_LIMIT_REQUESTS}")
        
        if self.RATE_LIMIT_PERIOD <= 0:
            raise ValueError(f"RATE_LIMIT_PERIOD doit être positif: {self.RATE_LIMIT_PERIOD}")
        
        # Validation des niveaux de log
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.LOG_LEVEL not in valid_log_levels:
            raise ValueError(f"Niveau de log invalide: {self.LOG_LEVEL}. Valide: {valid_log_levels}")
        
        # Validation des formats de log
        valid_log_formats = ["JSON", "TEXT"]
        if self.LOG_FORMAT not in valid_log_formats:
            raise ValueError(f"Format de log invalide: {self.LOG_FORMAT}. Valide: {valid_log_formats}")
        
        # Vérification des ressources
        if not self.MODEL_PATH.exists() and self.MODEL_PATH.name != "model_latest.pkl":
            raise FileNotFoundError(f"Modèle introuvable: {self.MODEL_PATH}")
        
        # Avertissements (ne pas échouer)
        if self.API_KEY is None:
            print("⚠️  AVERTISSEMENT: API_KEY non définie. Certaines fonctionnalités peuvent être limitées.")
        
        if "sqlite" in self.DATABASE_URL and "memory" not in self.DATABASE_URL:
            print("⚠️  AVERTISSEMENT: Utilisation de SQLite en production n'est pas recommandée.")
    
    def get_database_config(self) -> dict:
        """Retourner la configuration de la base de données."""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DB_POOL_SIZE,
            "max_overflow": self.DB_MAX_OVERFLOW,
            "pool_recycle": self.DB_POOL_RECYCLE,
            "echo": self.LOG_LEVEL == "DEBUG"
        }
    
    def get_api_config(self) -> dict:
        """Retourner la configuration de l'API."""
        return {
            "host": self.API_HOST,
            "port": self.API_PORT,
            "workers": self.API_WORKERS,
            "reload": self.LOG_LEVEL == "DEBUG"
        }
    
    def get_logging_config(self) -> dict:
        """Retourner la configuration du logging."""
        config = {
            "level": self.LOG_LEVEL,
            "format": self.LOG_FORMAT,
            "file": str(self.LOG_FILE) if self.LOG_FILE else None
        }
        
        # Configuration supplémentaire pour différents environnements
        if os.getenv("ENVIRONMENT") == "production":
            config["rotation"] = "500 MB"
            config["retention"] = "10 days"
            config["compression"] = "gz"
        
        return config
    
    def get_security_config(self) -> dict:
        """Retourner la configuration de sécurité."""
        return {
            "allowed_origins": self.ALLOWED_ORIGINS,
            "rate_limit_requests": self.RATE_LIMIT_REQUESTS,
            "rate_limit_period": self.RATE_LIMIT_PERIOD,
            "security_threshold": self.SECURITY_THRESHOLD,
            "critical_threshold": self.CRITICAL_THRESHOLD
        }
    
    def __str__(self) -> str:
        """Représentation lisible de la configuration (sans les secrets)."""
        return f"""
Configuration de l'application:
-------------------------------
API:
  Hôte: {self.API_HOST}:{self.API_PORT}
  Workers: {self.API_WORKERS}
  
Base de données:
  URL: {'*' * 20 if 'password' in self.DATABASE_URL else self.DATABASE_URL}
  Pool: {self.DB_POOL_SIZE}/{self.DB_MAX_OVERFLOW}
  
Logging:
  Niveau: {self.LOG_LEVEL}
  Format: {self.LOG_FORMAT}
  Fichier: {self.LOG_FILE or 'stdout'}
  
Modèle:
  Chemin: {self.MODEL_PATH}
  Existe: {self.MODEL_PATH.exists()}
  
Sécurité:
  Seuil normal: {self.SECURITY_THRESHOLD}
  Seuil critique: {self.CRITICAL_THRESHOLD}
  Origines autorisées: {self.ALLOWED_ORIGINS}
  Rate limit: {self.RATE_LIMIT_REQUESTS}/{self.RATE_LIMIT_PERIOD}s
  
Monitoring:
  Métriques: {'Activé' if self.METRICS_ENABLED else 'Désactivé'}
  Health check: {self.HEALTH_CHECK_INTERVAL}s
        """
    
    def to_dict(self, hide_secrets: bool = True) -> dict:
        """Convertir la configuration en dictionnaire."""
        config_dict = {
            "DATABASE_URL": self.DATABASE_URL if not hide_secrets else "********",
            "API_KEY": self.API_KEY if not hide_secrets and self.API_KEY else "********",
            "MODEL_PATH": str(self.MODEL_PATH),
            "LOG_LEVEL": self.LOG_LEVEL,
            "SECURITY_THRESHOLD": self.SECURITY_THRESHOLD,
            "CRITICAL_THRESHOLD": self.CRITICAL_THRESHOLD,
            "API_HOST": self.API_HOST,
            "API_PORT": self.API_PORT,
            "API_WORKERS": self.API_WORKERS,
            "DB_POOL_SIZE": self.DB_POOL_SIZE,
            "DB_MAX_OVERFLOW": self.DB_MAX_OVERFLOW,
            "DB_POOL_RECYCLE": self.DB_POOL_RECYCLE,
            "LOG_FORMAT": self.LOG_FORMAT,
            "LOG_FILE": str(self.LOG_FILE) if self.LOG_FILE else None,
            "METRICS_ENABLED": self.METRICS_ENABLED,
            "HEALTH_CHECK_INTERVAL": self.HEALTH_CHECK_INTERVAL,
            "ALLOWED_ORIGINS": self.ALLOWED_ORIGINS,
            "RATE_LIMIT_REQUESTS": self.RATE_LIMIT_REQUESTS,
            "RATE_LIMIT_PERIOD": self.RATE_LIMIT_PERIOD,
        }
        
        return config_dict


# Instance singleton de configuration
try:
    settings = Settings.from_env()
except Exception as e:
    print(f"❌ Erreur lors du chargement de la configuration: {e}")
    print("💡 Assurez-vous d'avoir un fichier .env ou de définir les variables d'environnement")
    raise


# Fonction utilitaire pour charger dynamiquement la configuration
def get_settings() -> Settings:
    """Retourner l'instance de configuration (pour l'injection de dépendances)."""
    return settings


if __name__ == "__main__":
    # Test de la configuration
    print(settings)
    
    # Afficher la configuration en dict (caché)
    import json
    print("\nConfiguration (sans secrets):")
    print(json.dumps(settings.to_dict(), indent=2))
    
    # Tester la validation
    print("\n✅ Configuration chargée avec succès")