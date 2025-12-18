from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Formatter qui sérialise les logs au format JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Champs supplémentaires passés via extra=...
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in logging.LogRecord.__dict__
        }
        if extras:
            log_data["extra"] = extras
        
        return json.dumps(log_data)

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Retourner un logger configuré avec sortie JSON sur stdout."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        # Déjà configuré (évite les doublons en FastAPI)
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger