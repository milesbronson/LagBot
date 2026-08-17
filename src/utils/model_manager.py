# src/utils/model_manager.py
from pathlib import Path
from datetime import datetime


class ModelManager:
    """Manages model loading and tracking for trained agents"""
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def get_latest_model(self) -> str:
        """
        Get the most recently saved model
        
        Returns:
            Path to the latest model
        """
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Find all final_model.zip files and their containing directories
        models = list(self.model_dir.glob("*/final_model.zip"))
        
        if not models:
            raise FileNotFoundError("No trained models found in models directory")
        
        # Sort by modification time (most recent last)
        latest = max(models, key=lambda p: p.stat().st_mtime)
        return str(latest)
    