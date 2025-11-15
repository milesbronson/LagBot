# src/utils/model_manager.py
import os
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
    
    def get_model_by_name(self, name: str) -> str:
        """
        Get a model by run name
        
        Args:
            name: Name of the training run (e.g., 'run_20231115_143022')
            
        Returns:
            Path to the model
        """
        model_path = self.model_dir / name / "final_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return str(model_path)
    
    def list_all_models(self) -> list:
        """
        List all available trained models with timestamps
        
        Returns:
            List of tuples (name, path, modification_time)
        """
        models = []
        for model_file in sorted(self.model_dir.glob("*/final_model.zip")):
            run_name = model_file.parent.name
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            models.append((run_name, str(model_file), mtime))
        
        return sorted(models, key=lambda x: x[2], reverse=True)