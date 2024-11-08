import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any, Union
import json
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

class AnalysisConfig:
    """Configuration manager for analysis"""
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_paths()
        self._setup_visualization()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_paths(self) -> None:
        """Create necessary directories"""
        for category in ['data', 'results']:
            for path in self.config['paths'][category].values():
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def _setup_visualization(self) -> None:
        """Configure visualization settings"""
        plt.style.use(self.config['visualization']['style'])
        sns.set_palette(self.config['visualization']['color_palette'])
        pio.templates.default = "plotly_white"

class ResultsManager:
    """Manager for analysis results"""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results_path = Path(config.config['paths']['results']['reports'])
    
    def save_results(self, results: Dict[str, Any], name: str) -> None:
        """Save analysis results"""
        file_path = self.results_path / f"{name}.json"
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
    
    def load_results(self, name: str) -> Dict[str, Any]:
        """Load analysis results"""
        file_path = self.results_path / f"{name}.json"
        with open(file_path, 'r') as f:
            return json.load(f)

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('analysis.log')
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add formatters to handlers
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger 