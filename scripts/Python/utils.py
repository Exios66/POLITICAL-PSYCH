import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional
import json
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

class AnalysisConfig:
    """Configuration manager for analysis settings and paths"""
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to YAML config file
            
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If config file has invalid YAML
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()
        self._setup_visualization()
        
        # Setup logging
        self.logger = setup_logging(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse configuration from YAML file"""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config file not found at {self.config_path}")
                
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if not config:
                raise ValueError("Config file is empty")
                
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file: {str(e)}")
            
    def _validate_config(self) -> None:
        """Validate required config sections and fields exist"""
        required_sections = ['paths', 'visualization']
        required_paths = ['data', 'results'] 
        
        for section in required_sections:
            if section not in self.config:
                raise KeyError(f"Missing required config section: {section}")
                
        for category in required_paths:
            if category not in self.config['paths']:
                raise KeyError(f"Missing required paths category: {category}")
    
    def _setup_paths(self) -> None:
        """Create necessary directories if they don't exist"""
        try:
            for category in ['data', 'results']:
                for path in self.config['paths'][category].values():
                    path_obj = Path(path)
                    path_obj.mkdir(parents=True, exist_ok=True)
                    
                    if not os.access(path_obj, os.W_OK):
                        raise PermissionError(f"No write access to directory: {path_obj}")
                        
        except Exception as e:
            raise RuntimeError(f"Error setting up directories: {str(e)}")
    
    def _setup_visualization(self) -> None:
        """Configure visualization settings with error handling"""
        try:
            style = self.config['visualization'].get('style', 'default')
            palette = self.config['visualization'].get('color_palette', 'deep')
            
            plt.style.use(style)
            sns.set_palette(palette)
            pio.templates.default = "plotly_white"
            
        except Exception as e:
            self.logger.error(f"Error setting up visualizations: {str(e)}")
            # Fall back to defaults
            plt.style.use('default')
            sns.set_palette('deep')

class ResultsManager:
    """Manager for saving and loading analysis results"""
    def __init__(self, config: AnalysisConfig):
        """
        Initialize results manager
        
        Args:
            config: AnalysisConfig instance
        """
        self.config = config
        self.results_path = Path(config.config['paths']['results']['reports'])
        self.logger = setup_logging(__name__)
        
    def save_results(self, results: Dict[str, Any], name: str, 
                    timestamp: bool = True) -> None:
        """
        Save analysis results to JSON
        
        Args:
            results: Dictionary of results to save
            name: Base filename for results
            timestamp: Whether to append timestamp to filename
            
        Raises:
            IOError: If unable to write results file
        """
        try:
            if timestamp:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp_str}.json"
            else:
                filename = f"{name}.json"
                
            file_path = self.results_path / filename
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
                
            self.logger.info(f"Results saved to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise IOError(f"Failed to save results: {str(e)}")
    
    def load_results(self, name: str) -> Dict[str, Any]:
        """
        Load analysis results from JSON
        
        Args:
            name: Results filename (without .json extension)
            
        Returns:
            Dictionary of loaded results
            
        Raises:
            FileNotFoundError: If results file not found
            json.JSONDecodeError: If invalid JSON
        """
        try:
            file_path = self.results_path / f"{name}.json"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Results file not found: {file_path}")
                
            with open(file_path, 'r') as f:
                results = json.load(f)
                
            self.logger.info(f"Results loaded from {file_path}")
            return results
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing results JSON: {str(e)}")
            raise
            
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            raise

def setup_logging(name: str, level: int = logging.INFO,
                 log_file: Optional[str] = 'analysis.log') -> logging.Logger:
    """
    Setup logging configuration with console and file handlers
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (None for no file logging)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            f_handler = logging.FileHandler(log_file)
            f_handler.setLevel(level)
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
            
        except Exception as e:
            logger.error(f"Failed to setup file logging: {str(e)}")
            
    return logger