import wandb
import dotenv

import numpy as np

import config

from typing import Dict

def initiate_wandb_run(config: Dict, project_name:str = config.PROJECT_NAME, note:str = '') -> None:
    """Function to start Wandb Experiment

    Args:
        config (Dict): Config to set as wandb config
    """
    dotenv.load_dotenv('.env')
    wandb.login()

    # Initialize new experiment
    wandb.init(
        project = project_name,
        notes = note,
    #          resume = 'must',
    #          id = '1c6ogt8c',
        save_code = True
    )

    # Initialize config for this experiment
    wandb.config = config
    
def initiate_config_only(config: Dict):
    """Function to initialize config

    Args:
        config (Dict): Config to set as wandb config
    """
    dotenv.load_dotenv('.env')
    wandb.login()
    
    wandb.config = config
    
class Logger:
    """Log train metrics and send to Wandb
    """
    def __init__(self, prefix: str, metrics_to_log: Dict[str, float]):
        """Initialize logger

        Args:
            prefix (str): Prefix will be added to each metric name
            metrics_to_log (Dict[str, float]): List of metrics & default value
        """
        self.metrics_to_log = metrics_to_log
        self.prefix = prefix
        self.reset()
        
    def reset(self):
        """Reset counter & metrics results
        """
        self.update_counter = 0
        self.metrics = {key: 0 for key in self.metrics_to_log}
        
    def log(self, metrics: Dict[str, float]):
        """Log metrics

        Args:
            metrics (Dict[str, float]): Metrics
        """
        self.update_counter += 1
        for key in metrics.keys():
            self.metrics[key] += metrics[key]
            
    def send_logs(self):
        """Send logs to Wandb
        """
        if self.update_counter != 0:
            
            self.metrics = {self.prefix + key : (value / self.update_counter) for key, value in self.metrics.items()}
            wandb.log(self.metrics)
            
            self.reset()