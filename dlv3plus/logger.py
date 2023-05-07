import dotenv
import wandb

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