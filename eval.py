import torch
import numpy as np
from omegaconf import OmegaConf
from model.DRDiff_model import DRDiffTrainer

if __name__ == "__main__":
    path = r".\config\config.yaml"
    configs = OmegaConf.load(path)

    torch.manual_seed(configs['seed'])
    torch.cuda.manual_seed_all(configs['seed'])
    torch.cuda.manual_seed(configs['seed'])
    np.random.seed(configs['seed'])

    Trainer = DRDiffTrainer(configs=configs)
    Trainer.evaluate()
    # Trainer.evaluate_frequency()