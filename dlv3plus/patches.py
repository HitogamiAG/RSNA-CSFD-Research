from segmentation_models_pytorch import utils as smp_utils
import wandb

def _format_logs(self, logs):
    to_wandb = logs.copy()
    to_wandb = {f"{self.stage_name}_{key}" : value for key, value in to_wandb.items()}
    if self.stage_name == 'train':
        to_wandb['lr'] = self.optimizer.param_groups[0]['lr']
    wandb.log(to_wandb)
    
    str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
    s = ", ".join(str_logs)
    return s

class TrainEpoch(smp_utils.train.Epoch):
    def __init__(self, model, loss, metrics, optimizer, loader_size, scheduler, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader_size = loader_size
        self.batch_counter = 1

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(
            (self.batch_counter // self.loader_size) + ((self.batch_counter % self.loader_size) / self.loader_size)
        )
        
        self.batch_counter += 1
        return loss, prediction