from pytorch_lightning.callbacks import Callback

class LogDistributions(Callback):

    def __init__(self, epochs=1, steps=-1):
        self.epochs = epochs
        self.steps = steps

    def on_after_backward(self, trainer, pl_module):
        record = False

        if self.steps > 0:
            record = trainer.global_step % self.steps == 0
        if self.epochs > 0 and trainer.is_last_batch:
            record = trainer.current_epoch % self.epochs == 0

        if record:
            for name, param in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(name, param, trainer.global_step)
                if param.requires_grad:
                    trainer.logger.experiment.add_histogram(f"{name}_grad", param.grad, trainer.global_step)
