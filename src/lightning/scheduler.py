from torch.optim.lr_scheduler import _LRScheduler
import warnings

class NoamScheduler(_LRScheduler):

    def __init__(self, optimizer, d_model, warmup, factor, last_epoch=-1, verbose=False):
        self.d_model = d_model
        self.warmup = warmup
        self.factor = factor
        super(NoamScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        lr = self._get_closed_form_lr()
        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.factor * \
            (self.d_model ** (-0.5) *
             min(self._step_count ** (-0.5), self._step_count * self.warmup ** (-1.5)))
