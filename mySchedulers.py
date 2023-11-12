
import math
import torch
import logging

from speechbrain.utils import checkpoints
from torch import nn

logger = logging.getLogger(__name__)




class MyIntervalScheduler:
    
    def __init__(
        self,
        lr_initial, # warmup 거쳐서 이 lr 로 도달할 것
        n_warmup_steps,
        anneal_steps, # lr annealing 할 step
        anneal_rates, # 어떤 rate로 줄일지
        model_size=None, # 12 layer
    ):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.current_lr = lr_initial
        self.losses = []
        self.n_steps = 0
        self.normalize = n_warmup_steps ** 0.5
        self.anneal_steps = anneal_steps
        self.anneal_rates = anneal_rates
        if model_size is not None:
            self.normalize = model_size ** (-0.5)

    def __call__(self, optimizer_steps, opt):
        """
        Arguments
        ---------
        opt : optimizer
            The optimizer to update using this scheduler.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        lr : float
            The learning rate after the update.
        """
        
        self.n_steps = optimizer_steps
        current_lr = opt.param_groups[0]["lr"]
        if self.n_steps <= self.n_warmup_steps:
            lr = self.lr_initial * self._get_lr_scale()
        else:
            lr = current_lr * self._get_lr_scale()
            
        # Changing the learning rate within the optimizer
        #for param_group in opt.param_groups:
        #    param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        
        if n_steps <= n_warmup_steps:
            lr_scale = self.normalize * min(
                n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
            )
        else:
            lr_scale = 1
        
        #for i in reversed(range(len(self.anneal_steps))):
        #    if self.n_steps > self.anneal_steps[i]:
        #        lr_scale = lr_scale * self.anneal_rates[i]
        #        break
            
        if self.n_steps > self.anneal_steps:
            lr_scale = lr_scale * self.anneal_rates
        
        return lr_scale
    
    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]
        


"""

from speechbrain.nnet.linear import Linear
import torch
#from speechbrain.nnet.schedulers import NoamIntervalScheduler

inp_tensor = torch.rand([1,660,3])
model = Linear(input_size=3, n_neurons=4)
optim = torch.optim.Adam(model.parameters(), lr=1)
output = model(inp_tensor)
scheduler = MyIntervalScheduler( lr_initial=optim.param_groups[0]["lr"], 
                                  n_warmup_steps=3, anneal_steps=[6, 9],
                                  anneal_rates=[0.5, 1.1],
                                  )
for i in range(15):
    curr_lr,next_lr=scheduler(optim)
    print(i+1, " : ", optim.param_groups[0]["lr"])

"""