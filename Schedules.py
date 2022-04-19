#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np

#template taken from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
class LRCosineDecay():
    '''
    A simple wrapper class for learning rate scheduling and updating parameters with SGD
    '''

    def __init__(self, optimizer, total_steps, lr_max = .2, lr_min = .001, n_warmup_steps = 10):
        self._optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.n_warmup_steps = n_warmup_steps
        self.total_steps = total_steps - n_warmup_steps
        self.n_steps = 0
        self.warmup = [self.lr_max / n_warmup_steps * i for i in range(1, n_warmup_steps + 1)]
        


    def step(self):
        '''
        Step with the inner optimizer
        '''
        self._update_learning_rate()
        self._optimizer.step()


    def _get_lr_scale(self):
        if self.n_steps < self.n_warmup_steps:
            lr = self.warmup[self.n_steps]
        else:
            current_steps = self.n_steps - self.n_warmup_steps
            lr = self.lr_min + .5*(self.lr_max - self.lr_min)*(1 + np.cos(np.pi * current_steps / self.total_steps))
        return lr


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        
        lr = self._get_lr_scale()
        self.n_steps += 1
        

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


# In[15]:


class EMACosineDecay():
    '''
    Uses the BYOL EMA decay given in the paper
    '''
    def __init__(self, total_steps, tau_base):
        self.n_steps = 0
        self.max_steps = total_steps
        self.tau_base = tau_base
        self.current_tau = tau_base
        
    def get_tau(self):
        self.current_tau = 1 - (1 - self.tau_base)*(np.cos(np.pi * self.n_steps / self.max_steps) + 1) / 2
        self.n_steps += 1
        if self.current_tau > 1: self.current_tau = 1
        return self.current_tau

