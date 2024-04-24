from collections import OrderedDict
import torch
from torch import nn
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')

'''
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
'''

class SaveHookOutput:
    def __init__(self):
        #self.outputs = []
        self.outputs = OrderedDict()

    def __call__(self, module, module_in, module_out):
        #self.outputs.append(module_out)
        self.outputs[module.__name__] = module_out

    def clear(self):
        self.outputs = None


def register_hook_vae(
        encoder: nn.Module,
        decoder: nn.Module
):
    track_features = SaveHookOutput()

    for name, layer in encoder.named_children():
        layer.__name__ = name
        layer.register_forward_hook(
            track_features.__call__
        )
    for name, layer in decoder.named_children():
        layer.__name__ = name
        layer.register_forward_hook(
            track_features.__call__
        )

    #return track_features
    return encoder, decoder, track_features


#https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.epoch_best = None

    def __call__(self, val_loss, epoch, model, optimizer):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, optimizer)

        #elif score < self.best_score + self.delta:
        elif score < self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, optimizer)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        print(f'=======> val_loss input: {val_loss:.3f}')
        print(f'=======> best_score: {self.best_score:.3f}')


    def save_checkpoint(self, val_loss, epoch, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        checkpoint_best = { 'epoch': epoch, 
			    'model_state_dict': model.state_dict(),
			    'optimizer_state_dict': optimizer.state_dict() }

        torch.save(checkpoint_best, self.path)
        self.val_loss_min = val_loss
        self.epoch_best = epoch
