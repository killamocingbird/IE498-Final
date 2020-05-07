import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


"""
Declare all basic blocks here
"""


# Foundation to base all models on
class Foundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
    
    # Tracking losses
    def update_loss(self, train_loss, val_loss=None):
        self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
            
    # Plots saved training curves
    def gen_train_plots(self, save_path='', header=''):
        if len(self.train_loss) < 1:
            raise "No losses to plot"

        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('Training Curve')
        plt.plot([i for i in range(len(self.train_loss))], self.train_loss, linestyle='dashed',
                 label='Training')
        if len(self.val_loss) >= 1:
            plt.plot([i for i in range(len(self.val_loss))], self.val_loss,
                     label='Validation')
        plt.legend()
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.savefig(os.path.join(save_path, header + 'train_curve.png'))
        plt.close()
    
    
    def save(self, save_path='', header='', optimizer=None):
        checkpoint = {
            'state_dict': self.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, save_path + header + 'checkpoint.pth')
        torch.save(self, save_path + header + 'model.pth')
    
    def load(self, checkpoint):
        self.load_state_dict(checkpoint['state_dict'])
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']
