# Library imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom imports
import models as m


# Header for saving files
header = 'Model1_'

checkpoint = None
device = 'cuda' if torch.cuda.is_availabe() else 'cpu'
epochs = 100


# Declare model
model = m.TemplateModel()
# Declare optimizer
optimizer = None


# Load in checkpoint to continue training if applicable
if checkpoint is not None:
    print("Loading checkpoint")
    checkpoint = torch.load(checkpoint)
    model.load(checkpoint)
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# Book keep lowest lost for early stopping
min_loss = 1e8
for epoch in range(len(epochs)):
    
    # Do training loop
    
    loss = 0
    val_loss = 0
    
    # Update the loss for plotting and book keeping
    model.update_loss(loss, val_loss=val_loss)
    
    if loss < min_loss:
        min_loss = loss
        # Save model if it achieves lowest lost alone with optimizer
        model.save(header=header, optimizer=optimizer)
        
