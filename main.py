# Library imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Custom imports
import models as m


# Header for saving files
header = 'Model1_'

# Hyperparameters for training
batch_size = 64
checkpoint = None
criteria = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_availabe() else 'cpu'
epochs = 100
lr = 1e-3

# Get dataset

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# load COCOs dataset
IMAGES_PATH = 'data/train2014'
CAPTION_FILE_PATH = 'data/annotations/captions_train2014.json'

vocab = load_vocab()
train_loader = get_coco_data_loader(path=IMAGES_PATH,
                                    json=CAPTION_FILE_PATH,
                                    vocab=vocab,
                                    transform=transform,
                                    batch_size=batch_size,
                                    shuffle=True)

IMAGES_PATH = 'data/val2014'
CAPTION_FILE_PATH = 'data/annotations/captions_val2014.json'
val_loader = get_coco_data_loader(path=IMAGES_PATH,
                                  json=CAPTION_FILE_PATH,
                                  vocab=vocab,
                                  transform=transform,
                                  batch_size=batch_size,
                                  shuffle=True)

# Declare model
model = m.ShowTell()
# Declare optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


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
    # Book keeping
    running_loss = 0
    model.train()
    for i, (image, caption) in enumerate(tqdm(train_loader)):
        """
        Veyr likely to be incorrect
        """
        # Expect shape [batch, 3, x, y], [batch, tokens, vocab_size]
        pred = model(image, caption[:,:-1,:])           # Shape: [batch, tokens-1, vocab_size]
        labels = torch.argmax(caption[:,1:,:], 2)       # Shape: [batch, tokens-1]
        
        loss = criteria(pred, labels)
        
        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Prevent memory leak
        del image, caption, pred, labels, loss
        
    running_loss /= (i + 1)
    
    running_val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (image, caption) in enumerate(tqdm(test_loader)):
            pred = model(image, caption[:,:-1,:])
            labels = torch.argmax(caption[:,1:,:])
            
            loss = criteria(pred, labels)
            
            running_val_loss += loss.item()
            
            # Prevent memory leak
            del image, caption, pred, labels, loss
    
    running_val_loss /= (i + 1)
    
    # Update the loss for plotting and book keeping
    model.update_loss(running_loss, val_loss=running_val_loss)
    
    # Do early stopping on validation loss
    if running_val_loss < min_loss:
        min_loss = running_val_loss
        # Save model if it achieves lowest lost alone with optimizer
        model.save(header=header, optimizer=optimizer)
        
