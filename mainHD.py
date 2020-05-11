# Library imports
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from torchvision import models, transforms

# Custom imports
import AdamHD
import models as m
from vocab import Vocabulary, load_vocab
from data_loader import get_coco_data_loader
import utils as u


torch.manual_seed(0)

# Header for saving files
header = 'ModelHyper_'

# Hyperparameters for training
<<<<<<< HEAD
batch_size = 1
checkpoint = header+'checkpoint.pth'
=======
batch_size = 128
>>>>>>> 5a562b6407f8a537aaed92614bd31aa968c935a6
checkpoint = None
criteria = nn.CrossEntropyLoss()
criteria = nn.MSELoss()
debug = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 1000
lr = 1e-1
verbose = 1

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
model = m.ShowTell(embed_size = 512, 
                    rnn_hidden_size = 512, 
                    vocab = vocab, 
                    rnn_layers = 1).to(device)
# Declare optimizer
optimizer = AdamHD.AdamHD(model.parameters(), lr=lr, hypergrad_lr=1e-8)

<<<<<<< HEAD



=======
>>>>>>> 5a562b6407f8a537aaed92614bd31aa968c935a6
# Load in checkpoint to continue training if applicable
if checkpoint is not None:
    u.b_print("Loading checkpoint")
    checkpoint = torch.load(checkpoint)
    model.load(checkpoint)
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

# Declare scheduler
scheduler = MultiStepLR(optimizer, [1, 2, 10])

# Book keep lowest lost for early stopping
min_loss = 1e8
u.b_print("Optimizing %d parameters on %s" % (u.count_parameters(model), device))
for epoch in range(epochs):
    # Book keeping
    running_loss = 0
    model.train()
    for i, (image, caption, lengths) in enumerate(train_loader):
        # Cast to device
        image = image.to(device)
        caption = caption.to(device)
        if debug:
            lengths[0] = 1
            caption = caption[:,:1]
            print(caption.shape)
        
        # Image shape:   [batch, 3, x, y], 
        # Caption shape: [batch, tokens]
        pred = model(image, caption, lengths)
        # Pred shape: [sum(lengths), vocab size]
        
<<<<<<< HEAD
        # Softmax probabilities
        #pred = torch.softmax(pred, 1)
        print()
        print("presigmoid, ", pred)
        pred = torch.sigmoid(pred)

=======
>>>>>>> 5a562b6407f8a537aaed92614bd31aa968c935a6
        labels = pack_padded_sequence(caption, lengths, batch_first = True)[0]

        print("prediction, ", pred)
        labels = pred.clone().detach() * 0.
        labels[0][0] = 1.
        print("labels,     ", labels)
        print()

        loss = criteria(pred, labels)
        
        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if debug:
<<<<<<< HEAD
            print("Linear Layer")
            #print(model.RNN.linear.weight.grad)
            #print(model.RNN.linear.weight.grad.shape)
            print(model.RNN.linear.bias.grad)
            print(model.RNN.linear.bias.grad.shape)


            print(2 * (pred - labels)/ model.RNN.linear.bias.grad)
            exit()

            print('RNN')
            u.get_grad_av_mag(model.RNN.parameters())
            print('CNN')
            u.get_grad_av_mag(model.CNN.parameters())
            print()
            u.b_print("Loss: %.8f CNN Grad: %.5f RNN Grad: %.5f"
                      % (loss.item(), u.get_grad_av_mag(model.CNN.parameters()), 10))#u.get_grad_av_mag(model.RNN.parameters())))
=======
            u.b_print("Loss: %.8f | Bias Grad %.10f" % (loss.item(), model.RNN.linear.bias.grad.mean()))
>>>>>>> 5a562b6407f8a537aaed92614bd31aa968c935a6
        
        # Prevent memory leak
        del image, caption, pred, labels, loss
        
    running_loss /= (i + 1)
    
    running_val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (image, caption, lengths) in enumerate(tqdm(val_loader)):
            # Cast to device
            image = image.to(device)
            caption = caption.to(device)
            
            pred = model(image, caption, lengths)
            labels = pack_padded_sequence(caption, lengths, batch_first = True)[0]
            
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
        
    # Do output
    if (epoch+1)%verbose==0:
        if debug:
            sys.stderr.write("[%d] train: %.8f val: %.8f" % (epoch+1, running_loss, running_val_loss))
        else:
            u.b_print("[%d] train: %.8f val: %.8f" % (epoch+1, running_loss, running_val_loss))
    
    # Step scheduler
    scheduler.step()
        
