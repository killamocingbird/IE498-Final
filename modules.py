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


# https://github.com/muggin/show-and-tell/blob/master/models.py

class CNN(Foundation):
    """Class to build new model including all but last layers"""
    def __init__(self, output_dim=1000):
        super(CNN, self).__init__()
        # TODO: change with resnet152?
        pretrained_model = models.resnet34(pretrained=True)
        self.resnet = Sequential(*list(pretrained_model.children())[:-1])
        self.linear = nn.Linear(pretrained_model.fc.in_features, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        # weight init, inspired by tutorial
        self.linear.weight.data.normal_(0,0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.resnet(x)
        x = Variable(x.data)
        x = x.view(x.size(0), -1) # flatten
        x = self.linear(x)

        return x

class RNN(Foundation):
    """
    Recurrent Neural Network for Text Generation.
    To be used as part of an Encoder-Decoder network for Image Captioning.
    """
    __rec_units = {
        'elman': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM }

    def __init__(self, emb_size, hidden_size, vocab_size, num_layers=1, rec_unit='gru'):
        """
        Initializer
        :param embed_size: size of word embeddings
        :param hidden_size: size of hidden state of the recurrent unit
        :param vocab_size: size of the vocabulary (output of the network)
        :param num_layers: number of recurrent layers (default=1)
        :param rec_unit: type of recurrent unit (default=gru)
        """
        rec_unit = rec_unit.lower()
        assert rec_unit in RNN.__rec_units, 'Specified recurrent unit is not available'

        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.unit = RNN.__rec_units[rec_unit](emb_size, hidden_size, num_layers,
                                                 batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        """
        Forward pass through the network
        :param features: features from CNN feature extractor
        :param captions: encoded and padded (target) image captions
        :param lengths: actual lengths of image captions
        :returns: predicted distributions over the vocabulary
        """
        # embed tokens in vector space
        embeddings = self.embeddings(captions)

        # append image as first input
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)

        # pack data (prepare it for pytorch model)
        inputs_packed = pack_padded_sequence(inputs, lengths, batch_first=True)

        # run data through recurrent network
        hiddens, _ = self.unit(inputs_packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, max_len=25):
        """
        Sample from Recurrent network using greedy decoding
        :param features: features from CNN feature extractor
        :returns: predicted image captions
        """
        output_ids = []
        states = None
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # pass data through recurrent network
            hiddens, states = self.unit(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # find maximal predictions
            predicted = outputs.max(1)[1]

            # append results from given step to global results
            output_ids.append(predicted)

            # prepare chosen words for next decoding step
            inputs = self.embeddings(predicted)
            inputs = inputs.unsqueeze(1)
        output_ids = torch.stack(output_ids, 1)
        return output_ids.squeeze()

cnn = CNN()