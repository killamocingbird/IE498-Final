import modules as m
import utils
import torch
import torch.nn as nn
from torch.nn import Sequential
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

"""
Declare all models here
"""


class ShowTell(m.Foundation):
    def __init__(self, embed_size, rnn_hidden_size, vocab, rnn_layers):
        super(ShowTell, self).__init__()
        """
        type  name                  default val
        (int) embed_size            512
        (int) rnn_hidden_size       512
        (Vocabulary) vocab          "prexisting vocabulary wrapper"
        (int) rnn_layers            1
        """
        self.vocab = vocab
        vocab_size = len(self.vocab)
        self.CNN = CNN(embed_size)
        self.RNN = RNN(embed_size, rnn_hidden_size, vocab_size, rnn_layers)

        self.features = None

    def forward(self, images, captions, lengths):
        features = self.CNN(images)
        outputs = self.RNN(features, captions, lengths)

        self.features = features # saved in case we want to access

        return outputs

    def sample(self, captions = None):
        """
        Returns:
        (str) sentence: The output of the RNN based on the most recently calculated features as a readable sentence
        """

        sampled_idxs = self.RNN.sample(self.features)
        sampled_idxs = sampled_idxs.cpu().data.numpy()[0]
        predicted_sentence = utils.convert_back_to_text(sampled_idxs, self.vocab)

        true_sentence = "<No target sentence provided>"
        try:
            true_idxs = captions.cpu().data.numpy()[0]
            true_sentence = utils.convert_back_to_text(true_idxs, self.vocab)
        except:
            pass

        return (predicted_sentence, true_sentence)


# https://github.com/muggin/show-and-tell/blob/master/models.py

class CNN(nn.Module):
    """Class to build new model including all but last layers"""
    def __init__(self, embed_size):
        super(CNN, self).__init__()
        # TODO: change with resnet152?
        pretrained_model = models.resnet34(pretrained=True)
        self.resnet = Sequential(*list(pretrained_model.children())[:-1])
        self.linear = nn.Linear(pretrained_model.fc.in_features, embed_size)
        self.batchnorm = nn.BatchNorm1d(embed_size, momentum=0.01)
#        self.init_weights()
#
#    def init_weights(self):
#        # weight init, inspired by tutorial
#        self.linear.weight.data.normal_(0,0.02)
#        self.linear.bias.data.fill_(0)

    def forward(self, x):
        """
        Since we're applying a linear layer to the end of the pretrained resnet, the features are a 1-D vector
        """
        x = self.resnet(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.linear(x)

        return x

class RNN(nn.Module):
    """
    Recurrent Neural Network for Text Generation.
    To be used as part of an Encoder-Decoder network for Image Captioning.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, layers=1):
        """
        Initializer
        :param embed_size: size of word embeddings
        :param hidden_size: size of hidden state of the recurrent unit
        :param vocab_size: size of the vocabulary (output of the network)
        :param num_layers: number of recurrent layers (default=1)
        """

        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        # options: nn.RNN, nn.GRU, nn.LSTM
        self.unit = nn.LSTM(embed_size, hidden_size, layers, batch_first=True)
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
