import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import os

# Custom imports
import models as m
from vocab import Vocabulary, load_vocab
from data_loader import get_coco_data_loader
import utils as u


# Image Preprocessing
transform = transforms.Compose([transforms.ToTensor()])
batch_size = 1


# load COCO test dataset
vocab = load_vocab()

dataset = 'val' # 'test' or 'val'

IMAGES_PATH = 'data/{}2014'.format(dataset)
CAPTION_FILE_PATH = 'data/annotations/captions_{}2014.json'.format(dataset)
test_loader = get_coco_data_loader(path=IMAGES_PATH,
                                  json=CAPTION_FILE_PATH,
                                  vocab=vocab,
                                  transform=transform,
                                  batch_size=batch_size,
                                  shuffle=True)

# Lists to store reference (true caption), and hypothesis (prediction) for each image
# If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
# references = [[ref1a], [ref2a], ...], hypotheses = [hyp1, hyp2, ...]

idx = 4
captions = ['a man is sitting on a bench with a dog .',
            'a man sitting on top of a wooden bench .',
            'a man is sitting on a bench in front of a building',
            'a man is holding a frisbee in his hand',
            'a man is holding a skateboard on a ramp']
hypotheses = captions[idx].split(' ')
hypotheses = [[vocab.word2idx[w] for w in hypotheses]]

bestpic = None
bestcap = None
highscore = 0

# Evaluate the model
try: # except KeyBoardInterrupt
    for i, (image, caption, lengths) in enumerate(test_loader):
        img_cap = caption.tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>']}],
                img_cap))  # remove <start> and pads
        reference = [img_captions]

        bleu4 = corpus_bleu(reference, hypotheses)
        
        if bleu4 > highscore:
            bestpic = image 
            bestcap = caption.tolist()
            highscore = bleu4
            print(highscore)


except KeyboardInterrupt:
    print('Early stopping of evaluation, trying bleu metric if possible')
finally:
    import matplotlib.pyplot as plt 
    print('\n\n\n\n\n')
    bestcap = ' '.join([vocab.idx2word[w] for w in bestcap[0]][1:-1])
    print('outp caption:', captions[idx])
    print('real caption:', bestcap)
    print('bleu4:',highscore)
    bestpic = np.transpose(bestpic.numpy()[0],(1,2,0))
    print(bestpic.shape)
    plt.imshow(bestpic)

    plt.show()