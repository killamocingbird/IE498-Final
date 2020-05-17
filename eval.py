import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from nlgeval import NLGEval
import os

# Custom imports
import models as m
from vocab import Vocabulary, load_vocab
from data_loader import get_coco_data_loader
import utils as u

torch.manual_seed(0)

# hyperparameters
batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
beam_size = 7
checkpoint = 'ModelFrozen_checkpoint.pth'
#checkpoint = 'Checkpoints/ModelFrozen_checkpoint.pth'

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

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

# Declare model
model = m.ShowTell(embed_size = 512, 
                    rnn_hidden_size = 512, 
                    vocab = vocab, 
                    rnn_layers = 1).to(device)

# Load in checkpoint to continue training if applicable
if checkpoint is not None:
    u.b_print("Loading checkpoint {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load(checkpoint)
    model = model.to(device)

# Lists to store reference (true caption), and hypothesis (prediction) for each image
# If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
# references = [[ref1a], [ref2a], ...], hypotheses = [hyp1, hyp2, ...]
references = list()
hypotheses = list()

print('training loss:', model.train_loss)
print('val loss:', model.val_loss)

print('<end> index: ', vocab.word2idx['<end>'])
print('<start> index: ', vocab.word2idx['<start>'])
print('<start> index: ', vocab.word2idx['<pad>'])
print('<unk> index: ', vocab.word2idx['<unk>'])

# Evaluate the model
try: # except KeyBoardInterrupt
    for i, (image, caption, lengths) in enumerate(test_loader):

        model(image.to(device), caption, lengths)
        pred, true = model.sample()
        print("--------------------------------------------------------------")
        print('True: \n', true)
        print('Greedy Predicted: \n', pred)

        k = beam_size

        # move to GPU device, if available
        image = image.to(device)
        caption = caption.to(device)

        # encode
        features = model.CNN(image)
        # we treat the problem as having a batch size of k
        features = features.unsqueeze(0).repeat(k, 1, 1)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[vocab.word2idx['<start>']]] * k).to(device)   # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words   # (k, 1)

        # Tensor to store top k sequences' scores; starting with 0
        top_k_scores = torch.zeros(k, 1).to(device) # (k, 1)

        # Lists to stored completed sequences and scores
        complete_seqs = []
        complete_seqs_scores = []

        # start decoding
        step = 1

        # init hidden state with image features, since we already have the start token, we already in a sense have the output from the first layer, so all we need is to get the hidden/context states.  
        # input shape: (seq_len, k, feature_size)
        output, (h, c) = model.RNN.unit(features, None)
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            # get embedding
            embeddings = model.RNN.embeddings(k_prev_words) # (1, s, embed_dim)
            # run through rnn
            output, (h, c) = model.RNN.unit(embeddings, (h, c))

            # get output scores
            scores = model.RNN.linear(output.squeeze(1)) # (s, vocab_size)
            scores = F.log_softmax(scores, dim = 1).squeeze(1)

            # add to get score of candidates + past
            scores = top_k_scores.expand_as(scores) + scores # (s, vocab_size)

            # For the first step, all k points will have the same scores
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) # (s)
            else:
                # Unroll and find top scores and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) # (s)

            # Convert unrolled indices to actual indices of scores
            # this indexes through s (seqs)
            prev_word_inds = top_k_words / len(vocab) # (s)
            # this indexes the vocab
            next_word_inds = top_k_words % len(vocab) # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], 
                              next_word_inds.unsqueeze(1)], dim = 1) 
                              # (s, step + 1)

            # which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocab.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly, the new variable k is what the 's' in the comments refer to

            # Proceed with incomplete sequences
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            h = h[:, prev_word_inds[incomplete_inds]]
            c = c[:, prev_word_inds[incomplete_inds]]
            # encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 25:
                break
            step += 1

        # if no end tokens are ever generated, the following code must be skipped
        if len(complete_seqs_scores) == 0:
            print('FINISHED NO SENTENCES')
            continue

        # obtain the best sequence out of k completed sequences
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # convert and print the hypothesis
        hypothesis_str = u.convert_back_to_text(seq, vocab)
        print('Beam Search: \n', hypothesis_str)

        # References (EXPECTED MULTIPLE CAPTIONS?)
        img_cap = caption.tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>']}],
                img_cap))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>']}])

        assert len(references) == len(hypotheses)
except KeyboardInterrupt:
    print('Early stopping of evaluation, evaluating metrics')
finally:
    # Calculate BLEU-4 scores
    #print(references)
    #print(hypotheses)
    nlgeval = NLGEval() 
    metrics_dict = nlgeval.compute_metrics(references, hypothesis)
    print(metrics_dict)
