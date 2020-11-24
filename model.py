#!/usr/bin/env python3
"""
Description of the model:

To classify the reviews, we used LTSM which is a type of RNN as our major network.
For rating and category, both have a linear layer with different output sizes to ensure correct classfication.

In PyTorch, the hidden state (and cell state) tensors returned by the forward and backward RNNs are stacked on top of each other in a single tensor.
We make our sentiment prediction using a concatenation of the last hidden state from the forward 
RNN (obtained from final word of the sentence),  ℎ→𝑇 , and the last hidden state from the
backward RNN (obtained from the first word of the sentence),  ℎ←𝑇 , i.e.  𝑦̂=𝑓(ℎ→𝑇,ℎ←𝑇) 

On top of this, the our LTSM network is Multilayered or popularly known as Deep RNN.
The idea is that we add additional RNNs on top of the initial standard RNN, 
where each RNN added is another layer. The hidden state output by the first (bottom) RNN 
at time-step  𝑡  will be the input to the RNN above it at time step  𝑡 . 
The prediction is then made from the final hidden state of the final (highest) layer.

The final fully connected linear layer is different for Category and Rating.
For Category, the wrong accuracy part was consistently stayinng in the range of 17%, So we implemented a second 
fully connected layer to reduce the error and it worked and finally reduced error to aout 12%

We used Cross Entropy Loss as our main loss funcntion as it combines LogSoftMax and NLLLoss togehter.
The reason we used this was the dynamic ability and flexiilty CrossEntropy offered to validate and check
any form of vectors.

After running with the above given architecture, we thought of implementing the pre processing function.
We used python's Regex library to replace all the characters apart from alphabets. This enabled faster learning
and better prediction.

We Used Adam Optimizer with low learning rate because Adam works best with low learning rates. 
Higher learning rates tend to decrease the accuracy.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import re

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # Filtering out words which have characters that are not in range
    # a-z or A-Z. 
    # taken from https://medium.com/@galhever/sentiment-analysis-with-pytorch-part-1-data-preprocessing-a51c80cc15fb
    newSample = []
    for word in sample:
        # remove punctuation
        word = re.sub('[^a-zA-Z0-9]', ' ', word)
        # remove multiple spaces
        word = re.sub(r' +', ' ', word)
        # remove newline
        word = re.sub(r'\n', ' ', word)      
        newSample.append(word)
    return newSample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

inputSize = 300
stopWords = {}
wordVectors = GloVe(name='6B', dim=inputSize)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):

    ratingOutputNew   = torch.argmax(ratingOutput, dim=1, keepdim=False)
    categoryOutputNew = torch.argmax(categoryOutput, dim=1, keepdim=False)
    
    return ratingOutputNew, categoryOutputNew

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):

    def __init__(self):
        super(network, self).__init__()
        hiddenDimension = 150
        #RNN layer
        self.RNNLayer = tnn.LSTM(input_size=inputSize, 
                            hidden_size=hiddenDimension, 
                            num_layers=2, 
                            batch_first = True
                            )
        
        # Fully connected linear Layer
        self.linearLayerForRating   = tnn.Linear(hiddenDimension, 2)
        self.linearLayerForCategoryBasic = tnn.Linear(hiddenDimension, 20)
        self.linearLayerForCategory = tnn.Linear(20, 5)

        
    def forward(self, input, length):
        # RNN ===================================
        ltsmOutput, _  = self.RNNLayer(input)
        
        # Linear Layer ==========================
        ratingOutput   = self.linearLayerForRating(ltsmOutput)
        categoryOutput = self.linearLayerForCategoryBasic(ltsmOutput)
        categoryOutput = self.linearLayerForCategory(categoryOutput)
        
        return ratingOutput[:,-1], categoryOutput[:,-1]

class loss(tnn.Module):

    def __init__(self):
        super(loss, self).__init__()
        self.crossEntropyLoss = tnn.CrossEntropyLoss()
    
    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingLoss   = self.crossEntropyLoss(ratingOutput, ratingTarget)
        categoryLoss = self.crossEntropyLoss(categoryOutput, categoryTarget)
        
        return ratingLoss + categoryLoss

net      = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize     = 96
epochs        = 10
optimiser     = toptim.Adam(net.parameters(), lr=0.001)