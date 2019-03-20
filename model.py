import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CharCNN(nn.Module):
    """Char CNN implementation based on Zhang et. al. (NIPS, 2015).
    The differences here are that we don't consider the same vocabulary
    as in paper as in twitter data there are emoji's that are necessary
    for sentiment analysis. 
    Parameter
    ---------
    vocab_size : Size of the vocab of the text
    embed_dim : Dimension of the character embedding(Trainable)
    class_num : Number of different classes for classification.
    kernel_num : The number of different kernels
    kernel_sizes : List of sizes of kernels
    dropout : Dropout for last fully connected

    """ 
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_num, kernel_sizes, dropout):
        super(CharCNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        # As we need convolution in 1D
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes]) 
        self.dropout = nn.Dropout(dropout)

        self.fully_connected = nn.Linear(len(kernel_sizes)*kernel_num, num_classes)


    def forward(self, x):
        
        # Shape [batch_size, sent_len]
        x = self.embed(x)  
        
        # Shape [batch_size, sent_len, embed_dim]
        x = x.unsqueeze(1)  

        # Shape [batch_size, 1, sent_len, embed_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        # Shape [[batch_size, kernel_num, sent_len], ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  

        # Shape [[batch_size, kernel_num], ..]*len(kernel_sizes)
        x = torch.cat(x, 1)

        # Shape [[batch_size, kernal_num * len(kernal_sizes)]]
        x = self.dropout(x)

        # Shape [[batch_size, kernal_num * len(kernal_sizes)]] 
        logit = self.fully_connected(x)  

        # Shape [[batch_size, num_classes]]
        return logit




class RNNClassifier(nn.Module):
    """ Character based RNN classifier which takes a sentence 
        converts into characters and runs RNN over it.

        Parameters:
        ----------
        vocabulary_size : The size of vocabulary of characters.
        embedding_dim : The embedding dimension for each element of vocab.
        hidden_dim : The hidden dimensions of the RNN layer.
        rnn_layers : The number of layers of RNN layers.
        n_classes : The class of labels present. (No. of different labels)
        
    """

    def __init__(self, vocabulary_size, embedding_dim,
                 hidden_dim, rnn_layers, n_classes):

        super(RNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, 
                                      padding_idx=0)

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                          num_layers=rnn_layers, bidirectional=False,
                          batch_first = True)

        self.dense = nn.Linear(in_features=hidden_dim, out_features=n_classes)


    def forward(self, input_sents, seq_len):
        """ 
        Parameters:
        ----------
        sentences : Having shape (batch_size, sentence_len)
        seq_len : Length of sequences sorted in descending order shape (batch_size).
        Returns:
        --------
        logits : Logits of the shape (batch_size, n_classes)
        """

        # Shape [batch_size, sentence_len]
        embedding = self.embedding(input_sents)
        # Shape [batch_size, sentence_len, embed_dim]
        embedding = pack_padded_sequence(embedding, seq_len, batch_first = True)
        out, _ = self.gru(embedding)
        out, lengths = pad_packed_sequence(out, batch_first = True)

        # Shape out => [batch_size, sentence_len, hiddem_dim]

        # Since we are doing classification, we only need the last
        # output from RNN which will be at different places 
        lengths = [l - 1 for l in lengths]
        last_output = out[range(len(lengths)), lengths]

        # Shape [sentence_len, hidden_dim]
        logits = self.dense(last_output)
        
        # Shape [sentence_len, num_classes]
        return logits
