#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import numpy as np
np.random.seed(3)
torch.manual_seed(3) # Keeping it fixed


parser = argparse.ArgumentParser(description='classification_model text classificer')
# learning
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('--batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('--log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('--test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('--save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('--save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('--early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('--save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data every epoch')
parser.add_argument('--dataset-dir', default='data', help = 'dataset directory path which contains negetive/positive/neutral files')
parser.add_argument('--max-length', type=int, default=600, help='The maximum number of characters in sequence.')
parser.add_argument('--min-freq', type=int, default=20, help='The minimum frequency of a character to be a vocab member')
# model
parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('--max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('--embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('--kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('--static', action='store_true', default=False, help='fix the embedding')
#rnn model
parser.add_argument('--rnn', action = 'store_true', default = False, help = 'activate char rnn')
parser.add_argument('--hidden_dim', type = int, default = 100, help = 'Hidden Dimensions of the RNN.')
parser.add_argument('--rnn-layers', type = int, default = 1, help = "Number of stacked RNN layers")
# device
parser.add_argument('--device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('--predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('--test', action='store_true', default=False, help='train or test')

args = parser.parse_args()

tokenizer = lambda sent: list(sent[::-1])


# load data
print("\nLoading data...")
text_field = data.Field(lower=True, include_lengths = True, batch_first = True, 
                        tokenize = tokenizer, fix_length = args.max_length)
label_field = data.Field(sequential=False, batch_first = True)
train_iter, dev_iter = mydatasets.load_twitter_dataset_vocab(text_field, label_field, 
                                                             args.dataset_dir, args.min_freq,
                                                             args.batch_size, args.rnn)



args.embed_num = len(text_field.vocab)
print("Size of Vocab :", args.embed_num)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now()
                                                    .strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model

if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    classification_model = torch.load(args.snapshot)
elif args.rnn:
    classification_model = model.RNNClassifier(args.embed_num, args.embed_num,
                                               args.hidden_dim, args.rnn_layers,
                                               args.class_num)
else:
    classification_model = model.CharCNN(args.embed_num, args.embed_dim,
                                         args.class_num, args.kernel_num,
                                         args.kernel_sizes, args.dropout)


if args.cuda:
    torch.cuda.set_device(args.device)
    classification_model = classification_model.cuda()
        

# train or predict
if args.predict is not None:
    label = predict.predict(args.predict, classification_model, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    train.eval(dev_iter, classification_model, args) 
    #Print index of labels
    for i in range(args.class_num):
        print(str(i) +" --> " + label_field.vocab.itos[i+1])
else:
    print()
    try:
        train.train(train_iter, dev_iter, classification_model, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
