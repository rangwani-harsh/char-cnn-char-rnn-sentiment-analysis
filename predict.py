import os
import sys
import torch
import mydatasets
import torch.autograd as autograd
import argparse
import torchtext.data as data
torch.manual_seed(3)

parser = argparse.ArgumentParser(description='Predictor api')
parser.add_argument('--snapshot', type=str, default='saved-models/best-cnn.pt', help='filename of model snapshot [default: None]')
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('--input', help='Input file name', required=True)
parser.add_argument('--output', type=str, default = 'output.txt', help='File path of the predictions')
# Path to training data for vocab
parser.add_argument('--dataset-dir', default='data', help = 'dataset directory path which contains negetive/positive/neutral files')
parser.add_argument('--rnn', action = 'store_true', default = False, help = 'activate char rnn')
parser.add_argument('--max-length', type=int, default=600, help='The maximum number of characters in sequence.')
parser.add_argument('--min-freq', type=int, default=20, help='The minimum frequency of a character to be a vocab member')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')

args = parser.parse_args()

tokenizer = lambda sent: list(sent[::-1])

#For building the same vocab an easier alternative would be to save vocab but the dataset size is not large.
# Umm I couldn't find a fix for it.

print("\nLoading data...")
text_field = data.Field(lower=True, tokenize = tokenizer, fix_length = args.max_length)
label_field = data.Field(sequential=False)
train_iter, dev_iter = mydatasets.load_twitter_dataset_vocab(text_field, label_field, args.dataset_dir, 
                                                             args.min_freq, args.batch_size, args.rnn)


# Optimal for inferencing one instance at a time.
def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
   
    text = text_field.tokenize(text)
 
    text = [[text_field.vocab.stoi[x] for x in text]]

    x = torch.tensor(text)
    x = autograd.Variable(x)
    
    if args.rnn:
    	output = model(x, torch.Tensor([len(text)]))
    else:
        output = model(x)

    _, predicted = torch.max(output, 1)

    return label_feild.vocab.itos[predicted.data[0]+1]

    

if __name__ == '__main__':
    
    model = torch.load(args.snapshot, map_location=lambda storage, loc: storage)
    model.cpu()
    model.eval()

    output_file = open(args.output, 'w')
    input_file = open(args.input)

    for line in input_file:
        line = line.strip()
        prediction = predict(line, model, text_field, label_field,  True)
        output_file.write(prediction + "\n")
    
    output_file.close()
    input_file.close()

