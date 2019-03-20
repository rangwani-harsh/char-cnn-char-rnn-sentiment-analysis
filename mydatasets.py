import re
import os
import random
from torchtext import data
random.seed(3) #Please keep it fixed for generating the same validation set

class TwitterDataset(data.Dataset):
    """Create an Twitter dataset instance given a path and fields.

        Parameters:
        ----------
        text_field: The field that will be used for text data.
        label_field: The field that will be used for label data.
        path: Path to the data directory.
        examples: The examples contain all the data.
    """


    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None):

        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'positive'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
            with open(os.path.join(path, 'negative'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'neutral'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'neutral'], fields) for line in f]
            
        super(TwitterDataset, self).__init__(examples, fields)

    @classmethod
    def splits(cls, text_field, label_field, path, dev_ratio=.1, shuffle = True):
        """Create dataset objects for splits of the twitter dataset.
        The custom implementation is required as we require different 
        dataset objects for training and validation.

        Parameters:
        ----------
        text_field: The field that will be used for the sentence.
        label_field: The field that will be used for label data.
        dev_ratio: The ratio that will be used to get split validation dataset.
        shuffle: Whether to shuffle the data before split.

        Returns:
        --------
        A tuple of (train_data, validation_data).
        """

        examples = cls(text_field, label_field, path=path).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


# This doesn't fit into above dataset class  (Need suggestions to place it at a better place.)
def load_twitter_dataset_vocab(text_field, label_field, dataset_dir, min_freq, batch_size, is_rnn):
    """ The function to build vocab and generate test and train splits.
        Parameters:
        ----------
        text_field : Text field of the tweet.
        label_field : The label_field of the tweet.
        dataset_dir : Directory of dataset.
        batch_size : The size of the batch.
        is_rnn : Is the model a RNN. (For sorting the batches)

        Returns:
        --------
        train_iter : The iterator to training batches.
        dev_iter : The iterator to development set batches.

    """

    train_data, dev_data = TwitterDataset.splits(text_field, label_field, path = dataset_dir)
    text_field.build_vocab(train_data, dev_data, min_freq = min_freq)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(batch_size, len(dev_data)),
                                sort_key = TwitterDataset.sort_key,
                                sort_within_batch = is_rnn)
    return train_iter, dev_iter
