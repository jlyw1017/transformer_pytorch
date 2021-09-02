# -*- coding: utf-8 -*-
"""Tools to preprocess data and definatin of dataloader."""
import unicodedata
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torchtext

from hparameters import hparameters


class DataLoader:
    """Definition of dataloader for pytorch training.

    Attributes:
        data_iter
    """
    def __init__(self, data_iter) -> None:
        self.data_iter = data_iter

    def __len__(self):
        return len(self.data_iter)

    def __iter__(self):
        for batch in self.data_iter:
            # Why this shape?
            yield(torch.transpose(batch.src, 0, 1), 
                  torch.transpose(batch.targ, 0, 1))


class CreateDataLoaders:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        tokenizer = lambda x: x.split(" ")
        self.src_text = torchtext.legacy.data.Field(
                sequential=True,
                tokenize=tokenizer,
                # lower=True,
                fix_length=hparameters['max_sentence_len'] + 2,
                preprocessing=lambda x: ['<start>'] + x + ['<end>']
            )
        self.targ_text = torchtext.legacy.data.Field(
                sequential=True,
                tokenize=tokenizer,
                # lower=True,
                fix_length=hparameters['max_sentence_len'] + 2,
                preprocessing=lambda x: ['<start>'] + x + ['<end>']
            )
        self.data_df = pd.read_csv(dataset_path,
                                   encoding='UTF-8', sep='\t', header=None,
                                   names=['deut', 'eng'], index_col=False)

    def normalize_data(self):
        """Normalizes data by means of coding, format and max length.

        Returns:
            (list): list of normalized data.
        """
        sentence_pairs = [[normalizes_string(src_language),
                           normalizes_string(tgt_language)]
                          for src_language, tgt_language in self.data_df.values]

        sentence_pairs = filter_sentence_length(sentence_pairs,
                                                hparameters["max_sentence_len"])
        return sentence_pairs

    def create_dataset(self, pairs):
        """Creates dataset from list of sentence pairs.

        Args:
            pairs (list): list of sentence pairs.

        Returns:
            dataset (Dataset): a torchtext.legacy.data.Dataset
        """
        # filed信息 fields dict[str, Field])
        fields = [('src', self.src_text), ('targ', self.targ_text)]

        examples = []
        for deu, eng in tqdm(pairs, desc="Creating examples."):
            # Calls field.preprocess during when creating examples.
            # 创建Example时会调用field.preprocess方法
            examples.append(
                torchtext.legacy.data.Example.fromlist([deu, eng], fields))
        dataset = torchtext.legacy.data.Dataset(examples, fields)
        return dataset

    def build_data_loader(self):
        """Builds dataloader from input data.

        Returns:
            train_dataloader (DataLoader): dataloader for training.
            val_dataloader (DataLoader): dataloader for validation.
        """
        sentence_pairs = self.normalize_data()
        train_pairs, val_pairs = train_test_split(sentence_pairs, test_size=0.2,
                                                  random_state=1234)

        train_dataset = self.create_dataset(train_pairs)
        val_dataset = self.create_dataset(val_pairs)

        self.src_text.build_vocab(train_dataset)
        self.targ_text.build_vocab(val_dataset)

        train_iter, val_iter = torchtext.legacy.data.Iterator.splits(
            (train_dataset, val_dataset),
            sort_within_batch=True,
            sort_key=lambda x : len(x.src),
            batch_sizes=(hparameters['batch_size'], hparameters['batch_size'])
        )

        train_dataloader = DataLoader(train_iter)
        val_dataloader = DataLoader(val_iter)
        return train_dataloader, val_dataloader


def filter_sentence_length(sentence_pairs, max_length):
    """Filters sentence pairs by max length.

    Args:
        sentence_pairs (list): list of pairs of sentences.
        max_length (int): max words num in each sentence.

    Returns:
      filter_result (list): filtered sentence pairs.
    """
    filter_result = [[src, tgt] for src, tgt in sentence_pairs
                     if len(src.split(" ")) < max_length and
                     len(tgt.split(" ")) < max_length]
    return filter_result


def unicode_to_ascii(s):
    """Decodes unicode to ascii.

    Args:
        s (str): input string.

    Returns:
        (str): string in ascii.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalizes_string(s):
    """Normalizes input string to standard format in tree steps.

    1 Adds space before the first "?" "!" or "."
    2 Replace any non-alphabet not "?" "!" or "." char with space.
    3 Merges multiple spaces.

    Args:
        s (str): input string

    Returns:
        s (str): normalized string.
    """
    # Unify code.
    s = s.lower().strip()
    s = unicode_to_ascii(s)

    # Standardized formats.
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]", r" ", s)
    s = re.sub(r"[\s]+", " ", s)
    return s
