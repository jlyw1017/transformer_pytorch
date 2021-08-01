"""Script to download dataset."""
import torch
import torchtext

from tqdm import tqdm  # 进度条
import pandas as pd
from sklearn.model_selection import train_test_split
import unicodedata, re

from hparameters import hparameters

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) 
                    if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = s.lower().strip()
    s = unicodeToAscii(s)

    # \1表示group(1)即第一个匹配到的 即匹配到'.'或者'!'或者'?'后，
    # 一律替换成'空格.'或者'空格!'或者'空格？'
    s = re.sub(r"([.!?])", r" \1", s)
    # 非字母以及非.!?的其他任何字符 一律被替换成空格
    s = re.sub(r"[^a-zA-Z.!?]", r" ", s)
    # 将出现的多个空格，都使用一个空格代替。
    # 例如：w='abc  1   23  1' 处理后：w='abc 1 23 1'
    s = re.sub(r"[\s]+", " ", s)
    
    return s

class DataLoader:
    def __init__(self, data_iter) -> None:
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        for batch in self.data_iter:
            yield(torch.transpose(batch.src, 0, 1), 
                  torch.transpose(batch.targ, 0, 1))


class CreateDataLoaders:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        tokenizer = lambda x: x.split() # 分词器
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

    def preprocess_data(self):
        pairs = [[normalizeString(s) for s in line] for line in self.data_df.values]
        # pairs = [[normalizeString(s) for s in line] for line in data_df.values]

        pairs = self.filter_pairs(pairs)
        train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=1234)
        return train_pairs, val_pairs

    def filter_pairs(self, pairs):
        def filterPair(p):
            return len(p[0].split(' ')) < hparameters['max_sentence_len'] and \
                   len(p[1].split(' ')) < hparameters['max_sentence_len']

        return [[pair[0], pair[1]] for pair in pairs if filterPair(pair)]

    def get_dataset(self, pairs):
        fields = [('src', self.src_text), ('targ', self.targ_text)]  # filed信息 fields dict[str, Field])
        examples = []  # list(Example)
        for deu, eng in tqdm(pairs): # 进度条
            # 创建Example时会调用field.preprocess方法
            examples.append(torchtext.legacy.data.Example.fromlist([deu, eng], fields))
        return examples, fields

    def build_data_loader(self):
        train_pairs, val_pairs = self.preprocess_data()
        train_examples, train_fields = self.get_dataset(train_pairs)
        val_examples, val_fields = self.get_dataset(val_pairs)

        train_dataset = torchtext.legacy.data.Dataset(train_examples, train_fields)
        val_dataset = torchtext.legacy.data.Dataset(val_examples, val_fields)

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