from ...utils import *

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

"""
Datasets contract:
------------------
All datasets return a list of length 2 where the first element is "x" and the second element is "y", eg [x,y]. In most 
cases the dict values will be tensors. In other cases,'x' consists of multiple tensors (e.g., variable length padded 
sequence outputs a padded tensor and a sequence length tensor). In these cases, the values will be lists where the 
order of the tensors must align with the signature of the forward method in the model.
"""


class SparseMatrixDataset(Dataset):
    """Basic pytorch `Dataset` for sparse matrices. For memory efficiency, stores the data on the CPU in its input
    sparse format and converts the data to a dense format in the `__getitem__` method.
    """
    def __init__(self, X):
        """
        Args:
            X (scipy.sparse.csr.csr_matrix): A sparse data matrix with data in the rows. For example,
                this could be the output from sklearn's `CountVectorizer.transform` method.
        """
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    @property
    def data_size(self):
        return self.X.shape[1]

    def __getitem__(self, i):
        return [
            torch.tensor(np.asarray(self.X[i].todense())).type(torch.FloatTensor).squeeze(),
            torch.tensor(np.nan)
        ]


class VarLenPadTokenSeqDataset(Dataset):
    """Pytorch Dataset that returns padded tokenid tensors and can enforce a maximum sequence length. This kind of
    dataset is typically used to train an LSTM, for example.
    """
    # TODO: have dataset return the mask
    def __init__(self, text, vocab, y=None, pad_idx=1, max_seq_len=500):
        """
        Args:
            text (iterable[str]): Text corpus
            vocab (.Vocab): Vocab object
            y (iterable): Dependent variable
            pad_idx (int):  Padding index
            max_seq_len (int): Maximum sequence length. Documents/sequences longer than this are truncated.
        """
        self.vocab = vocab
        self.pad_idx = pad_idx

        # get tokenid tensors and sequence lengths for all data points
        tensors_lens = parallel_apply_dask_delayed(text, self.get_tensor_and_seq_len)

        # self.X is a tuple with length equal to number of data points
        # where each element of the tuple is a variable length 1-d tensor
        # these tensors contain token ids (integers) corresponding to the sequence
        self.X, lens = zip(*tensors_lens)
        self.max_seq_len = min(max(lens), max_seq_len)

        if y is not None:
            self.y = torch.from_numpy(y.values).type(torch.FloatTensor)
        else:
            self.y = torch.empty(len(text))

    def get_tensor_and_seq_len(self, string):
        tokens = string.split()
        tokenids = [self.vocab.w2i[tok] for tok in tokens]
        seq_len = len(tokenids)
        tensor = torch.tensor(tokenids)
        return tensor, seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        len_x = len(x)
        return [
            [
                F.pad(self.X[i], (0, self.max_seq_len - len_x), value=self.pad_idx),
                min(len_x, self.max_seq_len)
            ],
            self.y[i]
        ]


class TabularDataset(Dataset):
    """
    General purpose pytorch dataset for tabular data.
    """
    def __init__(self, X, cat_names, cont_names, y=None):
        """
        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.
            cat_names (list[str]): List of categorical column names.
            cont_names (list[str]): List of continuous column names.
        """
        self.has_y = y is not None
        cat_names = listify(cat_names)
        cont_names = listify(cont_names)
        self.X_cat = torch.from_numpy(X[cat_names].values).type(torch.LongTensor)
        self.X_cont = torch.from_numpy(X[cont_names].astype('float').values).type(torch.FloatTensor)
        if self.has_y: self.y = torch.from_numpy(y.values).reshape(-1,1)

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, index):
        return [
            [
                self.X_cat[index],
                self.X_cont[index]
            ],
            self.y[index] if self.has_y else torch.tensor(np.nan)
        ]