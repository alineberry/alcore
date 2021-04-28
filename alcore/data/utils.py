from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from fastprogress import progress_bar
import os
import requests
from shutil import unpack_archive
from pathlib import Path
import torch
import torch.nn as nn
from collections import Counter
from itertools import chain
from scipy.sparse import csr_matrix


def get_fitted_countvectorizer(train_text, min_df=1, max_df=1., max_features=None):
    """Function that returns a fitted `sklearn.feature_extraction.text.CountVectorizer` from the training corpus

    Args:
        train_text (iterable[string]): Training corpus
        min_df: See sklearn's `CountVectorizer` docs
        max_df: See sklearn's `CountVectorizer` docs
        max_features: See sklearn's `CountVectorizer` docs. Dimensionality of the unigram data. Vocabulary size.

    Returns (sklearn.feature_extraction.text.CountVectorizer):
        A fitted CountVectorizer
    """
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), stop_words=None, ngram_range=(1, 1),
                                 max_df=max_df, min_df=min_df, max_features=max_features)
    print('fitting CountVectorizer...')
    vectorizer.fit(train_text)
    return vectorizer


def compute_unigram_matrix(corpus, vocab):
    """Returns a binary matrix (# docs, vocab size) where the i,j entry is the
    number of times word j occurred in document i.
    """
    data, row_idxs, col_idxs = [], [], []
    pb = progress_bar(corpus)
    pb.comment = 'computing unigram matrix'
    for row_idx, doc in enumerate(pb):
        doc = doc.split()
        for word in doc:
            col_idx = vocab.w2i[word]
            data.append(1)
            row_idxs.append(row_idx)
            col_idxs.append(col_idx)
    return csr_matrix((data, (row_idxs, col_idxs)), shape=(len(corpus), len(vocab)))


def compute_binary_unigram_matrix(corpus, vocab):
    """Returns a binary matrix (# docs, vocab size) where the i,j entry is in {0,1}
    and indicates whether or not word j occurred in document i.

    A simple wrapper over `compute_unigram_matrix`.
    """
    unigram = compute_unigram_matrix(corpus, vocab)
    return convert_unigram_matrix_to_binary(unigram)


def convert_unigram_matrix_to_binary(unigram):
    return (unigram > 0).astype(float)


def download_url(url, dest, overwrite=False, show_progress=True, chunk_size=1024 * 1024, timeout=10.):
    "Download `url` to `dest` unless it exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite: return

    u = requests.get(url, stream=True, timeout=timeout)
    try:
        file_size = int(u.headers["Content-Length"])
    except:
        show_progress = False

    with open(dest, 'wb') as f:
        nbytes = 0
        if show_progress:
            pbar = progress_bar(range(file_size), auto_update=False, leave=False)
            pbar.comment = url
        for chunk in u.iter_content(chunk_size=chunk_size):
            nbytes += len(chunk)
            if show_progress: pbar.update(nbytes)
            f.write(chunk)


def download_glove(path=None, overwrite=False):

    if Path(path).joinpath('glove.840B.300d.txt').exists() and not overwrite:
        print(f"using existing glove vectors at {str(Path(path).joinpath('glove.840B.300d.txt').resolve())}")
        return Path(path).joinpath('glove.840B.300d.txt')

    # set up source and destination paths
    DATA_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    DATA_DIR = Path('glove') if path is None else Path(path)
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    ZIP_FNAME = DATA_DIR.joinpath(Path(DATA_URL).name)

    # stream download the zip file
    download_url(DATA_URL, ZIP_FNAME, overwrite=overwrite, show_progress=True)

    # unzip
    unpack_archive(str(ZIP_FNAME), extract_dir=str(DATA_DIR))

    # delete the zip file
    ZIP_FNAME.unlink()

    return DATA_DIR / 'glove.840B.300d.txt'


def create_pretrained_glove_embeddings(weight_file, vocab, device=torch.device('cpu'), vector_size=300, sparse=False,
                                       freeze=True, verbose=False):
    """This function is used to build a pretrained Glove word embedding matrix for a given vocabulary. It ensures
    that the embedding vector at the padding index is set to zeros. All words in the vocabulary (built from the
    corpus at hand) are not expected to be found in the pretrained glove embeddings; words that are not matched have
    their embedding vectors set to the mean of all embedding vectors that were found for that vocabulary. The
    function also handles cases where multiple vectors are found for a given word; in this case, the final
    embedding is set to the mean of the matched embeddings for that word (this can be useful if you want to search
    in lowercase space).

    Args:
        weight_file (str/Path): Path to the previously downloaded glove embedding file
        vocab (Vocab): Vocab object constructed from a given corpus
        device (torch.device): Which device to place the final `nn.Embedding` object on
        vector_size (int): Dimensionality of the pretrained word vectors. For glove, this should be set to 300.
        sparse (bool): Whether or not to return a sparse `nn.Embedding` object
        freeze (bool): Whether or not to freeze the word embedding vectors (ie prevent gradient descent updates)
        verbose (bool): Whether or not to print progress and results

    Returns:
        nn.Embedding: A Pytorch embedding layer constructed from pretrained glove embeddings
    """

    pad_idx = vocab.w2i['__PAD__']

    # numpy matrix of zeros of the final shape of the embedding matrix
    embeddings = np.zeros(shape=(len(vocab), vector_size))

    # keep track of how many matches each word gets
    match_counts = np.zeros(len(vocab))

    # loop over lines of the weight file, extracting embedding vectors for words that appear in `vocab`
    with open(file=weight_file, mode='r', encoding='utf-8', errors='ignore') as file:
        if verbose: print('Creating pretrained glove word embedding matrix...')
        lines = progress_bar(file.readlines()) if verbose else file.readlines()
        for line in lines:
            line = line.split()
            word = ' '.join(line[:-vector_size])
            if word not in vocab.w2i.keys(): continue
            embed = np.asarray(line[-vector_size:], dtype='float32')
            i = vocab.w2i[word]
            match_counts[i] += 1
            embeddings[i] += embed

    # compute the mean of all embeddings matched to the vocabulary
    embeddings_mean = embeddings.sum(axis=0) / match_counts.sum()
    # identify which words in the vocab did not receive a glove vector (ie, no match found)
    no_match_mask = match_counts == 0
    # set the embeddings for non-matched words to the vocab mean
    embeddings[no_match_mask] = embeddings_mean
    # ensure the pad vector is zero
    embeddings[pad_idx] = 0
    match_counts[no_match_mask] = 1
    # if words were found twice in the file, set their embedding to the mean of the matched embeddings
    embeddings = embeddings / match_counts.reshape(-1, 1)

    if verbose:
        print(f'found pre-trained glove embedding vectors for {(~no_match_mask).sum()} of {len(vocab.w2i)} words in '
              f'the vocabulary')

    return nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), padding_idx=pad_idx, sparse=sparse,
                                        freeze=freeze).to(device)


def calc_token_freq(series):
    """
    Function to quickly compute all token frequencies in a corpus. Assumes the text has already been tokenized.

    Args:
        series: A pandas series containing the document corpus

    Returns:
        A Counter object
    """
    return Counter(chain.from_iterable(map(str.split, series)))


class InputData:
    """
    Holds all input dataframes
    """
    def __init__(self, train_df_lab, valid_df_lab, train_df_unlab=None, valid_df_unlab=None):
        self.train_df_lab = train_df_lab
        self.valid_df_lab = valid_df_lab
        self.train_df_unlab = train_df_unlab
        self.valid_df_unlab = valid_df_unlab

    def get_unlabeled_data(self, col_name):
        if (self.train_df_unlab is not None) and (col_name in self.train_df_unlab.columns):
                train = self.train_df_unlab[col_name]
                valid = self.valid_df_unlab[col_name]
                return train, valid
        else:
            return None, None

    def get_labeled_data(self, col_name):
        if col_name in self.train_df_lab.columns:
            train = self.train_df_lab[col_name]
            valid = self.valid_df_lab[col_name]
            return train, valid
        else: return None, None


class DataBunch:
    """
    Holds train and valid datasets and dataloaders
    """
    def __init__(self, train_ds, valid_ds, train_dl=None, valid_dl=None):
        self.train_ds, self.valid_ds = train_ds, valid_ds
        self.train_dl, self.valid_dl = train_dl, valid_dl


class TextDataBunch(DataBunch):
    def __init__(self, train_ds, valid_ds, train_text_raw, valid_text_raw,
                 train_text_tok, valid_text_tok, vocab, train_dl=None, valid_dl=None):
        super().__init__(train_ds=train_ds, valid_ds=valid_ds, train_dl=train_dl, valid_dl=valid_dl)
        self.train_text_raw, self.valid_text_raw = train_text_raw, valid_text_raw
        self.train_text_tok, self.valid_text_tok = train_text_tok, valid_text_tok
        self.vocab = vocab
