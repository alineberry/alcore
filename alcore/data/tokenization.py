import re
import spacy
from fastprogress import progress_bar


def re_tokenizer(doc):
    """Performs tokenization of text using regex

    Args:
        doc (string): A document to be tokenized
    """
    return ' '.join(re.findall(r'(?:\b\w+\b|[^\s\n])', doc))


def spacy_tokenizer(text, lowercase=True, remove_stop=False, min_len=1):
    # TODO: add docstring
    nlp = spacy.load('en')
    docs = list(nlp.tokenizer.pipe(text))
    tokd = []
    for d in progress_bar(docs):
        toks = [t.lower_ if lowercase else t.text for t in d if not (t.is_stop and remove_stop) and len(t)>=min_len]
        tokd.append(' '.join(toks))
    return tokd
