import numpy as np
from multiprocessing import Pool, cpu_count
from subprocess import check_output
import datetime
import sys
import dask
import pandas as pd
from collections.abc import Iterable
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import HTML


def stretch_notebook():
    from IPython.core.display import display
    display(HTML("<style>.container { width:80% !important; }</style>")) 


def parallelize_df(df, func):
    """
    Paralellizes a map/apply function onto a Pandas dataframe
    :param df: Pandas df
    :param func: the map/apply function
    :return: Pandas series; result of the map/apply function
    """
    cores = cpu_count()
    partitions = cores*2
    data_split = np.array_split(df, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def plot_missingness_heatmap(df):
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')


def describe_scalar(df, col):
    x = df[df[col].notnull()][col]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.distplot(x, rug=False, ax=axs[0])
    plt.sca(axs[0])
    plt.xticks(rotation=90)
    x.plot.box(vert=True, ax=axs[1], showfliers=False)
    sns.violinplot(x=x, ax=axs[2])
    plt.sca(axs[2])
    plt.xticks(rotation=90)
    print(x.describe())


def execute_bash(bash):
    check_output(bash, shell=True)


def get_max_date_from_strings(datestringslist, formatstring='%Y-%m-%d'):
    """input: a list of dates in string format. output: most recent date in the input string format"""
    if not datestringslist:
        return []
    return datetime.datetime.strftime(max([datetime.datetime.strptime(date, formatstring) for date in datestringslist]),
                                      formatstring)


def print_now():
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%b-%d %H:%M:%S'))


def get_now():
    return pd.Timestamp(datetime.datetime.now())


def remove_colname_prefix(df, sep='.'):
    """
    For use when a SQL query returns column names with an alias or table name are prepended to the column name,
    eg, a.employeeno or takeoff.item. Will remove the prefix along with the separator from all column names.
    Input: pandas Dataframe, output: pandas Dataframe
    """
    cols = df.columns.tolist()
    new_cols = []
    for col in cols:
        if sep in col:
            new_cols.append(col.split(sep)[1])
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df


def single_line_percent_complete(total, completed, message=None):
    sys.stdout.write('\r')
    if message!= None:
        sys.stdout.write(
            "[%-50s] %0.2f%% %d / %d - %s " % (
            '=' * int(completed * 50 / total), float(completed) * float(100) / float(total), completed, total, message))
    else:
        sys.stdout.write(
            "[%-50s] %0.2f%% %d / %d " % ('=' * int(completed * 50 / total), float(completed) * float(100) / float(total), completed, total))
    sys.stdout.flush() 


def parallel_apply_dask_delayed(inp, func, **kwargs):
    """Parallel apply using dask delayed. See https://examples.dask.org/applications/embarrassingly-parallel.html for
    details.

    Note: this has been the fastest in the limited testing performed so far.

    Parameters
    ----------
    inp
    func

    Returns
    -------

    """
    lazy_results = []
    for s in inp:
        lazy_result = dask.delayed(func)(s, **kwargs)
        lazy_results.append(lazy_result)
    res = dask.compute(*lazy_results)
    return res


class DictWithDefaults(dict):
    """Simple subclass of python's standard `dict` that overrides the `__missing__` method to return a specified
    default value
    """
    def __init__(self, *args, default=0, **kwargs):
        """
        Args:
            *args: args to `dict`
            default: The value to return when looking up a key that doesn't exist
            **kwargs: kwargs to `dict`
        """
        super().__init__(*args, **kwargs)
        self.default = default

    def __missing__(self, key):
        return self.default


def set_pandas_colwidth(w): pd.set_option('display.max_colwidth', w)


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    elif isinstance(p, torch.Tensor): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def plot_dist(pandas_column, title=None, xlabel='', ylabel='', figsize=(15, 4), bins=100, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    sns.distplot(pandas_column.dropna(), kde=False, ax=ax, bins=bins, hist_kws={'alpha': 0.9})
    title = title if title is not None else f'Distribution of {pandas_column.name}'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()


def plot_cumprop_from_counter(counter, return_df=False, ax=None, title=None, xlabel=None, ylabel=None, figsize=None,
                              xlim=None):
    df = pd.DataFrame(
        {
            'keys': list(counter.keys()),
            'freq': list(counter.values())
        }).sort_values(by='freq', ascending=False)
    df['cumprop'] = df.freq.cumsum() / df.freq.sum()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(df)), df['cumprop'])
    if xlim is not None: ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    return ax, df if return_df else None


def plot_date_dist(series, title=None, figsize=(15, 4), by='day', xlabel=None, ylabel=None):
    xlabel = 'Date' if xlabel is None else xlabel
    ylabel = f'Number Records per {by}' if ylabel is None else ylabel
    assert by in ['day', 'month'], f'invalid value passed for day parameter ({by})'
    columns = ['year', 'month', 'day']
    if by == 'day':
        gb = [series.dt.year, series.dt.month, series.dt.day]
    elif by == 'month':
        gb = [series.dt.year, series.dt.month]
    counts = series.groupby(gb).size().to_frame('num_recs')
    dateparts = list(counts.index.values)
    if by == 'month': dateparts = list(map(lambda x: list(x) + [1], dateparts))
    x = pd.to_datetime(pd.DataFrame(dateparts, columns=columns))
    y = counts['num_recs']
    fig, ax = plt.subplots(figsize=figsize)
    title = title if title is not None else f'{series.name} | Number of records per {by}'
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel)
    plt.show()
