import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from subprocess import check_output
import datetime
import sys

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
    if message <> None:
        sys.stdout.write(
            "[%-50s] %0.2f%% %d / %d - %s " % (
            '=' * int(completed * 50 / total), float(completed) * float(100) / float(total), completed, total, message))
    else:
        sys.stdout.write(
            "[%-50s] %0.2f%% %d / %d " % ('=' * int(completed * 50 / total), float(completed) * float(100) / float(total), completed, total))
    sys.stdout.flush() 
