from ..utils import *

import numpy as np
import pandas as pd
import re
from pandas.api.types import is_numeric_dtype
from fastprogress import progress_bar
import dill


ALL_DATEPART_ATTR = ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
                     'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Hour', 'Minute',
                     'Second', 'Elapsed')


def add_datepart(df, fldname, attr='all', drop=True):
    "Helper function that adds columns relevant to a date in the column `fldname` of `df`."

    if attr == 'all':
        attr = list(ALL_DATEPART_ATTR)

    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    new_colnames = []

    if 'Elapsed' in attr:
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        new_colnames.append(targ_pre + 'Elapsed')

    for n in attr:
        if n == 'Elapsed': continue
        new_colname = targ_pre + n
        df[new_colname] = getattr(fld.dt, n.lower())
        new_colnames.append(new_colname)

    if drop: df.drop(fldname, axis=1, inplace=True)
    return df, new_colnames


class TabularPreprocessor:

    # TODO: add a verbose option
    # TODO: clean up docstring
    # TODO: add ability to call fit_transform on the dependent variable only

    def __init__(self, cat_names=None, cont_names=None, dep_var=None, dep_var_type=None, date_names=None,
                 datepart_attr='all', drop_orig_date_cols=False, procs=('FillMissing', 'Categorify')):
        """Class for preprocessing tabular datasets into machine-learning ready form.

        Args:
            df (pandas df): Input dataset, likely not cleaned and processed yet.
            cat_names (list): List of column names corresponding to categorical features.
            cont_names (list): List of column names corresponding to continuous features.
            dep_var (str): Name of dependent variable column.
            date_names (list): List of date column names.
            datepart_attr (list): Which date attributes to extract from date columns
            drop_orig_date_cols (bool): Whether or not to drop the original date columns after extracting date parts.
            procs (list-like): List of processes to run the data through.
        """

        for p in procs: assert p in ['FillMissing','Categorify'], 'invalid proc passed'
        if dep_var is not None:
            assert dep_var_type in ['continuous', 'categorical'], f'invalid value ({dep_var_type}) passed for ' \
                                                                  f'dep_var_type parameter'

        cat_names = [] if cat_names is None else cat_names
        cont_names = [] if cont_names is None else cont_names
        date_names = [] if date_names is None else date_names
        self.cat_names,self.cont_names,self.dep_var,self.date_names=cat_names,cont_names,dep_var,date_names
        self.dep_var_type = dep_var_type
        self.drop_orig_date_cols = drop_orig_date_cols
        self.procs = procs
        if datepart_attr == 'all':
            datepart_attr = list(ALL_DATEPART_ATTR)
        self.datepart_attr = datepart_attr

        self.catcode2str = {}
        self.str2catcode = {}
        self._augmntd_cat_names, self._augmntd_cont_names = [], []

        self._medians = {}

        self.depvar_value2code = {}
        self.depvar_code2value = {}

    @property
    def _fill_missing_bool(self):
        """Boolean indicating whether to perform the FillMissing transform"""
        return 'FillMissing' in self.procs and len(self.cont_names + self._augmntd_cont_names) > 0

    @property
    def _categorify_bool(self):
        """Boolean indicating whether to perform the Categorify transform"""
        return 'Categorify' in self.procs and len(self.cat_names + self._augmntd_cat_names) > 0

    def X_return(self, df):
        return df[self.all_cat_names + self.all_cont_names]

    def y_return(self, df):
        return None if ((self.dep_var is None) or (self.dep_var not in df.columns)) else df[self.dep_var]

    def fit_transform(self, train_df):

        # add date features
        train_df = self._extract_date_features(train_df, self.date_names, self.drop_orig_date_cols)

        # fill missing values (if applicable)
        if self._fill_missing_bool:
            self._get_medians(train_df)
            train_df = self._fill_missing(train_df)

        # convert categorical fields into numeric fields, remembering a mapping to original values
        if self._categorify_bool:
            train_df = self._convert_catflds_to_str(train_df)
            self._get_category_mappings(train_df)
            train_df = self._categorify(train_df)

        if self.dep_var is not None:
            self._fit_dep_var(train_df)
            train_df = self._transform_depvar(train_df)

        return self.X_return(train_df), self.y_return(train_df)

    def _cast_contcols_to_numeric(self, df):
        """Used to ensure all continuous columns are of a proper datatype"""
        if len(self.cont_names)==0: return df
        pb = progress_bar(self.cont_names)
        print('Casting continuous columns to numeric datatypes...')
        for c in pb:
            if not is_numeric_dtype(df[c]): df[c] = pd.to_numeric(df[c])
        return df

    def _extract_date_features(self, df, date_names, drop):
        if len(self.date_names)==0: return df
        df, new_cat_names, new_cont_names = self.add_date_features(df, date_names, self.datepart_attr, drop)
        self._update_augmntd_cat_names(new_cat_names)
        self._update_augmntd_cont_names(new_cont_names)
        return df

    def _update_augmntd_cat_names(self, new_cat_names):
        self._augmntd_cat_names.extend(new_cat_names)
        self._augmntd_cat_names = list(set(self._augmntd_cat_names))

    def _update_augmntd_cont_names(self, new_cont_names):
        self._augmntd_cont_names.extend(new_cont_names)
        self._augmntd_cont_names = list(set(self._augmntd_cont_names))

    @property
    def all_cat_names(self):
        return self.cat_names + self._augmntd_cat_names

    @property
    def all_cont_names(self):
        return self.cont_names + self._augmntd_cont_names

    @property
    def new_cat_names(self):
        return self._augmntd_cat_names

    @property
    def new_cont_names(self):
        return self._augmntd_cont_names

    def _get_medians(self, train_df):
        if len(self.cont_names + self._augmntd_cont_names)==0: return
        print('Calculating medians...')
        pb = progress_bar(self.cont_names + self._augmntd_cont_names)
        for c in pb:
            null_mask = train_df[c].isnull()
            if null_mask.sum() > 0:
                self._update_augmntd_cat_names([c+'_null'])
                self._medians[c] = train_df[c].median()
            pb.comment = f'finished {c}'

    def _fill_missing(self, df):
        if len(self._medians)==0: return df
        print('Filling missing values...')
        pb = progress_bar(self._medians.items())
        for c,median in pb:
            null_mask = df[c].isnull()
            df[c+'_null'] = null_mask
            df.loc[null_mask,c] = median
            pb.comment = f'finished {c}'
        return df

    def _convert_catflds_to_str(self, df):
        for c in self.cat_names + self._augmntd_cat_names:
            df[c] = df[c].astype(str)
        return df

    def _get_category_mappings(self, train_df):
        print('Calculating category mappings...')
        pb = progress_bar(self.cat_names + self._augmntd_cat_names)
        for c in pb:
            tmp = train_df[c].astype('category')
            self.catcode2str[c] = ['x__missing__x'] + list(tmp.cat.categories)
            self.str2catcode[c] = DictWithDefaults({v: k for k, v in enumerate(self.catcode2str[c])})
            pb.comment = f'finished {c}'

    def _categorify(self, df):
        print('Categorifying...')
        pb = progress_bar(self.cat_names + self._augmntd_cat_names)
        for c in pb:
            # df[c] = parallel_apply_dask_delayed(df[c], lambda x: self.str2catcode[c][x])
            df[c] = df[c].fillna('x__missing__x').apply(lambda x: self.str2catcode[c][x])
            df[c] = df[c].astype('int32')
            pb.comment = f'finished {c}'
        return df

    def _fit_dep_var(self, train_df):

        if self.dep_var_type == 'categorical':
            original_values = list(train_df[self.dep_var].unique())
            adjudication = adjudicate_categorical_dep_var(train_df[self.dep_var])
            if adjudication in ['use_existing_string', 'use_existing_num']:
                self._create_depvar_mappings_from_existing(original_values)
            elif adjudication in ['create_new_mapping']:
                self._create_depvar_mappings(original_values)

    def _create_depvar_mappings(self, original_values):
        self.depvar_code2value = {code:value for code,value in enumerate(original_values)}
        self.depvar_value2code = {value:code for code,value in self.depvar_code2value.items()}

    def _create_depvar_mappings_from_existing(self, original_values):
        if not is_numeric_dtype(pd.Series(original_values)):
            codes = list(pd.to_numeric(pd.Series(original_values)))
        else:
            codes = original_values
        self.depvar_value2code = {v:c for v,c in zip(original_values, codes)}
        self.depvar_code2value = {c:v for c,v in zip(codes, original_values)}

    def _transform_depvar(self, df):
        df[self.dep_var] = df[self.dep_var].apply(lambda value: self.depvar_value2code[value])
        return df

    @staticmethod
    def add_date_features(df, date_names, datepart_attr=ALL_DATEPART_ATTR, drop=True):
        """For the given date columns, adds features like...
            - year
            - day of week
            - month of year

        Args:
            df (pandas df): df to modify (it will be modified in-place)
            date_names (list): list of column names containing date types
            datepart_attr (list of strings): Which date parts to extract from date columns
            drop (bool): whether or not to delete the original date column once the information has been extracted from
                it.

        Returns:
            (pandas df, list, list): Returns the modified dataframe as well as lists of new categorical and
                continuous fields created as part of the date extraction.
        """
        date_names = listify(date_names)
        new_cat_names, new_cont_names = [], []
        print('Adding date features...')
        pb = progress_bar(date_names)
        for c in pb:
            df, new_colnames = add_datepart(df, c, attr=datepart_attr, drop=drop)
            elapsed_colnames = [x for x in new_colnames if 'elapsed' in x.lower()]
            new_cont_names.extend(elapsed_colnames)
            new_colnames = [x for x in new_colnames if x not in elapsed_colnames]
            new_cat_names.extend(new_colnames)
            pb.comment = f'finished {c}'
        return df, new_cat_names, new_cont_names

    @staticmethod
    def randomly_sample_df_indices(df, pct):
        size = int(pct * len(df))
        return np.random.choice(df.index.values, size, replace=False)

    def transform(self, df):
        """Used to transform futher dataframes, after `TabularPreprocessor` has been fit on the training dataframe"""

        df = self._cast_contcols_to_numeric(df)

        df = self._extract_date_features(df, self.date_names, self.drop_orig_date_cols)

        # fill missing values (if applicable)
        if self._fill_missing_bool:
            df = self._fill_missing(df)

        # convert categorical fields into numeric fields, remembering a mapping to original values
        if self._categorify_bool:
            df = self._convert_catflds_to_str(df)
            df = self._categorify(df)

        if self.dep_var is not None:
            df = self._transform_depvar(df)

        X = self.X_return(df)
        y = self.y_return(df)

        return X, y

    def export(self, path):
        with open(path, 'wb') as fh:
            dill.dump(self, fh)


def adjudicate_categorical_dep_var(series):
    dep_var_tmp = series.copy()
    # store the original levels in the Series
    if not is_numeric_dtype(dep_var_tmp):
        # if it's not numeric, try to convert it to numeric (eg when data is like ['1', '2', '2', '3'])
        if is_convertable_to_numeric(dep_var_tmp):
            dep_var_tmp = pd.to_numeric(dep_var_tmp)
            if dep_var_has_desired_values(dep_var_tmp):
                return 'use_existing_string'
            else:
                return 'create_new_mapping'
        # if it's not numeric, and can't be converted to numeric (eg data like ['red', 'blue', 'orange']
        else:
            return 'create_new_mapping'
    # if it is numeric
    else:
        if dep_var_has_desired_values(dep_var_tmp):
            return 'use_existing_num'
        else:
            return 'create_new_mapping'


def is_convertable_to_numeric(series):
    try:
        _ = pd.to_numeric(series)
        return True
    except ValueError:
        return False


def dep_var_has_desired_values(dep_var):
    # compute the desired values for the dependent variable. these should be [0, n_classes-1]
    n_unique = dep_var.nunique()
    desired_values = list(range(n_unique))

    # get the existing values
    unique = list(dep_var.unique())
    unique.sort()

    return desired_values == unique
