import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import mlxtend.frequent_patterns
import mlxtend.preprocessing


def append_if_not_na(list: list, obj: object):
    '''Append "obj" to "list" if "obj" is not NA'''
    if not pd.isna(obj):
        list.append(obj)


def convert_to_age_group(age: float) -> str:
    '''Convert an age into an age group.

    The age groups are defined by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3825015/.'''

    if age < 3.0:
        return 'Age 0-2'
    elif age < 6.0:
        return 'Age 3-5'
    elif age < 14.0:
        return 'Age 6-13'
    elif age < 19.0:
        return 'Age 14-18'
    elif age < 34.0:
        return 'Age 19-33'
    elif age < 49.0:
        return 'Age 34-48'
    elif age < 65.0:
        return 'Age 49-64'
    elif age < 79.0:
        return 'Age 65-78'
    else:
        return 'Age 79-older'


def convert_to_sex_group(sex: str) -> str:
    '''Convert a VERES sex data value into representative string.'''
    
    if sex == 'F':
        return 'Female'
    elif sex == 'M':
        return 'Male'
    else:
        return 'Unknown Sex'


def convert_to_labeled_item(data: str, label: str) -> str:
    '''Convert a VERES boolean data value to the given label if the value is "Y".'''
    
    return label if not pd.isna(data) and data == 'Y' else pd.NA


def build_basket(data_row: pd.Series) -> list:
    
    '''
    Build a "basket" of itemset that consists of a list of values representing the data fields of interest.
    Use with the Pandas apply function with axis=1 to operate on rows of VAERS data.
    
    Parameters
    ----------
    data_row : Pandas Series
        a row of merged VAERS data
    
    Returns
    -------
    list
        values representing the data fields of interest
    '''
    
    basket = list()

    append_if_not_na(basket, data_row['STATE'])
    append_if_not_na(basket, convert_to_age_group(data_row['AGE_YRS']))
    append_if_not_na(basket, convert_to_sex_group(data_row['SEX']))
    append_if_not_na(basket, convert_to_labeled_item(data_row['DIED'], 'Died'))
    append_if_not_na(basket, convert_to_labeled_item(data_row['L_THREAT'], 'Life-threatening illness'))
    append_if_not_na(basket, convert_to_labeled_item(data_row['ER_VISIT'], 'Emergency room visit'))
    append_if_not_na(basket, convert_to_labeled_item(data_row['HOSPITAL'], 'Hospitalized '))
    append_if_not_na(basket, convert_to_labeled_item(data_row['X_STAY'], 'Prolongation of existing hospitalization'))
    append_if_not_na(basket, convert_to_labeled_item(data_row['DISABLE'], 'Disability'))
    append_if_not_na(basket, convert_to_labeled_item(data_row['RECOVD'], 'Recovered'))
    append_if_not_na(basket, convert_to_labeled_item(data_row['BIRTH_DEFECT'], 'Birth defect'))
    append_if_not_na(basket, data_row['VAX_NAME'])
    append_if_not_na(basket, data_row['SYMPTOM1'])
    append_if_not_na(basket, data_row['SYMPTOM2'])
    append_if_not_na(basket, data_row['SYMPTOM3'])
    append_if_not_na(basket, data_row['SYMPTOM4'])
    append_if_not_na(basket, data_row['SYMPTOM5'])

    return basket


def build_one_hot_basket_dataset(baskets: list) -> pd.DataFrame:
    '''Transform list of baskets to a dataframe in the one-hot format that the mlxtend frequen itemsets functions expect.'''

    te = mlxtend.preprocessing.TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df
