import pandas as pd
import numpy as np
import mlxtend.frequent_patterns
import mlxtend.preprocessing
import functools


def read_vaers_csv(filename, dtype=None, parse_dates=False):
    '''Read a VAERS CSV file into a dataframe with "VAERS_ID" as index.'''
    return pd.read_csv(filename, encoding='iso-8859-1', dtype=dtype, parse_dates=parse_dates).set_index('VAERS_ID')


def read_data_file(filename):
    '''Read a VAERSDATA file into a dataframe with "VAERS_ID" as index.'''
    return read_vaers_csv(
        filename,
        dtype={
            'STATE': str,
            'AGE_YRS': float,
            'CAGE_YR': float,
            'CAGE_MO': float,
            'SEX': str,
            'SYMPTOM_TEXT': str,
            'DIED': str,
            'L_THREAT': str,
            'ER_VISIT': str,
            'HOSPITAL': str,
            'HOSPDAYS': float,
            'X_STAY': str,
            'DISABLE': str,
            'RECOVD': str,
            'NUMDAYS': float,
            'LAB_DATA': str,
            'V_ADMINBY': str,
            'V_FUNDBY': str,
            'OTHER_MEDS': str,
            'CUR_ILL': str,
            'HISTORY': str,
            'PRIOR_VAX': str,
            'SPLTTYPE': str,
            'FORM_VERS': float,
            'BIRTH_DEFECT': str,
            'OFC_VISIT': str,
            'ER_ED_VISIT': str,
            'ALLERGIES': str
        },
        parse_dates=['RECVDATE', 'RPT_DATE', 'DATEDIED', 'VAX_DATE', 'ONSET_DATE', 'TODAYS_DATE'])


def read_symptoms_file(filename):
    '''Read a VAERSSYMPTOMS file into a dataframe with "VAERS_ID" as index.'''
    return read_vaers_csv(filename)


def read_vax_file(filename):
    '''Read a VAERSVAX file into a dataframe with "VAERS_ID" as index.'''
    return read_vaers_csv(filename)


def merge_dataframes(dataframes: list) -> pd.DataFrame:
    '''Merge VAERS dataframes on the index ("VAERS_ID")'''
    return functools.reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, sort=False), dataframes)

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
    te_ary = te.fit_transform(baskets, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    return df


def main(
    data_path='gs://input-data-2zu7/2021VAERSDATA.csv',
    symptoms_path='gs://input-data-2zu7/2021VAERSSYMPTOMS.csv',
    vax_path='gs://input-data-2zu7/2021VAERSVAX.csv',
    freq_itemsets_output_path='gs://output-data-2zu7/freq_itemsets.csv',
    assoc_rules_output_path='gs://output-data-2zu7/assoc_rules.csv',
    freq_itemsets_min_support=0.001,
    assoc_rule_metric="confidence",
    assoc_rule_min_threshold=0.8):
    '''
    Using the given input VAERS data files, produce in the outpu_path CSV files
    containing frequen itemsets and association rules using the given parameters.

    This function makes use of the mlxtend library.

    Proof of concept of producing frequent itemsets results using a function
    that can run on a serverless function service such as Google Cloud Functions.

    Parameters:
        data_path : path to input VAERS data CSV file
        symptoms_path : path to input VAERS data CSV file
        vax_path : path to input VAERS data CSV file

        freq_itemsets_output_path : path to save the frequent itemsets result file
        assoc_rules_output_path : path to save the association rules result file

        freq_itemsets_min_support : float
            A float between 0 and 1 for minumum support of the itemsets returned.
        assoc_rule_metric : string
            'support', 'confidence', 'lift', 'leverage', or 'conviction'
        assoc_rule_min_threshold : float
            Minimal threshold for the evaluation metric
    '''

    print(f'Reading {data_path}...')
    data = read_data_file(data_path)
    print(f'Reading {symptoms_path}...')
    symptoms = read_symptoms_file(symptoms_path)
    print(f'Reading {vax_path}...')
    vax = read_vax_file(vax_path)
    print(f'Merging data...')
    merged_data = merge_dataframes([data, symptoms, vax])

    print(f'Creating baskets...')
    baskets = merged_data.apply(build_basket, axis=1)
    print(f'One-hot encoding baskets...')
    one_hot_baskets_df = build_one_hot_basket_dataset(baskets.to_list())
    print(f'Extracting frequent itemsets with min_support={freq_itemsets_min_support}...')
    frequent_itemsets = mlxtend.frequent_patterns.fpgrowth(
        one_hot_baskets_df, min_support=freq_itemsets_min_support, use_colnames=True)
    print(f'Generating association rules with metric=\'{assoc_rule_metric}\', min_threshold={assoc_rule_min_threshold}...')
    assoc_rules = mlxtend.frequent_patterns.association_rules(
        frequent_itemsets, metric=assoc_rule_metric, min_threshold=assoc_rule_min_threshold)

    print(f'Saving frequent itemsets to {freq_itemsets_output_path}...')
    frequent_itemsets.to_csv(freq_itemsets_output_path)    
    print(f'Saving association rules to {assoc_rules_output_path}...')
    assoc_rules.to_csv(assoc_rules_output_path)

if __name__ == '__main__':
    main('../data/2021VAERSDATA.csv', '../data/2021VAERSSYMPTOMS.csv', '../data/2021VAERSVAX.csv', \
        'freq_itemsets.csv', 'assoc_rules.csv', \
        0.001, "confidence", 0.8)
