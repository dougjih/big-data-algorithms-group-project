import functools
import pandas as pd


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
