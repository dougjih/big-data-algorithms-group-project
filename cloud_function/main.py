import pandas as pd
import numpy as np
import mlxtend.frequent_patterns
import mlxtend.preprocessing
import proj_code_pkg.vaers_csv
import proj_code_pkg.freq_itemsets

def main(
    data_path, symptoms_path, vax_path,
    freq_itemsets_output_path, assoc_rules_output_path,
    freq_itemsets_min_support, assoc_rule_metric, assoc_rule_min_threshold):
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
    data = proj_code_pkg.vaers_csv.read_data_file(data_path)
    print(f'Reading {symptoms_path}...')
    symptoms = proj_code_pkg.vaers_csv.read_symptoms_file(symptoms_path)
    print(f'Reading {vax_path}...')
    vax = proj_code_pkg.vaers_csv.read_vax_file(vax_path)
    print(f'Merging data...')
    merged_data = proj_code_pkg.vaers_csv.merge_dataframes([data, symptoms, vax])

    print(f'Creating baskets...')
    baskets = merged_data.apply(proj_code_pkg.freq_itemsets.build_basket, axis=1)
    print(f'One-hot encoding baskets...')
    one_hot_baskets_df = proj_code_pkg.freq_itemsets.build_one_hot_basket_dataset(baskets.to_list())
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
