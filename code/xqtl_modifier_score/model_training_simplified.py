import os
import sys
import argparse

import pandas as pd
from dask import dataframe as dd
from dask.diagnostics import ProgressBar


pbar = ProgressBar(dt=1)
pbar.register()
# pbar.unregister()

import time
from sklearn.model_selection import cross_val_score, BaseCrossValidator
from sklearn.model_selection import LeaveOneGroupOut

import optuna
from optuna.samplers import TPESampler, CmaEsSampler, GPSampler

from tqdm import tqdm

tqdm.pandas()

import requests
import numpy as np
from os import walk
import os
import dask

import pickle

import json
import pysam
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn import metrics
import sklearn
from sklearn.linear_model import SGDClassifier
import yaml
import pickle
import torch
import random
from sklearn.calibration import CalibratedClassifierCV
import optunahub
import joblib

parser = argparse.ArgumentParser(description="Train eQTL prediction model")
parser.add_argument("cohort", type=str, help="Cohort/project name (e.g., Mic_mega_eQTL)")
parser.add_argument("chromosome", type=str, help="Chromosome number (e.g., 2)")
parser.add_argument("--data_config", type=str, required=True, help="Path to data configuration YAML")
parser.add_argument("--model_config", type=str, required=True, help="Path to model configuration YAML")

args = parser.parse_args()

cohort = args.cohort
chromosome = args.chromosome
data_config_path = args.data_config
model_config_path = args.model_config

# Load configurations
params_data = yaml.safe_load(open(data_config_path))
model_params = yaml.safe_load(open(model_config_path))

# Set random seeds from configuration
torch.manual_seed(params_data['system']['random_seeds']['torch_seed'])
np.random.seed(params_data['system']['random_seeds']['numpy_seed'])
random.seed(params_data['system']['random_seeds']['random_seed'])

# Set dask temporary directory from configuration
dask.config.set({'temporary_directory': params_data['system']['dask_temp_directory']})

# Extract paths from data config
gene_lof_file = params_data['gene_constraint_file']
maf_file_pattern = params_data['maf_file_pattern']
data_dir_pattern = params_data['training_data']['base_dir']

chromosome_out = f'chr{chromosome}'

NPR_tr = params_data['sampling']['npr_train']
NPR_te = params_data['sampling']['npr_test']

chromosome_out = f'chr{chromosome}'

# Set up chromosomes - will validate data availability later
train_chromosomes = [f'chr{chromosome}']
test_chromosomes = [f'chr{chromosome}']
num_train_chromosomes = len(train_chromosomes)

print(f"NOTE: Currently using same chromosome ({chromosome}) for train/test due to limited data.")
print(f"The train/test split will rely on different data directories with different sampling thresholds.")

# Load gene constraint data with configurable sheet name
gene_lof_df = pd.read_excel(gene_lof_file, params_data['gene_constraint_sheet'])

gene_id_col = params_data['gene_constraint_columns']['gene_id']
gene_lof_col = params_data['gene_constraint_columns']['gene_lof']
gene_lof_df = gene_lof_df[[gene_id_col, gene_lof_col]]

gene_lof_df = gene_lof_df.rename(columns={gene_id_col: 'gene_id', gene_lof_col: 'gene_lof'})

gene_lof_df['gene_lof'] = np.log2(gene_lof_df['gene_lof'])


maf_files = maf_file_pattern.format(chromosome=chromosome)
maf_df = dd.read_csv(maf_files, sep='\t')

maf_df = maf_df[['variant_id', 'gnomad_MAF']].compute()

data_dir = params_data['training_data']['base_dir'].format(cohort=cohort)
write_dir = params_data['output']['base_dir'].format(cohort=cohort)

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

predictions_dir = f"{write_dir}/{params_data['output']['predictions_dir']}"
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

# Use configurable columns dictionary file
columns_dict_file = params_data['columns_dict_file']

# open pickle file as column_dict
with open(columns_dict_file, 'rb') as f:
    column_dict = pickle.load(f)


def make_variant_features(df):
    #split variant_id by
    df[['chr','pos','ref','alt']] = df['variant_id'].str.split(':', expand=True)
    # calculate difference between length ref and length alt
    df['length_diff'] = df['ref'].str.len() - df['alt'].str.len()
    df['is_SNP'] = df['length_diff'].apply(lambda x: 1 if x == 0 else 0)
    df['is_indel'] = df['length_diff'].apply(lambda x: 1 if x != 0 else 0)
    df['is_insertion'] = df['length_diff'].apply(lambda x: 1 if x > 0 else 0)
    df['is_deletion'] = df['length_diff'].apply(lambda x: 1 if x < 0 else 0)
    df.drop(columns=['chr','pos','ref','alt'], inplace=True)
    #make label the last column
    cols = df.columns.tolist()
    cols.insert(len(cols)-1, cols.pop(cols.index('label')))
    df = df.loc[:, cols]
    return df


# Load columns to remove from configuration
columns_to_remove = params_data['features']['columns_to_remove']

#######################################################STANDARD TRAINING DATA#######################################################
train_files = []
valid_train_chromosomes = []

for i in train_chromosomes:
    train_dir = params_data['training_data']['train_dir_pattern'].format(
        npr_tr=NPR_tr,
        pos_threshold=params_data['thresholds']['train']['positive_class_threshold'],
        neg_threshold=params_data['thresholds']['train']['negative_class_threshold']
    )
    file_pattern = params_data['training_data']['file_pattern'].format(cohort=cohort, chromosome=i)
    file_path = f'{data_dir}/{train_dir}/{file_pattern}'
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist. Skipping chromosome {i}.")
        continue
    # Add the file to the list
    train_files.append(file_path)
    valid_train_chromosomes.append(i)

if not train_files:
    raise ValueError("No training files found for any chromosome. Cannot proceed.")

# Read standard training data
train_df = dd.read_parquet(train_files, engine='pyarrow')
train_df = train_df.compute()

#train_df = train_df.drop(columns=residual_cols)

train_df = make_variant_features(train_df)


train_df = train_df.merge(gene_lof_df, on='gene_id', how='left')
train_df = train_df.merge(maf_df, on='variant_id', how='left')

#find rows with missing gene_lof

# Calculate imputation statistics from training data only to prevent data leakage
train_gene_lof_median = train_df['gene_lof'].median()
train_gnomad_maf_median = train_df['gnomad_MAF'].median()

print(f"Training data imputation - gene_lof median: {train_gene_lof_median:.6f}, gnomad_MAF median: {train_gnomad_maf_median:.6f}")

# Apply imputation using training statistics
train_df['gene_lof'] = train_df['gene_lof'].fillna(train_gene_lof_median)
train_df['gnomad_MAF'] = train_df['gnomad_MAF'].fillna(train_gnomad_maf_median)

#######################################################TEST DATA#######################################################
test_files = []
for i in test_chromosomes:
    test_dir = params_data['training_data']['test_dir_pattern'].format(
        npr_te=NPR_te,
        pos_threshold=params_data['thresholds']['test']['positive_class_threshold'],
        neg_threshold=params_data['thresholds']['test']['negative_class_threshold']
    )
    file_pattern = params_data['training_data']['file_pattern'].format(cohort=cohort, chromosome=i)
    file_path = f'{data_dir}/{test_dir}/{file_pattern}'
    # Check if file exists
    if os.path.exists(file_path):
        test_files.append(file_path)
    else:
        print(f"Warning: Test file {file_path} does not exist. Skipping chromosome {i}.")

if not test_files:
    raise ValueError("No test files found. Cannot proceed.")

test_df = dd.read_parquet(test_files, engine='pyarrow')
test_df = test_df.compute()

#test_df = test_df.drop(columns=residual_cols)

test_df = make_variant_features(test_df)


test_df = test_df.merge(gene_lof_df, on='gene_id', how='left')
test_df = test_df.merge(maf_df, on='variant_id', how='left')

# Apply imputation using training data statistics (prevent data leakage)
test_df['gene_lof'] = test_df['gene_lof'].fillna(train_gene_lof_median)
test_df['gnomad_MAF'] = test_df['gnomad_MAF'].fillna(train_gnomad_maf_median)

print(f"Applied training imputation statistics to test data")

##############################################################################################################
# Calculate weights for standard training data
train_class_0 = train_df[train_df['label'] == 0].shape[0]
train_class_1 = train_df[train_df['label'] == 1].shape[0]
train_total_pip = train_df[train_df['label'] == 1].pip.sum()
train_pip_percent = train_class_0 / train_total_pip if train_total_pip > 0 else 1

# Create a column called weight where everything with label = 0 has weight 1 and label = 1 has weight pip * train_pip_percent
train_df['weight'] = np.where(train_df['label'] == 0, 1, train_df['pip'] * train_pip_percent)

# Calculate weights for test data
test_class_0 = test_df[test_df['label'] == 0].shape[0]
test_class_1 = test_df[test_df['label'] == 1].shape[0]
test_total_pip = test_df[test_df['label'] == 1].pip.sum()
test_pip_percent = test_class_0 / test_total_pip if test_total_pip > 0 else 1

# Create a column called weight where everything with label = 0 has weight 1 and label = 1 has weight pip * test_pip_percent
test_df['weight'] = np.where(test_df['label'] == 0, 1, test_df['pip'] * test_pip_percent)

# Check weight distribution
print("Standard training data weight distribution:")
print(train_df.groupby('label')['weight'].sum())

print("Test data weight distribution:")
print(test_df.groupby('label')['weight'].sum())

##############################################################################################################
# Load meta data columns from configuration
meta_data = params_data['metadata_columns']

# Prepare standard training data
X_train = train_df.drop(columns=meta_data)
Y_train = train_df['label']
weight_train = train_df['weight']
X_train = X_train.replace([np.inf, -np.inf], 0)
X_train = X_train.fillna(0)

cols_order = X_train.columns.tolist()

# Prepare test data
X_test = test_df.drop(columns=meta_data)
Y_test = test_df['label']
weight_test = test_df['weight']
X_test = X_test.replace([np.inf, -np.inf], 0)
X_test = X_test.fillna(0)

# Remove gene_id if present
if 'gene_id' in X_train.columns:
    X_train = X_train.drop(columns=['gene_id'])
    X_test = X_test.drop(columns=['gene_id'])

# Print class distributions
print("Standard training data class distribution:")
print(Y_train.value_counts())

print("Test data class distribution:")
print(Y_test.value_counts())

##############################################################################################################
# Create subset of columns based on column_dict keys - load from configuration
subset_keys = params_data['features']['subset_keys']

# Extract columns for each subset
subset_cols = []
for key in subset_keys:
    if key in column_dict:
        subset_cols.extend(column_dict[key])

# Keep only columns that exist in the dataframes
subset_cols = [col for col in subset_cols if col in X_train.columns]

# Add variant features to subset columns - load from configuration
variant_features = params_data['features']['variant_features']
subset_cols.extend(variant_features)

# Apply absolute value to configured columns
columns_to_abs = []
for key in params_data['features']['absolute_value_keys']:
    if key in column_dict:
        columns_to_abs.extend([col for col in column_dict[key] if col in X_train.columns])

# Create subset dataframes with absolute values applied
X_train_subset = X_train[subset_cols].copy()
X_test_subset = X_test[subset_cols].copy()

# Apply absolute values only to the specified columns (not to variant features)
for col in columns_to_abs:
    if col in X_train_subset.columns:
        X_train_subset[col] = X_train_subset[col].abs()
        X_test_subset[col] = X_test_subset[col].abs()

# Drop columns from configuration
columns_to_drop = params_data['features']['columns_to_remove']
for col in columns_to_drop:
    if col in X_train_subset.columns:
        X_train_subset = X_train_subset.drop(columns=[col])
    if col in X_test_subset.columns:
        X_test_subset = X_test_subset.drop(columns=[col])
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])


##############################################################################################################
from catboost import CatBoostClassifier

# Load model parameters from config
original_params = model_params['catboost_params']['standard']
conservative_params = model_params['catboost_params']['conservative']

# Create feature weight dictionary from configuration
feature_weights = {}
default_weight = model_params['feature_weights']['default_weight']
high_weight_value = model_params['feature_weights']['high_weight_value']
high_priority_patterns = model_params['feature_weights']['high_priority_patterns']

# Set default weight for all features
for col in X_train_subset.columns:
    feature_weights[col] = default_weight

# Set high weight for priority features based on patterns
for col in X_train_subset.columns:
    if col in columns_to_abs:
        if any(pattern in col for pattern in high_priority_patterns):
            feature_weights[col] = high_weight_value

print("Feature weights distribution:")
print(f"Number of features with weight {high_weight_value}: {sum(value == high_weight_value for value in feature_weights.values())}")
print(f"Number of features with weight {default_weight}: {sum(value == default_weight for value in feature_weights.values())}")

# Initialize 4 models (only standard training data models)
# Model 1: Standard data, subset features (original params)
cat_standard_subset = CatBoostClassifier(
    **original_params,
    loss_function='Logloss',
    name="Standard-Subset"
)

# Model 3: Standard data, subset features (conservative params)
cat_standard_subset_conservative = CatBoostClassifier(
    **conservative_params,
    loss_function='Logloss',
    name="Standard-Subset-Conservative"
)

# Model 5: Standard data, subset features (original params) with feature weighting
cat_standard_subset_weighted = CatBoostClassifier(
    **original_params,
    loss_function='Logloss',
    feature_weights=feature_weights,
    name="Standard-Subset-Weighted"
)

# Model 7: Standard data, subset features (conservative params) with feature weighting
cat_standard_subset_conservative_weighted = CatBoostClassifier(
    **conservative_params,
    loss_function='Logloss',
    feature_weights=feature_weights,
    name="Standard-Subset-Conservative-Weighted"
)

# Train all models
print("Training model 1: Standard data, subset features (original params)")
cat_standard_subset.fit(X_train_subset, Y_train, sample_weight=weight_train)

print("Training model 3: Standard data, subset features (conservative params)")
cat_standard_subset_conservative.fit(X_train_subset, Y_train, sample_weight=weight_train)

print("Training model 5: Standard data, subset features (original params) with feature weighting")
cat_standard_subset_weighted.fit(X_train_subset, Y_train, sample_weight=weight_train)

print("Training model 7: Standard data, subset features (conservative params) with feature weighting")
cat_standard_subset_conservative_weighted.fit(X_train_subset, Y_train, sample_weight=weight_train)

# Calculate predictions for all models
preds_standard_subset = cat_standard_subset.predict_proba(X_test_subset)[:, 1]
preds_standard_subset_conservative = cat_standard_subset_conservative.predict_proba(X_test_subset)[:, 1]
preds_standard_subset_weighted = cat_standard_subset_weighted.predict_proba(X_test_subset)[:, 1]
preds_standard_subset_conservative_weighted = cat_standard_subset_conservative_weighted.predict_proba(X_test_subset)[:, 1]

# Calculate metrics for all models
metrics_dict = {
    'standard_subset': {
        'AP': metrics.average_precision_score(Y_test, preds_standard_subset),
        'AUC': metrics.roc_auc_score(Y_test, preds_standard_subset)
    },
    'standard_subset_conservative': {
        'AP': metrics.average_precision_score(Y_test, preds_standard_subset_conservative),
        'AUC': metrics.roc_auc_score(Y_test, preds_standard_subset_conservative)
    },
    'standard_subset_weighted': {
        'AP': metrics.average_precision_score(Y_test, preds_standard_subset_weighted),
        'AUC': metrics.roc_auc_score(Y_test, preds_standard_subset_weighted)
    },
    'standard_subset_conservative_weighted': {
        'AP': metrics.average_precision_score(Y_test, preds_standard_subset_conservative_weighted),
        'AUC': metrics.roc_auc_score(Y_test, preds_standard_subset_conservative_weighted)
    }
}

# Print metrics for all models
print("\nTest Set Metrics:")
print(f"1. Standard data, subset features (original) - AP: {metrics_dict['standard_subset']['AP']:.4f}, AUC: {metrics_dict['standard_subset']['AUC']:.4f}")
print(f"3. Standard data, subset features (conservative) - AP: {metrics_dict['standard_subset_conservative']['AP']:.4f}, AUC: {metrics_dict['standard_subset_conservative']['AUC']:.4f}")
print(f"5. Standard data, subset features (original) weighted - AP: {metrics_dict['standard_subset_weighted']['AP']:.4f}, AUC: {metrics_dict['standard_subset_weighted']['AUC']:.4f}")
print(f"7. Standard data, subset features (conservative) weighted - AP: {metrics_dict['standard_subset_conservative_weighted']['AP']:.4f}, AUC: {metrics_dict['standard_subset_conservative_weighted']['AUC']:.4f}")

# Save all models
joblib.dump(cat_standard_subset, f'{write_dir}/model_standard_subset_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')
joblib.dump(cat_standard_subset_conservative, f'{write_dir}/model_standard_subset_conservative_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')
joblib.dump(cat_standard_subset_weighted, f'{write_dir}/model_standard_subset_weighted_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')
joblib.dump(cat_standard_subset_conservative_weighted, f'{write_dir}/model_standard_subset_conservative_weighted_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')

# Get feature importances for all models
feature_importances = {
    'standard_subset': {
        'importances': cat_standard_subset.feature_importances_,
        'features': X_train_subset.columns
    },
    'standard_subset_conservative': {
        'importances': cat_standard_subset_conservative.feature_importances_,
        'features': X_train_subset.columns
    },
    'standard_subset_weighted': {
        'importances': cat_standard_subset_weighted.feature_importances_,
        'features': X_train_subset.columns
    },
    'standard_subset_conservative_weighted': {
        'importances': cat_standard_subset_conservative_weighted.feature_importances_,
        'features': X_train_subset.columns
    }
}

# Create individual dataframes for each model
feature_dfs = []
for model_name, model_data in feature_importances.items():
    features = model_data['features']
    importances = model_data['importances']
    model_df = pd.DataFrame({
        'feature': features,
        'importance': importances,
        'model': model_name
    })
    model_df = model_df.sort_values(by='importance', ascending=False)
    feature_dfs.append(model_df)
    # Print top 20 features for each model
    print(f"\nTop 20 features for {model_name}:")
    for i, (feature, importance) in enumerate(zip(model_df['feature'][:20], model_df['importance'][:20])):
        print(f"{i + 1}. {feature}: {importance:.6f}")

# Combine all feature importance dataframes
all_features_df = pd.concat(feature_dfs, ignore_index=True)

# Save the combined feature importance dataframe
all_features_df.to_csv(f'{write_dir}/features_importance_4models_chr_{chromosome_out}_NPR_{NPR_tr}.csv',
                       index=False)

# Create summary dictionary with all models' parameters and metrics
summary_dict = {
    'CatBoost': {
        'standard_subset': {
            'AP_test': metrics_dict['standard_subset']['AP'],
            'AUC_test': metrics_dict['standard_subset']['AUC'],
            'params': original_params
        },
        'standard_subset_conservative': {
            'AP_test': metrics_dict['standard_subset_conservative']['AP'],
            'AUC_test': metrics_dict['standard_subset_conservative']['AUC'],
            'params': conservative_params
        },
        'standard_subset_weighted': {
            'AP_test': metrics_dict['standard_subset_weighted']['AP'],
            'AUC_test': metrics_dict['standard_subset_weighted']['AUC'],
            'params': original_params,
            'feature_weights': f'{", ".join(high_priority_patterns)} features set to {high_weight_value}, others to {default_weight}'
        },
        'standard_subset_conservative_weighted': {
            'AP_test': metrics_dict['standard_subset_conservative_weighted']['AP'],
            'AUC_test': metrics_dict['standard_subset_conservative_weighted']['AUC'],
            'params': conservative_params,
            'feature_weights': f'{", ".join(high_priority_patterns)} features set to {high_weight_value}, others to {default_weight}'
        },
        'test_num_positive_labels': Y_test.value_counts().get(1, 0),
        'test_num_negative_labels': Y_test.value_counts().get(0, 0),
        'train_standard_num_positive_labels': Y_train.value_counts().get(1, 0),
        'train_standard_num_negative_labels': Y_train.value_counts().get(0, 0)
    }
}

print('Writing results to file')

# Write summary_dict to pickle file
with open(f'{write_dir}/summary_dict_catboost_4models_chr_{chromosome_out}_NPR_{NPR_tr}.pkl', 'wb') as f:
    pickle.dump(summary_dict, f)

# Add predictions to test_df and actual labels
test_df['standard_subset_pred_prob'] = preds_standard_subset
test_df['standard_subset_pred_label'] = cat_standard_subset.predict(X_test_subset)

test_df['standard_subset_conservative_pred_prob'] = preds_standard_subset_conservative
test_df['standard_subset_conservative_pred_label'] = cat_standard_subset_conservative.predict(X_test_subset)

test_df['standard_subset_weighted_pred_prob'] = preds_standard_subset_weighted
test_df['standard_subset_weighted_pred_label'] = cat_standard_subset_weighted.predict(X_test_subset)

test_df['standard_subset_conservative_weighted_pred_prob'] = preds_standard_subset_conservative_weighted
test_df['standard_subset_conservative_weighted_pred_label'] = cat_standard_subset_conservative_weighted.predict(
    X_test_subset)

test_df['actual_label'] = Y_test  # Add actual labels to test_df

# Save the test_df with all 4 models' predictions
test_df.to_csv(f'{write_dir}/predictions_parquet_catboost/predictions_4models_chr{chromosome}.tsv', sep='\t',
               index=False)

# Save the feature weights dictionary for reference
with open(f'{write_dir}/feature_weights_chr_{chromosome}_NPR_{NPR_tr}.pkl', 'wb') as f:
    pickle.dump(feature_weights, f)

# Save the subset columns list for future reference
with open(f'{write_dir}/subset_columns_chr_{chromosome}_NPR_{NPR_tr}.pkl', 'wb') as f:
    pickle.dump({
        'subset_columns': subset_cols,
        'abs_columns': columns_to_abs
    }, f)

print("\nTraining and evaluation complete for 4 standard models (1, 3, 5, 7).")