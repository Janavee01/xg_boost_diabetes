

import os
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import json

warnings.filterwarnings('ignore')
RANDOM_STATE = 42


def load_csv_maybe_zip(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} not found")
    # If it's a zip file, try to find a csv inside
    if path.suffix == '.zip':
        import zipfile
        with zipfile.ZipFile(path, 'r') as z:
            # pick the first csv
            names = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not names:
                raise ValueError('No CSV found inside the zip')
            with z.open(names[0]) as f:
                df = pd.read_csv(f, low_memory=False)
    else:
        # try several encodings to be safe
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError('Could not read CSV with tried encodings')
    return df


def quick_clean(df, drop_thresh=0.99):
    # drop constant columns or mostly-empty
    nunique = df.nunique(dropna=True)
    to_drop = nunique[nunique <= 1].index.tolist()
    # drop columns with too many nulls
    null_frac = df.isna().mean()
    to_drop += null_frac[null_frac > drop_thresh].index.tolist()
    df = df.drop(columns=list(set(to_drop)), errors='ignore')
    # fill numeric nulls with median, categorical with mode
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'NA')
    return df


def main(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print('Loading data...')
    df = load_csv_maybe_zip(args.input)
    print(f'Data shape before cleaning: {df.shape}')

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in input CSV")

    # quick clean
    df = quick_clean(df)
    print(f'Data shape after cleaning: {df.shape}')

    # Separate X,y
    y = df[args.target].copy()
    X = df.drop(columns=[args.target])

    # encode categorical columns if any (simple): one-hot would explode; instead use ordinal-ish label encoding
    # but to keep things simple and fast we use pandas factorize for object columns
    obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f'Found {len(obj_cols)} object/category columns. Factorizing...')
    for c in obj_cols:
        X[c], _ = pd.factorize(X[c])

    # ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y)

    # compute scale_pos_weight for XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    if pos == 0:
        raise ValueError('No positive samples in training set')
    scale_pos_weight = neg / pos
    print(f'Class balance in train -> neg: {neg}, pos: {pos}, scale_pos_weight: {scale_pos_weight:.3f}')

    # set a compact but meaningful param distribution
    param_dist = {
        'n_estimators': [100, 200, 400, 800],
        'max_depth': [3, 4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0.5, 1, 2, 5],
        'min_child_weight': [1, 3, 5, 10]
    }

    base_clf = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='hist',  # faster on many features
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs
    )

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    print('Starting RandomizedSearchCV (this will take a bit)...')
    rs = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring='roc_auc',
        n_jobs=args.n_jobs,
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True
    )

    rs.fit(X_train, y_train)

    print('\nBest params:')
    print(rs.best_params_)
    print(f'Best CV ROC-AUC: {rs.best_score_:.4f}')

    # Save best params and cv results
    with open(outdir / 'search_results.json', 'w') as f:
        json.dump({'best_params': rs.best_params_, 'best_score': float(rs.best_score_)}, f, indent=2)

    # Train final model with early stopping on a validation split
    print('\nTraining final model with early stopping...')
    final_params = rs.best_params_.copy()
    final_params.update({'objective': 'binary:logistic', 'use_label_encoder': False, 'eval_metric': 'auc', 'tree_method': 'hist', 'random_state': RANDOM_STATE, 'n_jobs': args.n_jobs})

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train)

    final_clf = XGBClassifier(**final_params)
    final_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    # Evaluate on test
    y_proba = final_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= args.threshold).astype(int)

    roc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    print(f'Final test ROC-AUC: {roc:.4f}')
    print(f'Final test AUPRC : {auprc:.4f}')
    print(f'Final test Accuracy: {acc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Save metrics
    metrics = {'roc_auc': float(roc), 'auprc': float(auprc), 'accuracy': float(acc)}
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save model
    joblib.dump(final_clf, outdir / 'final_xgb_model.joblib')

    # Save predictions
    preds_df = X_test.copy()
    preds_df[args.target] = y_test.values
    preds_df['pred_proba'] = y_proba
    preds_df['pred_label'] = y_pred
    preds_df.to_csv(outdir / 'test_predictions.csv', index=False)

    # Feature importances (gain)
    try:
        booster = final_clf.get_booster()
        fmap = booster.get_score(importance_type='gain')
        fmap_items = sorted(fmap.items(), key=lambda x: x[1], reverse=True)
        fi_df = pd.DataFrame(fmap_items, columns=['feature', 'gain'])
        fi_df.to_csv(outdir / 'feature_importances_gain.csv', index=False)
        print(f'Wrote feature importances to {outdir / "feature_importances_gain.csv"}')
    except Exception as e:
        print('Could not extract feature importances:', e)

    print(f'All outputs written to {outdir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to feature-engineered CSV or ZIP')
    parser.add_argument('--target', type=str, default='Readmitted_30d', help='Name of target column')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--n_iter', type=int, default=30, help='RandomizedSearch n_iter')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Parallel jobs')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold for probability -> label')
    args = parser.parse_args()
    main(args)
