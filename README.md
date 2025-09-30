# setup_advanced_credit_app.py
import os, textwrap, sys, subprocess

PROJECT = "credit_scoring_app"

files = {
    # requirements
    os.path.join(PROJECT, "requirements.txt"): textwrap.dedent("""\
        fastapi==0.95.2
        uvicorn[standard]==0.22.0
        scikit-learn==1.2.2
        pandas==2.2.2
        numpy==1.26.2
        joblib==1.3.2
        pydantic==1.10.11
        lightgbm==4.4.0
        shap==0.41.0
        matplotlib==3.8.0
        chart-studio==1.1.0
        python-multipart==0.0.6
        aiofiles==23.1.0
    """),

    # backend/sample_data_generator.py
    os.path.join(PROJECT, "backend", "sample_data_generator.py"): textwrap.dedent("""\
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        import random

        np.random.seed(42)
        random.seed(42)

        N = 5000

        # Basic financial signals
        income = np.random.normal(60000, 20000, N).clip(8000, None).round(2)
        debts = np.random.normal(15000, 12000, N).clip(0, None).round(2)
        credit_limits = (np.random.normal(20000, 8000, N).clip(1000, None)).round(2)
        used_ratio = (debts / (credit_limits + 1)).clip(0, 2).round(3)

        # behavioral / time features
        avg_monthly_spend = (income * np.random.uniform(0.05, 0.5, N)).round(2)
        months_since_last_default = np.random.exponential(scale=24, size=N).astype(int)
        late_payments_12m = np.random.poisson(0.8, N)
        missed_payments_24m = np.random.poisson(0.5, N)

        # categorical signals
        employment_status = np.random.choice(['employed', 'self-employed', 'unemployed', 'retired'], N, p=[0.62,0.12,0.2,0.06])
        residence_status = np.random.choice(['rent', 'mortgage', 'own', 'other'], N, p=[0.4,0.4,0.18,0.02])
        education = np.random.choice(['high_school', 'bachelor', 'master', 'phd', 'other'], N, p=[0.3,0.45,0.18,0.04,0.03])

        # account age in months
        account_age_months = np.random.randint(1, 300, N)

        # a synthetic 'payment history score' [0..100]
        payment_history_score = (100 - (late_payments_12m * 8) - (missed_payments_24m * 6) - (np.clip(used_ratio, 0, 2)*15)).clip(1, 99).round(1)

        # crude target: combine features into risk score then threshold
        raw_score = (income / (debts + 1000)) + (payment_history_score / 10) + (account_age_months / 120) - (late_payments_12m * 0.8)
        prob = 1 / (1 + np.exp(-(raw_score - np.median(raw_score)) / 0.6))
        target = (prob > np.quantile(prob, 0.48)).astype(int)  # ~balanced with slight bias

        df = pd.DataFrame({
            'income': income,
            'debts': debts,
            'credit_limits': credit_limits,
            'used_ratio': used_ratio,
            'avg_monthly_spend': avg_monthly_spend,
            'months_since_last_default': months_since_last_default,
            'late_payments_12m': late_payments_12m,
            'missed_payments_24m': missed_payments_24m,
            'employment_status': employment_status,
            'residence_status': residence_status,
            'education': education,
            'account_age_months': account_age_months,
            'payment_history_score': payment_history_score,
            'target': target
        })

        # Shuffle and export
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        out_path = "credit_scoring_app/backend/sample_credit_data.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Wrote synthetic dataset to {out_path} with {len(df)} rows")
    """),

    # backend/utils.py
    os.path.join(PROJECT, "backend", "utils.py"): textwrap.dedent("""\
        import json, os
        from typing import Dict
        import numpy as np
        import pandas as pd

        METRICS_PATH = "credit_scoring_app/backend/models/metrics.json"

        def save_metrics(metrics: Dict):
            os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
            with open(METRICS_PATH, "w") as f:
                json.dump(metrics, f, indent=2)
            print("Saved metrics to", METRICS_PATH)

        def load_metrics():
            if not os.path.exists(METRICS_PATH):
                return {}
            with open(METRICS_PATH, "r") as f:
                return json.load(f)

        def top_features_from_importances(feature_names, importances, top_n=20):
            idx = np.argsort(importances)[::-1][:top_n]
            return [(feature_names[i], float(importances[i])) for i in idx]
    """),

    # backend/train_model.py
    os.path.join(PROJECT, "backend", "train_model.py"): textwrap.dedent("""\
        import pandas as pd
        import numpy as np
        import joblib, os, json
        from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier
        from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
        from utils import save_metrics, top_features_from_importances

        DATA_PATH = "credit_scoring_app/backend/sample_credit_data.csv"
        MODEL_DIR = "credit_scoring_app/backend/models"
        PIPE_PATH = os.path.join(MODEL_DIR, "pipeline.joblib")
        FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importances.json")

        def feature_engineering(df):
            # Example engineered features
            df = df.copy()
            df['debt_to_income'] = df['debts'] / (df['income'] + 1)
            df['spend_to_income'] = df['avg_monthly_spend'] / (df['income'] + 1)
            df['recent_default_flag'] = (df['months_since_last_default'] < 12).astype(int)
            # bucket account age
            df['account_age_bucket'] = pd.cut(df['account_age_months'], bins=[0,12,36,72,120,10000], labels=['<1y','1-3y','3-6y','6-10y','10y+'])
            return df

        def build_pipeline(numeric_features, categorical_features):
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='drop')

            # Stack a LightGBM and a RandomForest via simple ensemble using probabilities average
            lgb = LGBMClassifier(random_state=42, n_jobs= -1)
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)

            # We'll train a single pipeline for convenience with lgb as primary classifier
            pipeline = Pipeline(steps=[('pre', preprocessor), ('clf', lgb)])
            return pipeline, rf, lgb, preprocessor

        def train():
            print("Loading data from", DATA_PATH)
            df = pd.read_csv(DATA_PATH)
            df = feature_engineering(df)

            target = 'target'
            drop_cols = []
            features = [c for c in df.columns if c != target and c not in drop_cols]
            X = df[features]
            y = df[target]

            # identify types
            numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

            pipeline, rf_template, lgb_template, preprocessor = build_pipeline(numeric_features, categorical_features)

            # split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # simple hyperparameter grid for LGBM
            param_grid = {
                'clf__num_leaves': [31, 63],
                'clf__n_estimators': [100, 300],
                'clf__learning_rate': [0.05, 0.1]
            }

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            gs = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
            print("Starting GridSearchCV (this may take several minutes)...")
            gs.fit(X_train, y_train)

            best = gs.best_estimator_
            print("Best params:", gs.best_params_)
            print("Training complete. Evaluating on test set...")

            # predictions
            probs = best.predict_proba(X_test)[:,1]
            preds = (probs >= 0.5).astype(int)

            metrics = {}
            metrics['classification_report'] = classification_report(y_test, preds, output_dict=True)
            metrics['roc_auc'] = float(roc_auc_score(y_test, probs))
            metrics['precision'] = float(precision_score(y_test, preds))
            metrics['recall'] = float(recall_score(y_test, preds))
            metrics['f1'] = float(f1_score(y_test, preds))
            metrics['confusion_matrix'] = confusion_matrix(y_test, preds).tolist()

            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(best, PIPE_PATH)
            print("Saved pipeline to", PIPE_PATH)

            # feature importance (if classifier supports it)
            try:
                # build feature names after preprocessing
                # get numeric names
                numeric_names = list(numeric_features)
                # get onehot encoded names
                ohe = best.named_steps['pre'].transformers_[1][1].named_steps['onehot']
                cat_cols = categorical_features
                if hasattr(ohe, 'get_feature_names_out'):
                    cat_names = list(ohe.get_feature_names_out(cat_cols))
                else:
                    # fallback
                    cat_names = cat_cols
                feature_names = numeric_names + cat_names
                # try to pull importances
                clf = best.named_steps['clf']
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                    top = top_features_from_importances(feature_names, importances, top_n=40)
                    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
                        json.dump({"top_features": top}, f, indent=2)
                    print("Saved feature importances to", FEATURE_IMPORTANCE_PATH)
            except Exception as e:
                print("Could not compute feature importances:", e)

            # save metrics
            save_metrics(metrics)
            print("Training finished. Metrics:")
            print(json.dumps(metrics, indent=2))

        if __name__ == "__main__":
            train()
    """),

    # backend/app.py
    os.path.join(PROJECT, "backend", "app.py"): textwrap.dedent("""\
        import joblib, os, json
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
        import pandas as pd
        from utils import load_metrics
        import shap
        import numpy as np

        BASE = "credit_scoring_app/backend"
        PIPE_PATH = os.path.join(BASE, "models", "pipeline.joblib")
        METRICS_PATH = os.path.join(BASE, "models", "metrics.json")

        app = FastAPI(title="Advanced Credit Scoring API", version="1.0")

        # Input model
        class CreditRequest(BaseModel):
            income: float = Field(..., example=60000)
            debts: float = Field(..., example=10000)
            credit_limits: float = Field(..., example=20000)
            avg_monthly_spend: float = Field(..., example=2500)
            months_since_last_default: int = Field(..., example=24)
            late_payments_12m: int = Field(..., example=0)
            missed_payments_24m: int = Field(..., example=0)
            employment_status: str = Field(..., example="employed")
            residence_status: str = Field(..., example="rent")
            education: str = Field(..., example="bachelor")
            account_age_months: int = Field(..., example=36)
            payment_history_score: float = Field(..., example=85.0)

        # Load model lazily
        model = None
        explainer = None
        def load_model():
            global model, explainer
            if model is None:
                if not os.path.exists(PIPE_PATH):
                    raise FileNotFoundError("Model not found. Train first by running train_model.py")
                model = joblib.load(PIPE_PATH)
                # create a small explainer if possible
                try:
                    # use a small sample to initialize explainer
                    sample = pd.read_csv(os.path.join(BASE, "sample_credit_data.csv")).sample(200, random_state=42)
                    sample = sample.drop(columns=['target'])
                    # apply same feature engineering as training
                    sample['debt_to_income'] = sample['debts'] / (sample['income'] + 1)
                    sample['spend_to_income'] = sample['avg_monthly_spend'] / (sample['income'] + 1)
                    explainer = shap.Explainer(model.predict, sample, algorithm='permutation')
                except Exception as e:
                    print("SHAP explainer not initialized:", e)

        @app.get("/health")
        def health():
            return {"status":"ok", "model_available": os.path.exists(PIPE_PATH)}

        @app.post("/predict")
        def predict(req: CreditRequest):
            load_model()
            # convert to df and add engineered features (same as training)
            row = pd.DataFrame([req.dict()])
            row['debt_to_income'] = row['debts'] / (row['income'] + 1)
            row['spend_to_income'] = row['avg_monthly_spend'] / (row['income'] + 1)
            row['recent_default_flag'] = (row['months_since_last_default'] < 12).astype(int)
            # ensure same columns order as training expectation - model handles via pipeline
            try:
                proba = float(model.predict_proba(row)[0][1])
                pred = int(proba >= 0.5)
                score = round(proba * 100, 2)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            return {"prediction": pred, "probability": round(proba, 4), "score": score}

        @app.get("/metrics")
        def metrics():
            if not os.path.exists(METRICS_PATH):
                raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
            with open(METRICS_PATH, "r") as f:
                data = json.load(f)
            return data

        @app.post("/explain")
        def explain(req: CreditRequest):
            # returns SHAP values for the instance if explainer available
            load_model()
            global explainer
            if explainer is None:
                raise HTTPException(status_code=503, detail="Explainer not available")
            row = pd.DataFrame([req.dict()])
            row['debt_to_income'] = row['debts'] / (row['income'] + 1)
            row['spend_to_income'] = row['avg_monthly_spend'] / (row['income'] + 1)
            shap_values = explainer(row)
            # build simple response
            feature_names = list(row.columns)
            vals = []
            for name, val, s in zip(feature_names, row.iloc[0].values, shap_values.values[0]):
                vals.append({"feature": name, "value": float(val), "shap": float(s)})
            return {"shap": vals}
    """),

    # frontend/index.html
    os.path.join(PROJECT, "frontend", "index.html"): textwrap.dedent("""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width,initial-scale=1"/>
          <title>Credit Scoring — Demo</title>
          <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
          <link rel="stylesheet" href="style.css">
        </head>
        <body>
          <div class="container py-4">
            <div class="row">
              <div class="col-md-6">
                <div class="card shadow-sm p-3">
                  <h3 class="mb-3">Credit Scoring Form</h3>
                  <form id="creditForm">
                    <div class="mb-2"><label>Income</label><input name="income" class="form-control" type="number" value="60000" required></div>
                    <div class="mb-2"><label>Debts</label><input name="debts" class="form-control" type="number" value="12000" required></div>
                    <div class="mb-2"><label>Credit Limit</label><input name="credit_limits" class="form-control" type="number" value="20000" required></div>
                    <div class="mb-2"><label>Avg monthly spend</label><input name="avg_monthly_spend" class="form-control" type="number" value="2500" required></div>
                    <div class="mb-2"><label>Months since last default</label><input name="months_since_last_default" class="form-control" type="number" value="24" required></div>
                    <div class="mb-2"><label>Late payments (12m)</label><input name="late_payments_12m" class="form-control" type="number" value="0" required></div>
                    <div class="mb-2"><label>Missed payments (24m)</label><input name="missed_payments_24m" class="form-control" type="number" value="0" required></div>
                    <div class="mb-2"><label>Employment status</label>
                      <select name="employment_status" class="form-select">
                        <option>employed</option><option>self-employed</option><option>unemployed</option><option>retired</option>
                      </select>
                    </div>
                    <div class="mb-2"><label>Residence status</label>
                      <select name="residence_status" class="form-select">
                        <option>rent</option><option>mortgage</option><option>own</option><option>other</option>
                      </select>
                    </div>
                    <div class="mb-2"><label>Education</label>
                      <select name="education" class="form-select">
                        <option>high_school</option><option>bachelor</option><option>master</option><option>phd</option><option>other</option>
                      </select>
                    </div>
                    <div class="mb-2"><label>Account age (months)</label><input name="account_age_months" class="form-control" type="number" value="36" required></div>
                    <div class="mb-2"><label>Payment history score (0-100)</label><input name="payment_history_score" class="form-control" type="number" value="85" required></div>

                    <button class="btn btn-primary mt-2" type="submit">Predict</button>
                  </form>
                  <div id="resultCard" class="mt-3"></div>
                </div>
              </div>

              <div class="col-md-6">
                <div class="card shadow-sm p-3">
                  <h4>Model Metrics</h4>
                  <div id="metricsArea">
                    <p>Click <strong>Load Metrics</strong> to fetch model evaluation stats after training.</p>
                    <button id="loadMetrics" class="btn btn-outline-secondary btn-sm">Load Metrics</button>
                    <div id="metricsJson" class="mt-2"></div>
                  </div>

                  <h5 class="mt-3">SHAP Explanation</h5>
                  <div id="shapArea">Submit a prediction and click "Explain" to see feature contributions.</div>
                
