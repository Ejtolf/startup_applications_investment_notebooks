import os
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# Classification inference function
# Fits the model, optimizing by cross validation, validates and counts scores.
def train_evaluate_model(
    model_name,
    model,
    param_grid,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    step_name,
    scoring="roc_auc"
):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    os.makedirs("artifacts/reports", exist_ok=True)

    with mlflow.start_run(run_name=model_name):

        print("\n" + "=" * 60)
        print(f"TRAINING: {model_name}")
        print("=" * 60)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (step_name, model)
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)

        y_score = None
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_test)

        # Metrics
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        roc_auc = None
        if y_score is not None:
            roc_auc = roc_auc_score(y_test, y_score)

        # MLflow logging
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("cv_roc_auc", grid_search.best_score_)
        mlflow.log_metric("test_f1", f1)

        if roc_auc is not None:
            mlflow.log_metric("test_roc_auc", roc_auc)

        report_path = f"artifacts/reports/{model_name}_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"ROC-AUC (CV): {grid_search.best_score_}\n")
            f.write(f"ROC-AUC (TEST): {roc_auc}\n")
            f.write(f"F1 (TEST): {f1}\n\n")
            f.write(report)

        mlflow.log_artifact(report_path)
        mlflow.sklearn.log_model(best_model, name="model")

        return {
            "model": best_model,
            "cv_score": grid_search.best_score_,
            "roc_auc": roc_auc,
            "f1": f1,
            "report": report,
            "confusion_matrix": cm,
            "probabilities": y_score
        }