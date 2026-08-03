from pathlib import Path

import joblib

from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from modeling_evaluation import evaluate_classifier, train_random_forest

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

def main():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    db_path = "data/inventory.db"
    scaler_path = Path("models")

    # Load data
    df = load_invoice_data(db_path)
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, scaler_path)

    # Train and evaluate model
    search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save best model
    joblib.dump(search.best_estimator_, model_dir/"predict_invoice_flag.pkl")
    print(f"\nBest model saved to {model_dir}/predict_invoice_flag.pkl")

if __name__ == "__main__":
    main()