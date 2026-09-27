# train_models.py - OptiCrop V2 Model Comparison & Training Pipeline
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset", "Crop_recommendation.csv")
    static_dir = os.path.join(base_dir, "static")
    os.makedirs(static_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    data = pd.read_csv(data_path)
    
    X = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB()
    }

    comparison_results = []
    trained_models = {}

    print("\n" + "="*75)
    print(f'{"Model":<22} | {"Accuracy":<10} | {"Precision":<10} | {"Recall":<10} | {"F1-Score":<10}')
    print("="*75)

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        comparison_results.append({
            "model": name,
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "accuracy_raw": float(acc),
            "precision_raw": float(prec),
            "recall_raw": float(rec),
            "f1_score_raw": float(f1),
            "precision_macro": round(float(prec_macro) * 100, 2),
            "recall_macro": round(float(rec_macro) * 100, 2),
            "f1_macro": round(float(f1_macro) * 100, 2)
        })

        print(f"{name:<22} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {f1*100:>8.2f}%")

    print("="*75)

    sorted_results = sorted(comparison_results, key=lambda x: (x["accuracy_raw"], x["f1_score_raw"]), reverse=True)
    best_model_name = sorted_results[0]["model"]
    best_model = trained_models[best_model_name]
    best_metrics = sorted_results[0]

    print(f"\nBest Performing Model: {best_model_name} (Accuracy: {best_metrics['accuracy']}%, F1: {best_metrics['f1_score']}%)")

    model_save_path = os.path.join(base_dir, "model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Successfully saved final model to: {model_save_path}")

    json_path = os.path.join(static_dir, "model_comparison.json")
    summary_data = {
        "best_model": best_model_name,
        "results": comparison_results,
        "feature_names": list(X.columns),
        "total_classes": len(best_model.classes_),
        "classes": list(best_model.classes_)
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Comparison JSON saved to: {json_path}")

    chart_path = os.path.join(static_dir, "model_comparison.png")
    df_chart = pd.DataFrame(comparison_results)
    
    plt.figure(figsize=(11, 5.5), dpi=150)
    sns.set_theme(style="whitegrid")

    x = np.arange(len(df_chart["model"]))
    width = 0.18

    plt.bar(x - 1.5*width, df_chart["accuracy"], width, label="Accuracy", color="#1b5e20")
    plt.bar(x - 0.5*width, df_chart["precision"], width, label="Precision", color="#2e7d32")
    plt.bar(x + 0.5*width, df_chart["recall"], width, label="Recall", color="#66bb6a")
    plt.bar(x + 1.5*width, df_chart["f1_score"], width, label="F1-Score", color="#ff9800")

    plt.ylabel("Score (%)", fontsize=12, fontweight="bold", color="#1b5e20")
    plt.title("OptiCrop V2 — ML Classification Models Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, df_chart["model"], fontsize=10, fontweight="600")
    plt.ylim(90, 101)
    plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to: {chart_path}")

if __name__ == "__main__":
    main()
