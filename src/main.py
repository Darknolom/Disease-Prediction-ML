

import argparse
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def load_data(data_dir):
    """Загрузка и подготовка данных из Training.csv и Testing.csv"""
    train_path = os.path.join(data_dir, "Training.csv")
    test_path = os.path.join(data_dir, "Testing.csv")

    print(f"Загрузка данных из {data_dir}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Удаление лишнего столбца (как в оригинальном ноутбуке)
    if "Unnamed: 133" in df_train.columns:
        df_train.drop("Unnamed: 133", axis=1, inplace=True)
    if "Unnamed: 133" in df_test.columns:
        df_test.drop("Unnamed: 133", axis=1, inplace=True)

    print(f"Размер обучающих данных: {df_train.shape}")
    print(f"Размер тестовых данных: {df_test.shape}")
    print(f"Кол-во уникальных диагнозов: {df_train['prognosis'].nunique()}")

    return df_train, df_test


def preprocess_data(df_train, df_test, seed):
    """Предобработка: кодирование меток, разделение на признаки/целевую"""
    X_train_full = df_train.drop("prognosis", axis=1)
    y_train_full = df_train["prognosis"]

    X_test_ind = df_test.drop("prognosis", axis=1)
    y_test_ind = df_test["prognosis"]

    # Кодирование меток
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_full)
    y_test_enc = le.transform(y_test_ind)

    print(f"Классы: {len(le.classes_)}")

    # Разделение 80/20
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_enc,
        test_size=0.2, random_state=seed, stratify=y_train_enc
    )

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер валидационной выборки: {X_val.shape}")

    return X_train, X_val, y_train, y_val, X_test_ind, y_test_enc, le


def get_model_and_params(model_name, seed):
    """Возвращает модель и сетку параметров для GridSearchCV"""
    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=seed),
            {"C": [0.01, 0.1, 1, 10]}
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed),
            {"max_depth": [5, 10, 15, None], "min_samples_split": [2, 5, 10]}
        ),
        "knn": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed),
            {"n_estimators": [100, 200, 300], "max_depth": [10, 20, None]}
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"n_estimators": [100], "learning_rate": [0.05], "max_depth": [3]}
        ),
        "xgboost": (
            XGBClassifier(
                objective="multi:softmax",
                eval_metric="mlogloss",
                random_state=seed,
                use_label_encoder=False
            ),
            {"n_estimators": [100], "max_depth": [4], "learning_rate": [0.01, 0.1]}
        ),
    }

    if model_name not in models:
        raise ValueError(
            f"Модель '{model_name}' не поддерживается. "
            f"Доступные: {list(models.keys())}"
        )
    return models[model_name]


def evaluate_model(model, X_test, y_test, model_name):
    """Оценка модели и возврат метрик"""
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average="weighted")
    rec = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")

    return {
        "model": model_name,
        "accuracy": round(acc, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1_score": round(f1, 5)
    }


def train_single_model(model_name, X_train, X_val, y_train, y_val,
                       X_test_ind, y_test_enc, seed, model_dir, output_dir):
    """Обучение одной модели с GridSearchCV"""
    print(f"\nПодбор {model_name}...")

    model, param_grid = get_model_and_params(model_name, seed)

    grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Лучшие параметры: {grid.best_params_}")

    # Оценка на валидации (20%)
    results_val = evaluate_model(best_model, X_val, y_val, f"{model_name} (val 20%)")

    # Оценка на независимом Testing.csv
    results_test = evaluate_model(best_model, X_test_ind, y_test_enc, f"{model_name} (Testing.csv)")

    # Сохранение модели
    model_path = os.path.join(model_dir, f"{model_name}_seed{seed}.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nМодель сохранена: {model_path}")

    # Сохранение результатов
    result_data = {
        "model": model_name,
        "seed": seed,
        "best_params": str(grid.best_params_),
        "validation_results": results_val,
        "testing_results": results_test
    }

    result_path = os.path.join(output_dir, f"{model_name}_seed{seed}_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"Результаты сохранены: {result_path}")

    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="ML-пайплайн предсказания заболеваний по симптомам"
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        help="Имя модели или 'all' для обучения всех моделей"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Путь к папке с данными"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Путь к папке для результатов"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./models",
        help="Путь к папке для моделей"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Автоматическое создание каталогов (Пункт 9 задания)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print("="*60)
    print("  ML-ПАЙПЛАЙН: ПРЕДСКАЗАНИЕ ЗАБОЛЕВАНИЙ")
    print("="*60)
    print(f"  Модель: {args.model}")
    print(f"  Seed: {args.seed}")
    print(f"  Данные: {args.data_dir}")
    print(f"  Результаты: {args.output_dir}")
    print("="*60)

    # Загрузка данных
    df_train, df_test = load_data(args.data_dir)

    # Предобработка
    X_train, X_val, y_train, y_val, X_test_ind, y_test_enc, le = \
        preprocess_data(df_train, df_test, args.seed)

    # Сохранение LabelEncoder
    le_path = os.path.join(args.model_dir, "label_encoder.joblib")
    joblib.dump(le, le_path)

    # Обучение моделей
    all_models = [
        "logistic_regression", "decision_tree", "knn",
        "random_forest", "gradient_boosting", "xgboost"
    ]

    if args.model == "all":
        models_to_train = all_models
    else:
        models_to_train = [args.model]

    all_results = []
    for model_name in models_to_train:
        result = train_single_model(
            model_name, X_train, X_val, y_train, y_val,
            X_test_ind, y_test_enc, args.seed,
            args.model_dir, args.output_dir
        )
        all_results.append(result)

    # Сохранение сводной таблицы результатов
    summary_path = os.path.join(args.output_dir, f"summary_seed{args.seed}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("  ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'='*60}")
    summary_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Val Accuracy": r["validation_results"]["accuracy"],
            "Test Accuracy": r["testing_results"]["accuracy"]
        }
        for r in all_results
    ])
    print(summary_df.to_string(index=False))
    print(f"\nСводка сохранена: {summary_path}")


if __name__ == "__main__":
    main()
