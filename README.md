# Disease Prediction ML

## 1. Описание проекта

Проект машинного обучения для предсказания заболеваний (41 класс) на основе бинарных симптомов (132 признака). Разработан в рамках курсовой работы. Включает пайплайн обучения 6 моделей классификации с подбором гиперпараметров через GridSearchCV:

- Logistic Regression
- Decision Tree
- KNN
- Random Forest
- Gradient Boosting
- XGBoost

Данные: `Training.csv` (4920 наблюдений) и независимый `Testing.csv`.

## 2. Требования к окружению

- ОС: Linux (или WSL / Google Colab)
- Python: >= 3.10
- Пакетный менеджер: `uv`

Установка `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
