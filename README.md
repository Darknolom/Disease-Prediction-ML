# Disease Prediction ML

## Описание проекта

Проект машинного обучения для предсказания заболеваний на основе бинарных симптомов. Разработан в рамках курсовой работы. Включает пайплайн обучения 6 моделей классификации с подбором гиперпараметров через GridSearchCV:

- Logistic Regression
- Decision Tree
- KNN
- Random Forest
- Gradient Boosting
- XGBoost

Данные: `Training.csv` и `Testing.csv`.

## Требования к окружению

- ОС: Linux (или WSL / Google Colab)
- Python: >= 3.10
- Пакетный менеджер: `uv`

Установка `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
