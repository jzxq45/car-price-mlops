"""
train.py — Скрипт обучения модели машинного обучения.

В концепции MLOps этот этап отвечает за воспроизводимое обучение:
1. Подготовка и генерация тренировочных данных.
2. Обучение алгоритма регрессии.
3. Валидация качества модели на отложенной выборке (метрика MAE).
4. Сериализация (сохранение) артефакта модели в бинарный файл model.pkl.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Генерирует синтетический датасет с реалистичными зависимостями.
    
    В MLOps важно фиксировать random_state (seed), чтобы при каждом запуске
    получать идентичные данные и воспроизводимый результат.
    """
    np.random.seed(random_state)

    # 1. Год выпуска: от 2005 до 2023
    years = np.random.randint(2005, 2024, size=n_samples)

    # 2. Мощность двигателя: от 80 до 450 л.с.
    horsepowers = np.random.randint(80, 451, size=n_samples)

    # 3. Пробег: зависит от возраста машины + случайный шум
    # Чем старше автомобиль, тем в среднем выше его пробег
    current_year = 2024
    ages = current_year - years
    avg_annual_mileage = 15000  # средний пробег в год
    mileages = ages * avg_annual_mileage + np.random.normal(0, 10000, size=n_samples)
    # Пробег не может быть отрицательным:
    mileages = np.maximum(mileages, 1000).astype(int)

    # 4. Цена (таргет): моделируем логичную зависимость
    # Базовая стоимость + бонус за свежий год + бонус за л.с. - штраф за пробег + случайный разброс
    base_price = 500_000
    price = (
        base_price
        + (years - 2005) * 80_000
        + (horsepowers - 80) * 8_000
        - (mileages / 1000) * 3_000
        + np.random.normal(0, 50_000, size=n_samples)
    )
    # Ограничиваем минимальную цену автомобиля:
    price = np.maximum(price, 150_000).astype(int)

    return pd.DataFrame({
        "year": years,
        "mileage": mileages,
        "horsepower": horsepowers,
        "price": price
    })


def main():
    print("=" * 60)
    print("🚀 [MLOps Pipeline] Старт процесса обучения модели...")
    print("=" * 60)

    # 1. Создаем датасет
    df = generate_synthetic_data(n_samples=500, random_state=42)
    print(f"📊 Датасет успешно сгенерирован: {df.shape[0]} строк, {df.shape[1]} колонок.")
    print("Пример данных:")
    print(df.head(3))
    print("-" * 60)

    # 2. Разделяем признаки (X) и целевую переменную (y)
    feature_cols = ["year", "mileage", "horsepower"]
    X = df[feature_cols]
    y = df["price"]

    # 3. Делим на обучающую (80%) и валидационную (20%) выборки
    # В продакшене валидация обязательна, чтобы убедиться, что модель не переобучилась
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Инициализация и обучение алгоритма
    # RandomForestRegressor выбран благодаря устойчивости к нелинейным зависимостям и отсутствию нужды в масштабировании признаков
    print("🧠 Обучение RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Оценка качества модели (Метрика MAE)
    # MAE (Mean Absolute Error) показывает, на сколько рублей/долларов в среднем ошибается модель
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"📈 Метрика валидации MAE (Mean Absolute Error): {mae:,.2f} руб.")

    # 6. Сохранение артефакта модели
    # Относительный путь гарантирует работу как локально, так и внутри Docker
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Модель успешно сериализована и сохранена в: {os.path.abspath(model_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

