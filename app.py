"""
app.py — Сервер предсказаний на базе FastAPI.

В MLOps этот сервис выступает слоем Model Serving (инференс модели):
1. Валидация входных данных через строгие схемы Pydantic (защита от мусорных данных).
2. Загрузка артефакта модели ОДИН РАЗ при старте приложения (Singleton pattern).
   Загрузка модели на каждый входящий HTTP-запрос создала бы огромный оверхед на чтение с диска.
3. Логирование входящих запросов для мониторинга и отладки сервиса.
4. Health-check эндпоинт для оркестраторов (Docker Compose / Kubernetes).
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# ==============================================================================
# 1. Настройка системы логирования (Logging)
# ==============================================================================
# В MLOps логирование критически важно для отслеживания работоспособности,
# времени ответа (latency) и входящих данных (data drift).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("car_price_api")

# Глобальная переменная для хранения загруженной модели в оперативной памяти (RAM)
ml_model = None
MODEL_PATH = "model.pkl"


# ==============================================================================
# 2. Жизненный цикл сервиса (Lifespan Context Manager)
# ==============================================================================
# В современном FastAPI для инициализации ресурсов (БД, ML-моделей) при старте 
# и очистки при остановке используется lifespan context manager.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    logger.info("Запуск приложения. Загрузка ML-модели в память...")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Файл модели {MODEL_PATH} не найден! Запустите train.py перед стартом сервера.")
        raise FileNotFoundError(f"Файл {MODEL_PATH} отсутствует.")
    
    # Модель загружается в RAM один раз на весь жизненный цикл процесса
    ml_model = joblib.load(MODEL_PATH)
    logger.info("ML-модель успешно загружена в оперативную память и готова к инференсу.")
    
    yield
    
    logger.info("Остановка приложения. Освобождение ресурсов...")
    ml_model = None


# Создаем экземпляр FastAPI с метаданными для интерактивной документации Swagger
app = FastAPI(
    title="Car Price Prediction API",
    description="Микросервис машинного обучения для оценки стоимости автомобилей.",
    version="1.0.0",
    lifespan=lifespan
)


# ==============================================================================
# 3. Middleware для логирования входящих HTTP-запросов
# ==============================================================================
# Перехватывает каждый входящий запрос, замеряет время обработки и логирует результат.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    
    logger.info(
        f"HTTP {request.method} {request.url.path} | "
        f"Статус: {response.status_code} | "
        f"Время обработки: {duration_ms:.2f} ms"
    )
    return response


# ==============================================================================
# 4. Схемы валидации данных (Pydantic v2)
# ==============================================================================
# Pydantic автоматически валидирует типы, диапазоны и возвращает понятные ошибки 422
# клиенту, если он прислал некорректные данные (например, отрицательный пробег).
class CarFeatures(BaseModel):
    year: int = Field(
        ...,
        ge=1990,
        le=2026,
        description="Год выпуска автомобиля (от 1990 до текущего года)",
        example=2018
    )
    mileage: int = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Пробег автомобиля в километрах (неотрицательное число)",
        example=65000
    )
    horsepower: int = Field(
        ...,
        ge=30,
        le=1500,
        description="Мощность двигателя в лошадиных силах (л.с.)",
        example=150
    )


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Предсказанная стоимость автомобиля")
    currency: str = Field(default="RUB", description="Валюта оценки")
    features_received: CarFeatures = Field(..., description="Полученные характеристики")


# ==============================================================================
# 5. Эндпоинты API
# ==============================================================================

@app.get("/health", summary="Проверка состояния сервиса (Health Check)")
def health_check() -> Dict[str, str]:
    """
    Эндпоинт для систем оркестрации (Docker Compose, Kubernetes, Consul).
    Позволяет мониторингу убедиться, что контейнер жив и принимает трафик.
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Получение предсказания цены автомобиля"
)
def predict_price(features: CarFeatures) -> PredictionResponse:
    """
    Принимает характеристики автомобиля, передает их в модель и возвращает прогноз.
    """
    if ml_model is None:
        logger.error("Попытка предсказания, когда модель не инициализирована!")
        raise HTTPException(status_code=500, detail="Модель недоступна")
    
    # Модели Scikit-learn ожидают 2D структуру данных (DataFrame или 2D numpy array)
    # Имена колонок должны строго совпадать с теми, на которых обучалась модель в train.py
    input_data = pd.DataFrame([{
        "year": features.year,
        "mileage": features.mileage,
        "horsepower": features.horsepower
    }])
    
    # Выполняем инференс
    raw_prediction = ml_model.predict(input_data)[0]
    predicted_price = round(float(raw_prediction), 2)
    
    logger.info(
        f"Сделан прогноз для машины: год={features.year}, "
        f"пробег={features.mileage} км, л.с.={features.horsepower} -> {predicted_price:,.2f} руб."
    )
    
    return PredictionResponse(
        predicted_price=predicted_price,
        currency="RUB",
        features_received=features
    )

