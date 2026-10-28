# 🚗 Car Price Prediction — Твой первый MLOps проект

Добро пожаловать в мир **MLOps** (Machine Learning Operations)! 

Этот проект демонстрирует полный базовый жизненный цикл ML-сервиса в продакшене:
1. **Генерация данных и обучение** модели регрессии (`train.py`).
2. **Сериализация артефакта** в файл `model.pkl`.
3. **Создание REST API** с валидацией входных признаков на базе FastAPI (`app.py`).
4. **Контейнеризация** сервиса с автоматическим обучением при сборке (`Dockerfile`).
5. **Оркестрация и запуск** через `docker-compose.yml`.

---

## 📁 Структура проекта

* `requirements.txt` — список зависимостей с фиксированными версиями.
* `train.py` — воспроизводимый скрипт генерации синтетического датасета, обучения `RandomForestRegressor` и сохранения `model.pkl`.
* `app.py` — сервис FastAPI с эндпоинтами `/predict` и `/health`, логированием и валидацией данных через Pydantic.
* `Dockerfile` — инструкция создания легковесного контейнера на базе `python:3.10-slim`.
* `docker-compose.yml` — конфигурация запуска контейнера с пробросом портов и политикой `restart: always`.

---

## 🚀 Пошаговый запуск на Windows

Убедись, что на твоем компьютере запущен **Docker Desktop**. Открой терминал **PowerShell** в папке с проектом (`d:\projects\yago`).

### Вариант 1. Запуск через Docker Compose (Рекомендуемый способ)

1. **Сборка образа и запуск контейнера:**
   ```powershell
   docker compose up --build -d
   ```
   *Флаг `-d` (detached) запускает контейнер в фоновом режиме, освобождая терминал.*

2. **Просмотр логов контейнера:**
   ```powershell
   docker compose logs -f
   ```
   *Ты увидишь логи запуска Uvicorn и каждое обращение к API.*

3. **Остановка сервиса:**
   ```powershell
   docker compose down
   ```

---

### Вариант 2. Запуск через чистый Docker CLI

Если хочешь выполнить сборку вручную:

1. **Собрать Docker-образ:**
   ```powershell
   docker build -t car-price-predictor:v1 .
   ```
   *Во время сборки выполнится команда `RUN python train.py`, которая обучит модель и выведет MAE прямо в консоль.*

2. **Запустить контейнер:**
   ```powershell
   docker run -d --name car_price_service -p 8000:8000 --restart always car-price-predictor:v1
   ```

---

## 🧪 Проверка работы API

### 1. Интерактивная документация Swagger UI (В браузере)
Открой в браузере страницу:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

Здесь ты сможешь нажать кнопку **«Try it out»**, ввести любые параметры машины и протестировать работу сервиса с красивым интерфейсом.

---

### 2. Проверка Health Check (Работоспособность сервиса)

**В PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

**Через curl:**
```powershell
curl.exe -X GET "http://localhost:8000/health"
```

Ожидаемый ответ:
```json
{
  "status": "ok"
}
```

---

### 3. Отправка запроса на предсказание цены (`/predict`)

**В PowerShell (Invoke-RestMethod):**
```powershell
$body = @{
    year = 2019
    mileage = 45000
    horsepower = 180
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

**Через curl (в терминале Windows):**
```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{\"year\": 2019, \"mileage\": 45000, \"horsepower\": 180}'
```

Ожидаемый ответ:
```json
{
  "predicted_price": 2345000.0,
  "currency": "RUB",
  "features_received": {
    "year": 2019,
    "mileage": 45000,
    "horsepower": 180
  }
}
```

---

## 💡 Почему в продакшене делают именно так?

* **`lifespan` в FastAPI**: Модель загружается в память ровно один раз при старте сервера. Если бы мы загружали `model.pkl` на каждый запрос, время ответа (Latency) выросло бы в десятки раз.
* **Pydantic**: Автоматически защищает модель от некорректных данных (например, если передадут строку вместо числа или отрицательный пробег). Модель не упадет с ошибкой — клиент сразу получит читаемый статус `422 Unprocessable Entity`.
* **Health-check (`/health`)**: Любой современный оркестратор (Kubernetes, AWS ECS, Docker Swarm) использует такие эндпоинты, чтобы автоматически перезапускать упавшие контейнеры и не пускать клиентов на неготовые поды.

