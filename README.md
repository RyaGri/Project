# Построение рекомендательной системы постов для социальной сети.
## В рамках проекта были выполнены следующие задачи
- Выгрузка данных из базы данных PostgreSQL и проведение первичного анализа;

- Подготовка данных для обучения и создание новых признаков;

- Сделаны эмбеддинги из текстов постов с помощью предобученных нейронных моделей;

- Обучение различных моделей машинного обучения и оценка их качества на валидационной выборке;

- Написание сервиса с использованием FastAPI: загрузка модели и признаков для каждого уникального пользователя и  постов пользователю. Добавлено разделение пользователей на две группы для проведения  A/B-теста

- Проведение A/B-теста .