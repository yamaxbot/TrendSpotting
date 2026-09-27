# TrendSpotting

Сайт ищет в OpenAlex научные статьи по заданному направлению и показывает до 15 разных кандидатов в слабые технологические сигналы. CatBoost оценивает **отдельную статью**, а не доказывает существование тренда; статистика публикаций и ссылки на близкие работы собираются отдельно.

## Как работает поиск

1. `on_demand_parsing/` нормализует запрос, расширяет поиск по темам OpenAlex и сохраняет очищенный корпус.
2. `preprocessing/create_features.py` рассчитывает признаки статей и их научного окружения.
3. `website/main.py` применяет сохранённую модель из `outputs/catboost_splits/random/`, отбирает релевантные и непохожие статьи; `website/query_llm.py` готовит тексты карточек.
4. `website/trends/` отдаёт страницы, отдельно подбирает источники для карточки и запрашивает годовую статистику OpenAlex. Результаты повторного запроса читаются из `website/data/`.

Поиск выполняется в фоновой очереди одного процесса. Незавершённый запрос при перезапуске сервера нужно повторить. Данные за текущий год могут быть неполными.

## Структура

| Папка | Что в ней находится |
|---|---|
| [`website/`](website/website.md) | Django, интерфейс, ранжирование, пояснения и кэш результатов |
| [`on_demand_parsing/`](on_demand_parsing/PARSING.MD) | Поиск и очистка работ OpenAlex для запроса пользователя |
| [`preprocessing/`](preprocessing/preprocessing.md) | Расчёт признаков и меток для модели |
| [`outputs/`](outputs/README.md) | Сохранённые модели и отчёты экспериментов; рабочая модель сайта — `catboost_splits/random/` |
| [`download_dataset/`](download_dataset/RUN_NEW_DATASET.md) | Отдельные офлайн-сборщики обучающих корпусов и их конфигурации |
| [`education/`](education/EDUCATION.md) | Обучение и сравнение моделей на подготовленных данных |
| [`education2/`](education2/README.md) | Локальные дополнительные эксперименты и датасет; кроме README папка исключена из Git и Docker-образа |
| `logics/` | Генерируемые отладочные JSON парсера, не нужны для сборки сайта |

Офлайн-сбор и обучение **не выполняются** при поиске на сайте. Исходные большие Parquet-файлы и ключи не входят в Docker-образ.

## Запуск через Docker Compose

1. Скопируйте [`.env.example`](.env.example) в `.env` и задайте `OPENALEX_API_KEY` и `KIE_API_KEY`.
2. Выполните из корня проекта `docker compose up --build -d`.
3. Откройте `http://127.0.0.1:8000/`. Логи: `docker compose logs -f web`; остановка: `docker compose down`.

По умолчанию порт доступен только на самом сервере (`HOST_BIND=127.0.0.1`). Для доступа по IP задайте в `.env` `HOST_BIND=0.0.0.0`, `DJANGO_DEBUG=0`, длинный `DJANGO_SECRET_KEY`, `DJANGO_ALLOWED_HOSTS=<IP>` и откройте порт `PORT` в файрволе. Без внешнего прокси соединение будет по HTTP. `docker compose down` сохраняет тома с результатами и кэшем модели; `down -v` удаляет их.

## Запуск без Docker и проверка

Нужен Python 3.11+. Из корня проекта: `pip install -r requirements.txt`, заполнить `.env`, затем `python website/manage.py runserver`. Проверка Django и сайта: `python website/manage.py check` и `python website/manage.py test trends --noinput`. Тесты парсера: `python -m unittest on_demand_parsing.test_optimization on_demand_parsing.test_time_budget`.
