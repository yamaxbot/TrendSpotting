# TrendSpotting

LLM-вызовы сайта используют `qwen/qwen3.7-flash` через VseLLM и OpenAI SDK; общий клиент находится в `vsellm_chat.py`.

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

Офлайн-сбор и обучение **не выполняются** при поиске на сайте. Docker-образ содержит код сайта, парсера и расчёта признаков, а также рабочую CatBoost-модель с порогом; обучающие данные, эксперименты, локальные результаты поиска и ключи в него не копируются.

## Запуск через Docker Compose

1. Скопируйте [`.env.example`](.env.example) в `.env`, задайте `OPENALEX_API_KEY`, `VSELLM_API_KEY`, длинный случайный `DJANGO_SECRET_KEY` и замените `YOUR_PUBLIC_IP` в `DJANGO_ALLOWED_HOSTS` на публичный IP сервера. В `DJANGO_ALLOWED_HOSTS` указывается IP или домен, не API-ключ.
2. Выполните из корня проекта `docker compose up --build -d`.
3. Откройте `http://<публичный_IP>:8000/` (на самом сервере также работает `http://127.0.0.1:8000/`). Логи: `docker compose logs -f web`; остановка: `docker compose down`.

Пример `.env.example` публикует порт на всех интерфейсах (`HOST_BIND=0.0.0.0`); откройте порт `PORT` в файрволе сервера и у провайдера. Без внешнего прокси соединение будет по HTTP. Если нужен только локальный доступ, установите `HOST_BIND=127.0.0.1`. После изменения `.env` примените настройки командой `docker compose up -d --force-recreate --no-deps web`. `docker compose down` сохраняет тома с результатами и кэшем модели. Для повторного запуска без старых данных выполните `docker compose down -v` перед сборкой: это удалит сохранённые результаты и кэш без возможности восстановления из томов.

## Запуск без Docker и проверка

Нужен Python 3.11+. Из корня проекта: `pip install -r requirements.txt`, заполнить `.env`, затем `python website/manage.py runserver`. Проверка Django и сайта: `python website/manage.py check` и `python website/manage.py test trends --noinput`. Тесты парсера: `python -m unittest on_demand_parsing.test_optimization on_demand_parsing.test_time_budget`.
