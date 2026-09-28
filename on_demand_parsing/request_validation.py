import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, DefaultHttpxClient, OpenAI, OpenAIError


API_BASE_URL = "https://api.vsellm.ru/v1"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_api_key():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("VSELLM_API_KEY", "").strip()
    if not api_key:
        raise ValueError(f"VSELLM_API_KEY не найден. Добавьте его в {ENV_PATH}")
    return api_key


def create_client(api_key, use_system_proxy=False):
    return OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
        http_client=DefaultHttpxClient(trust_env=use_system_proxy),
        max_retries=0,
    )


def get_model_ids(client):
    models = client.with_options(timeout=30).models.list()
    return {model.id for model in models.data}


def check_completion(client, model):
    response = client.with_options(timeout=120).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Ответь одним словом: OK"}],
        max_tokens=256,
    )
    if not response.choices:
        raise ValueError("VseLLM вернул ответ без choices")
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("VseLLM вернул пустой ответ модели")
    print(f"Тестовый запрос к {model}: успешно")
    print(f"Ответ модели: {content}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Проверка подключения к VseLLM")
    parser.add_argument(
        "--model",
        help="Отправить платный тестовый запрос к указанной модели",
    )
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Использовать системный прокси вместо прямого соединения",
    )
    args = parser.parse_args(argv)

    try:
        api_key = load_api_key()
        with create_client(api_key, args.use_system_proxy) as client:
            model_ids = get_model_ids(client)
            print(f"VseLLM доступен. Моделей в API: {len(model_ids)}")
            if args.model:
                if args.model not in model_ids:
                    raise ValueError(f"Модель {args.model!r} отсутствует в /v1/models")
                check_completion(client, args.model)
        print("Баланс проверяйте в личном кабинете VseLLM: публичный API для него не документирован.")
        return 0
    except APIStatusError as exc:
        print(f"Ошибка API VseLLM: HTTP {exc.status_code}", file=sys.stderr)
    except (APIConnectionError, OpenAIError, ValueError) as exc:
        print(f"Проверка VseLLM не удалась: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
