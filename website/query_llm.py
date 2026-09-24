import json
import os
import re

import requests
from dotenv import load_dotenv


load_dotenv()


API_URL = "https://api.kie.ai/gpt-5-2/v1/chat/completions"
API_KEY = os.getenv("KIE_API_KEY")
TITLE_FORMAT_VERSION = "exact-v1"
ABSTRACT_MAX_CHARACTERS = 5000


DESCRIPTION_PROMPT = """Кратко изложи абстракт по-русски: 1–2 предложения, до 50 слов.
Сохрани только явно указанные цель, метод и главный результат. Не добавляй знания,
оценки, прогнозы или вводные фразы. Если содержательных данных нет, ответь None.
Выведи только описание."""


WEAK_SIGNAL_PROMPT = """Объясни по-русски, почему публикация может быть слабым
сигналом нового направления: 2–3 коротких предложения, до 70 слов. Опирайся только
на абстракт и переданные положительные факторы. Не упоминай модель, SHAP и имена
признаков; не выдумывай динамику, спрос или перспективы. Если обоснования нет,
ответь None. Выведи только объяснение."""


TITLE_TRANSLATION_PROMPT = """Переведи название научной статьи на русский язык
максимально точно. Ничего не сокращай, не перефразируй, не поясняй и не добавляй.
Сохрани смысл, структуру, пунктуацию, имена собственные, числа и степень уверенности;
общепринятые аббревиатуры оставь как есть. Русское название верни без изменений.
Выведи ровно одно название без кавычек, меток и комментариев."""


FEATURE_LABELS = {
    "source_tier": "тип научного источника",
    "novelty_raw": "отличие от более ранних работ темы",
    "cluster_density": "сформированность группы близких исследований",
    "is_outlier_cluster": "принадлежность к необычной группе исследований",
    "ppmi_domain_score": "необычность сочетания научных областей",
    "commercial_maturity_index": "участие коммерческих организаций",
    "citation_velocity": "накопленная цитируемость",
    "citation_acceleration": "изменение темпа цитирования",
    "distinct_venues_ratio": "разнообразие площадок",
    "author_growth_rate": "изменение числа авторов в теме",
    "has_ref_data": "наличие библиографических связей",
}


def _query_llm(
    system_prompt: str,
    abstract_text: str,
    max_tokens: int | None = None,
    temperature: float = 0.2,
) -> str:
    if not isinstance(abstract_text, str) or not abstract_text.strip():
        return "None"
    abstract_text = abstract_text.strip()
    if not API_KEY:
        raise RuntimeError("Переменная окружения KIE_API_KEY не задана")

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": abstract_text},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    try:
        result = data["choices"][0]["message"]["content"]
        if not isinstance(result, str):
            raise ValueError("LLM returned non-text content")
        result = result.strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("LLM returned an invalid response") from exc
    return result or "None"


def _limit_sentences(text: str, maximum: int) -> str:
    if text == "None":
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:maximum]).strip()


def _clean_title(text: str) -> str:
    """Remove response decoration without changing the translated title."""
    if text == "None":
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0] if lines else ""
    title = re.sub(
        r"^(?:перевод|название|переведённое название)\s*:\s*",
        "",
        title,
        flags=re.I,
    )
    return title.strip(" `*_\"'«»") or "None"


def _compact_abstract(text: str) -> str:
    """Keep a normalized abstract, including its conclusion, within a token budget."""
    compact = " ".join(text.split())
    if len(compact) <= ABSTRACT_MAX_CHARACTERS:
        return compact
    head_size = ABSTRACT_MAX_CHARACTERS * 3 // 4
    tail_size = ABSTRACT_MAX_CHARACTERS - head_size
    return f"{compact[:head_size].rstrip()} … {compact[-tail_size:].lstrip()}"


def _positive_factors(shap_values: dict, limit: int = 4) -> list[dict]:
    factors = []
    for name, payload in shap_values.items():
        if not isinstance(payload, dict):
            continue
        try:
            contribution = float(payload.get("shap", 0))
        except (TypeError, ValueError):
            continue
        if contribution <= 0:
            continue
        factors.append({
            "фактор": FEATURE_LABELS.get(name, name),
            "значение": payload.get("value"),
            "вклад": round(contribution, 4),
        })
    return sorted(factors, key=lambda item: item["вклад"], reverse=True)[:limit]


def summarize_abstract(abstract_text: str) -> str:
    """Вернуть краткое описание абстракта в одном или двух предложениях."""
    if not isinstance(abstract_text, str) or not abstract_text.strip():
        return "None"
    result = _query_llm(
        DESCRIPTION_PROMPT,
        _compact_abstract(abstract_text),
        max_tokens=120,
    )
    return _limit_sentences(result, 2)


def translate_article_title(title: str, abstract_text: str = "") -> str:
    """Точно перевести только исходное название статьи."""
    if not isinstance(title, str) or not title.strip():
        return "None"

    result = _query_llm(
        TITLE_TRANSLATION_PROMPT,
        title.strip(),
        max_tokens=128,
        temperature=0.0,
    )
    return _clean_title(result)


def explain_weak_signal(abstract_text: str, shap_values) -> str:
    """Объяснить слабый сигнал по абстракту, признакам и вкладам SHAP."""
    if not isinstance(abstract_text, str) or not abstract_text.strip():
        return "None"

    if isinstance(shap_values, str):
        try:
            shap_values = json.loads(shap_values)
        except json.JSONDecodeError:
            return "None"
    if not isinstance(shap_values, dict) or not shap_values:
        return "None"

    factors = _positive_factors(shap_values)
    if not factors:
        return "None"

    user_content = (
        f"Абстракт: {_compact_abstract(abstract_text)}\n"
        "Положительные факторы: "
        f"{json.dumps(factors, ensure_ascii=False, separators=(',', ':'))}"
    )
    result = _query_llm(WEAK_SIGNAL_PROMPT, user_content, max_tokens=160)
    return _limit_sentences(result, 3)
