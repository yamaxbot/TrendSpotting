import json
import logging
import os
import re

import requests
from dotenv import load_dotenv


load_dotenv()

API_URL = "https://api.kie.ai/gpt-5-2/v1/chat/completions"
API_KEY = os.getenv("KIE_API_KEY")
TITLE_FORMAT_VERSION = "exact-v2"
ARTICLE_ANALYSIS_VERSION = "grounded-v2"
ABSTRACT_MAX_CHARACTERS = 5000
MAX_LLM_ATTEMPTS = 2
MAX_ARTICLE_ANALYSIS_ATTEMPTS = 3
RELATED_ABSTRACT_MAX_CHARACTERS = 1600
logger = logging.getLogger(__name__)
SCIENTIFIC_UNITS = re.compile(
    r"(?<![А-Яа-яЁё])(?:[кмнгдпс]?(?:Гц|Вт|В|А|Па|Дж|Тл|Ом|См|С)|[кМ]?эВ)(?![А-Яа-яЁё])"
)

RELATED_SIGNAL_PROMPT = """По аннотациям нескольких строго связанных научных работ напиши по-русски, почему их общая тема может быть слабым технологическим сигналом. Найди конкретный новый метод, применение или сдвиг относительно прежнего подхода, подтверждённый минимум двумя связанными работами. Не называй само совпадение тем или количество работ открытием. Не утверждай, что технология впервые появилась, редка, растёт, внедрена или поддержана компаниями, если аннотации этого не доказывают. Не добавляй прогнозы и вымышленные результаты. Если конкретного общего наблюдения нет, верни {"signal":"None","supporting_ids":[]}.
Иначе верни только JSON {"signal":"2–3 содержательных предложения; четвёртое лишь если добавляет факт, не более 90 слов","supporting_ids":["id связанной работы 1","id связанной работы 2"]}. В тексте сначала назови конкретное наблюдение, затем его возможное значение; избегай общих фраз и слов «модель», «SHAP», «признаки». Указывай только id работ, действительно подтверждающих главный тезис."""

DESCRIPTION_PROMPT = """Кратко изложи абстракт по-русски: 1–2 предложения, до 50 слов.
Сохрани только явно указанные цель, метод и главный результат. Не добавляй знания,
оценки, прогнозы или вводные фразы. Если содержательных данных нет, ответь None.
Выведи только описание."""

ARTICLE_ANALYSIS_PROMPT = """По аннотации научной статьи ответь по-русски одним JSON-объектом:
{"summary":"...","problem":"...","advantage":"...","case_result":"...","weak_signal":"..."}.
summary: суть исследования, до 40 слов. problem: какую конкретную задачу решает,
до 25 слов. advantage: предложенный подход или явно заявленное преимущество, до 25 слов.
case_result: конкретный результат этой работы, до 30 слов. weak_signal: 1–3 предложения,
до 75 слов. Сначала назови конкретный метод, применение или результат именно этой работы,
затем объясни, почему это может быть ранним сигналом; положительные факторы оценки
используй лишь как дополнительное подтверждение. Не заменяй содержание статьи списком
факторов. Не утверждай, что направление впервые появилось, растёт или внедрено, если
данные этого не показывают. Используй только факты из аннотации и переданных факторов,
без прогнозов и домыслов. Если поле не подтверждено, запиши строку "None".
Никакого текста вне JSON."""

WEAK_SIGNAL_PROMPT = """Объясни по-русски, почему публикация может быть слабым
сигналом нового направления: 2–3 коротких предложения, до 70 слов. Опирайся только
на абстракт и переданные положительные факторы. Не упоминай модель, SHAP или имена
признаков; не выдумывай динамику, спрос или перспективы. Если обоснования нет,
ответь None. Выведи только объяснение."""

TITLE_TRANSLATION_PROMPT = """Точно переведи название научной статьи на русский.
Ничего не сокращай, не перефразируй, не поясняй и не добавляй. Сохрани смысл,
структуру, пунктуацию, имена собственные, числа и степень уверенности;
общепринятые аббревиатуры оставь как есть. Выведи ровно одно полное название без
кавычек, меток и комментариев."""

FEATURE_LABELS = {
    "source_tier": "тип научного источника",
    "topic_publication_growth": "рост числа публикаций по теме",
    "topic_growth_acceleration": "ускорение роста темы",
    "topic_age_years": "возраст исследовательской темы",
    "novelty_raw": "отличие от более ранних работ темы",
    "cluster_density": "сформированность группы близких исследований",
    "nearest_topic_similarity": "сходство с ближайшим тематическим направлением",
    "cross_topic_gap": "отличие от соседних тематических направлений",
    "is_outlier_cluster": "принадлежность к необычной группе исследований",
    "ppmi_domain_score": "необычность сочетания научных областей",
    "commercial_maturity_index": "участие коммерческих организаций",
    "domain_history_count": "объём исторических работ научной области",
    "source_history_count": "историческая активность источника",
    "source_novelty": "новизна источника для темы",
    "topic_source_diversity": "разнообразие источников по теме",
    "reference_count": "число библиографических ссылок",
    "citation_velocity": "накопленная цитируемость",
    "citation_acceleration": "изменение темпа цитирования",
    "author_growth_rate": "изменение числа авторов в теме",
    "author_growth_acceleration": "ускорение роста авторского сообщества",
    "topic_historical_share": "историческая доля публикаций темы",
    "topic_local_share": "текущая доля публикаций темы",
    "historical_author_share": "историческая доля авторов темы",
    "has_ref_data": "наличие библиографических связей",
}


def _query_llm(system_prompt, abstract_text, max_tokens=None, temperature=0.2):
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


def _is_none(value):
    return not isinstance(value, str) or value.strip().casefold() == "none"


def has_corrupt_text(value):
    """Detect common truncation, encoding and token-splicing artifacts."""
    if not isinstance(value, str) or not value.strip():
        return True
    text = SCIENTIFIC_UNITS.sub("", value.strip())
    if any(
        re.search(pattern, text)
        for pattern in (
            r"\ufffd",
            r"[.!?][а-яё]",
            r"[А-Яа-яЁё][A-Za-z]|[A-Za-z][А-Яа-яЁё]",
        )
    ):
        return True
    return any(
        re.search(r"[а-яё][А-ЯЁ]", word.group())
        and not re.match(r"[А-ЯЁ]", word.group())
        for word in re.finditer(r"[А-ЯЁа-яё]+", text)
    )


def is_valid_title(value, source_title):
    if not isinstance(source_title, str) or not source_title.strip():
        return _is_none(value)
    if _is_none(value) or has_corrupt_text(value):
        return False
    if re.search(r"[A-Za-z]", source_title) and not re.search(r"[А-Яа-яЁё]", value):
        return False
    return not _starts_with_lowercase_cyrillic(value)


def _starts_with_lowercase_cyrillic(value):
    first_letter = re.search(r"[A-Za-zА-Яа-яЁё]", value)
    return bool(first_letter and re.match(r"[а-яё]", first_letter.group()))


def is_valid_description(value, abstract_text):
    if not isinstance(abstract_text, str) or not abstract_text.strip():
        return _is_none(value)
    return (
        not _is_none(value)
        and not has_corrupt_text(value)
        and not _starts_with_lowercase_cyrillic(value)
    )


def is_valid_weak_signal(value):
    return not _is_none(value) and (
        not has_corrupt_text(value)
        and not _starts_with_lowercase_cyrillic(value)
    )


def is_generic_weak_signal(value):
    """Identify cached factor-only explanations that should be enriched."""
    return isinstance(value, str) and value.strip().startswith((
        "На оценку статьи положительно повлияли",
        "Ключевые факторы для этой статьи не выделены.",
    ))


def fallback_weak_signal(shap_values, *, summary=None, title=None):
    """Keep a result article-specific even when no grounded signal was generated."""
    if isinstance(shap_values, str):
        try:
            shap_values = json.loads(shap_values)
        except json.JSONDecodeError:
            shap_values = None
    factors = _positive_factors(shap_values) if isinstance(shap_values, dict) else []
    context = _limit_sentences(summary, 2) if is_valid_description(summary, summary) else ""
    if context and context[-1] not in ".!?":
        context += "."
    if not context and isinstance(title, str) and title.strip() and title != "None":
        context = f"Работа «{title.strip()}» рассматривает это направление."
    if not factors:
        return context or "Для этой статьи не удалось выделить обоснование слабого сигнала."
    labels = ", ".join(factor["фактор"] for factor in factors[:2])
    if context:
        return f"{context} В оценке также значимы {labels}."
    return f"Для этой публикации в оценке значимы {labels}."


def clean_weak_signal(value):
    """Remove the obsolete disclaimer from already cached explanations."""
    if not isinstance(value, str):
        return value
    return value.replace(
        " Это признаки в данных, а не доказательство появления нового тренда.",
        "",
    ).strip()


def _limit_sentences(text, maximum):
    if _is_none(text):
        return "None"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:maximum]).strip()


def _clean_title(text):
    """Remove response decoration without changing the translated title."""
    if _is_none(text):
        return "None"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0] if lines else ""
    title = re.sub(
        r"^(?:перевод|название|переведённое название)\s*:\s*",
        "",
        title,
        flags=re.I,
    )
    return title.strip(" `*_\"'«»") or "None"


def _compact_abstract(text):
    compact = " ".join(text.split())
    if len(compact) <= ABSTRACT_MAX_CHARACTERS:
        return compact
    head_size = ABSTRACT_MAX_CHARACTERS * 3 // 4
    tail_size = ABSTRACT_MAX_CHARACTERS - head_size
    return f"{compact[:head_size].rstrip()} … {compact[-tail_size:].lstrip()}"


def _positive_factors(shap_values, limit=4):
    factors = []
    for name, payload in shap_values.items():
        if name == "split" or not isinstance(payload, dict):
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


def summarize_abstract(abstract_text):
    if not isinstance(abstract_text, str) or not abstract_text.strip():
        return "None"
    for _ in range(MAX_LLM_ATTEMPTS):
        result = _limit_sentences(
            _query_llm(DESCRIPTION_PROMPT, _compact_abstract(abstract_text), max_tokens=120),
            2,
        )
        if is_valid_description(result, abstract_text):
            return result
    return "None"


def _parse_article_analysis(content):
    """Validate a compact, source-grounded response before caching it."""
    if not isinstance(content, str):
        raise ValueError("Article analysis is not text")
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    values = json.loads(content)
    if not isinstance(values, dict):
        raise ValueError("Article analysis is not an object")
    result = {}
    for key in ("summary", "problem", "advantage", "case_result", "weak_signal"):
        value = values.get(key)
        if not isinstance(value, str) or len(value) > 450:
            raise ValueError(f"Invalid article analysis field: {key}")
        value = value.strip()
        if not value or (value.casefold() != "none" and has_corrupt_text(value)):
            raise ValueError(f"Invalid article analysis field: {key}")
        if key == "weak_signal" and value.casefold() != "none" and not is_valid_weak_signal(value):
            raise ValueError("Invalid article weak signal")
        result[key] = value
    return result


def analyze_abstract(abstract_text, shap_values=None):
    """Create article fields and its own weak-signal explanation in one LLM call."""
    empty = dict.fromkeys(("summary", "problem", "advantage", "case_result", "weak_signal"), "None")
    if not isinstance(abstract_text, str) or not abstract_text.strip():
        return empty
    if isinstance(shap_values, str):
        try:
            shap_values = json.loads(shap_values)
        except json.JSONDecodeError:
            shap_values = None
    factors = _positive_factors(shap_values, limit=3) if isinstance(shap_values, dict) else []
    content = json.dumps({
        "abstract": _compact_abstract(abstract_text),
        "positive_factors": [factor["фактор"] for factor in factors],
    }, ensure_ascii=False, separators=(",", ":"))
    for attempt in range(1, MAX_ARTICLE_ANALYSIS_ATTEMPTS + 1):
        try:
            analysis = _parse_article_analysis(_query_llm(
                ARTICLE_ANALYSIS_PROMPT,
                content,
                max_tokens=500,
                temperature=0.0,
            ))
            if not is_valid_description(analysis["summary"], abstract_text):
                raise ValueError("Article analysis has no valid summary")
            return analysis
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            logger.warning(
                "Article analysis attempt %s/%s failed: %s: %s",
                attempt, MAX_ARTICLE_ANALYSIS_ATTEMPTS, type(exc).__name__, exc,
            )
    return empty


def summarize_related_signal(anchor, related_articles):
    """Return a source-grounded signal, or None when abstracts show no shared finding."""
    if (
        not isinstance(anchor.get("abstract_text"), str)
        or not anchor["abstract_text"].strip()
        or len(related_articles) < 2
    ):
        return None

    articles = [{
        "id": anchor["doc_id"],
        "year": anchor.get("pub_year"),
        "title": anchor["title"],
        "abstract": " ".join(anchor["abstract_text"].split())[:RELATED_ABSTRACT_MAX_CHARACTERS],
    }]
    for item in related_articles[:3]:
        articles.append({
            "id": item["doc_id"],
            "year": item["year"],
            "title": item["title"],
            "abstract": " ".join(item["abstract_text"].split())[:RELATED_ABSTRACT_MAX_CHARACTERS],
        })

    response = _query_llm(
        RELATED_SIGNAL_PROMPT,
        json.dumps(articles, ensure_ascii=False, separators=(",", ":")),
        max_tokens=300,
        temperature=0.0,
    )
    try:
        content = response.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
        payload = json.loads(content)
        signal = payload["signal"]
        supporting_ids = payload["supporting_ids"]
    except (ValueError, TypeError, KeyError):
        return None
    valid_ids = {item["id"] for item in articles[1:]}
    if not isinstance(supporting_ids, list) or not all(
        isinstance(item, str) for item in supporting_ids
    ):
        return None
    if (
        not isinstance(signal, str)
        or len(set(supporting_ids)) < 2
        or not set(supporting_ids).issubset(valid_ids)
        or not is_valid_weak_signal(signal)
        or len(signal) > 650
    ):
        return None
    return signal.strip()


def translate_article_title(title, abstract_text=""):
    if not isinstance(title, str) or not title.strip():
        return "None"
    source_title = title.strip()
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            result = _clean_title(
                _query_llm(
                    TITLE_TRANSLATION_PROMPT,
                    source_title,
                    max_tokens=160,
                    temperature=0.0,
                )
            )
        except (RuntimeError, ValueError, requests.RequestException) as exc:
            logger.warning(
                "Title translation attempt %s/%s failed: %s",
                attempt, MAX_LLM_ATTEMPTS, type(exc).__name__,
            )
            continue
        if is_valid_title(result, source_title):
            return result
    return source_title


def explain_weak_signal(abstract_text, shap_values):
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
    for _ in range(MAX_LLM_ATTEMPTS):
        result = _limit_sentences(
            _query_llm(WEAK_SIGNAL_PROMPT, user_content, max_tokens=160),
            3,
        )
        if is_valid_weak_signal(result):
            return result
    return "None"
