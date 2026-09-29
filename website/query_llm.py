import json
import logging
import re
import time

from openai import OpenAIError

from vsellm_chat import chat_completion


TITLE_FORMAT_VERSION = "qwen37-exact-v2"
ARTICLE_ANALYSIS_VERSION = "qwen37-grounded-v1"
ABSTRACT_MAX_CHARACTERS = 5000
MAX_LLM_ATTEMPTS = 2
MAX_TITLE_TRANSLATION_ATTEMPTS = 3
MAX_ARTICLE_ANALYSIS_ATTEMPTS = 3
BATCH_ARTICLE_COUNT = 3
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

BATCH_ARTICLE_PROMPT = """Обработай каждую статью отдельно. Верни только JSON:
{"articles":[{"id":"исходный id","title":"точный перевод названия на русский","summary":"...","problem":"...","advantage":"...","case_result":"...","weak_signal":"..."}]}.
Для каждого входного id верни ровно один объект. title переведи полностью, без сокращений, пояснений и добавлений; сохрани имена, числа и сокращения. Остальные поля основывай только на аннотации данной статьи, не смешивай сведения разных статей. summary — суть до 40 слов; problem — задача до 25; advantage — метод или заявленное преимущество до 25; case_result — конкретный результат до 30. weak_signal — конкретный метод или результат и осторожное объяснение его возможной значимости, до 75 слов. Не выдумывай результаты, рост направления или внедрение. Если факт для поля не указан, напиши строку "None". Не добавляй текст вне JSON."""

CASE_RESULT_PROMPT = """Из аннотации извлеки один конкретный результат исследования на русском, до 30 слов. Сохрани числа и единицы измерения без изменений. Не добавляй фактов. Если результата нет, верни "None". Ответ только JSON: {"case_result":"..."}."""
RESULT_EVIDENCE = re.compile(
    r"\b(?:demonstrat\w*|show\w*|observ\w*|achiev\w*|measur\w*|"
    r"result\w*|reduc\w*|improv\w*|показ\w*|получ\w*|измер\w*)\b",
    re.I,
)

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


def _query_llm(system_prompt, abstract_text, max_tokens=None, temperature=0.2, timeout=30):
    if not isinstance(abstract_text, str) or not abstract_text.strip():
        return "None"
    abstract_text = abstract_text.strip()
    try:
        result = chat_completion(
            system_prompt, abstract_text, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout,
        )
        if not isinstance(result, str):
            raise ValueError("LLM returned non-text content")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("LLM returned an invalid response") from exc
    return result or "None"


def _is_none(value):
    return not isinstance(value, str) or value.strip().casefold() == "none"


def has_corrupt_text(value):
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
    latin_words = re.findall(r"[A-Za-z]{2,}", value)
    cyrillic_words = re.findall(r"[А-Яа-яЁё]{2,}", value)
    if len(latin_words) >= 4 and len(latin_words) > len(cyrillic_words):
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
    return isinstance(value, str) and value.strip().startswith((
        "На оценку статьи положительно повлияли",
        "Ключевые факторы для этой статьи не выделены.",
    ))


def fallback_weak_signal(shap_values, *, summary=None, title=None):
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
        if key not in values:
            raise ValueError(f"Missing article analysis field: {key}")
        value = values.get(key)
        valid = isinstance(value, str) and len(value) <= 450
        if valid:
            value = value.strip()
            valid = bool(value) and (value.casefold() == "none" or not has_corrupt_text(value))
            if key == "weak_signal" and valid and value.casefold() != "none":
                valid = is_valid_weak_signal(value)
        if not valid:
            if key == "summary":
                raise ValueError("Invalid article analysis field: summary")
            value = "None"
        result[key] = value
    return result


def analyze_articles_batch(articles, *, timeout=60):
    if not 2 <= len(articles) <= BATCH_ARTICLE_COUNT:
        raise ValueError("Invalid article batch size")

    source_by_id = {}
    payload = []
    for article in articles:
        doc_id = article.get("id")
        title = article.get("title")
        abstract = article.get("abstract")
        if (
            not isinstance(doc_id, str) or not doc_id
            or doc_id in source_by_id
            or not isinstance(title, str) or not title.strip()
            or not isinstance(abstract, str) or not abstract.strip()
        ):
            raise ValueError("Invalid article batch input")
        source_by_id[doc_id] = (title, abstract)
        shap_values = article.get("shap_values")
        if isinstance(shap_values, str):
            try:
                shap_values = json.loads(shap_values)
            except json.JSONDecodeError:
                shap_values = None
        factors = _positive_factors(shap_values, limit=3) if isinstance(shap_values, dict) else []
        payload.append({
            "id": doc_id,
            "title": title.strip(),
            "abstract": _compact_abstract(abstract),
            "positive_factors": [factor["фактор"] for factor in factors],
        })

    response = _query_llm(
        BATCH_ARTICLE_PROMPT,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        max_tokens=2100,
        temperature=0.0,
        timeout=timeout,
    ).strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.I)
    decoded = json.loads(response)
    if not isinstance(decoded, dict) or not isinstance(decoded.get("articles"), list):
        raise ValueError("Article batch response is not an object with articles")

    results = {}
    for item in decoded["articles"]:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("id")
        if not isinstance(doc_id, str) or doc_id not in source_by_id or doc_id in results:
            continue
        source_title, abstract = source_by_id[doc_id]
        translated = _clean_title(item.get("title")) if isinstance(item.get("title"), str) else None
        if not is_valid_title(translated, source_title):
            translated = None
        try:
            analysis = _parse_article_analysis(json.dumps(item, ensure_ascii=False))
            if not is_valid_description(analysis["summary"], abstract):
                analysis = None
        except (ValueError, TypeError):
            analysis = None
        results[doc_id] = {"title": translated, "analysis": analysis}
    return results


def recover_case_result(abstract_text, *, timeout=30):
    if (
        not isinstance(abstract_text, str)
        or len(abstract_text.strip()) < 200
        or not RESULT_EVIDENCE.search(abstract_text)
    ):
        return "None"
    source_numbers = set(re.findall(r"\d+(?:\.\d+)?", abstract_text.replace(",", ".")))
    content = _compact_abstract(abstract_text)
    for attempt in range(2):
        try:
            response = _query_llm(
                CASE_RESULT_PROMPT, content,
                max_tokens=150, temperature=0.0, timeout=timeout,
            ).strip()
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.I)
            data = json.loads(response)
            value = data.get("case_result") if isinstance(data, dict) else None
            if not isinstance(value, str):
                raise ValueError("Invalid case result response")
            value = value.strip()
            if value.casefold() == "none":
                return "None"
            if not value or len(value) > 450 or has_corrupt_text(value):
                raise ValueError("Invalid case result text")
            result_numbers = set(re.findall(r"\d+(?:\.\d+)?", value.replace(",", ".")))
            if not result_numbers <= source_numbers:
                raise ValueError("Case result contains unsupported numbers")
            return value
        except (ValueError, RuntimeError, OpenAIError) as exc:
            logger.warning("Case result attempt %s/2 failed: %s", attempt + 1, type(exc).__name__)
    return "None"


def analyze_abstract(abstract_text, shap_values=None, *, budget=None):
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
        if budget and budget.expired():
            break
        try:
            analysis = _parse_article_analysis(_query_llm(
                ARTICLE_ANALYSIS_PROMPT,
                content,
                max_tokens=500,
                temperature=0.0,
                timeout=budget.request_timeout(30) if budget else 30,
            ))
            if not is_valid_description(analysis["summary"], abstract_text):
                raise ValueError("Article analysis has no valid summary")
            return analysis
        except (ValueError, RuntimeError, OpenAIError) as exc:
            logger.warning(
                "Article analysis attempt %s/%s failed: %s: %s",
                attempt, MAX_ARTICLE_ANALYSIS_ATTEMPTS, type(exc).__name__, exc,
            )
    return empty


def summarize_related_signal(anchor, related_articles):
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


def translate_article_title(title, abstract_text="", *, timeout=60):
    if not isinstance(title, str) or not title.strip():
        return "None"
    source_title = title.strip()
    deadline = time.monotonic() + timeout
    for attempt in range(1, MAX_TITLE_TRANSLATION_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = _clean_title(
                _query_llm(
                    TITLE_TRANSLATION_PROMPT,
                    source_title,
                    max_tokens=384,
                    temperature=0.0,
                    timeout=remaining,
                )
            )
        except (RuntimeError, ValueError, OpenAIError) as exc:
            logger.warning(
                "Title translation attempt %s/%s failed: %s",
                attempt, MAX_TITLE_TRANSLATION_ATTEMPTS, type(exc).__name__,
            )
            continue
        if is_valid_title(result, source_title):
            return result
    return source_title if is_valid_title(source_title, source_title) else "None"


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
