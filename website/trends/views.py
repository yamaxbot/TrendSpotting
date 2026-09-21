from django.shortcuts import render


TRENDS = [
    {
        "rank": 1,
        "slug": "explainable-artificial-intelligence",
        "name": "Explainable Artificial Intelligence (XAI)",
        "area": "Искусственный интеллект",
        "description": "Методы, которые делают решения моделей понятными для исследователей и пользователей.",
        "confidence": 82,
        "direction": "растет",
        "signal": "Публикации ускоряются, а интерес выходит за пределы узкого ML-сообщества.",
        "color": "orange",
        "years": [18, 24, 31, 39, 48, 61, 76, 100, 118],
        "predictors": ["Рост междисциплинарных публикаций", "Ускорение цитируемости", "Расширение авторского сообщества"],
        "sources": [
            ("Towards A Rigorous Science of Interpretable Machine Learning", "Nature Machine Intelligence, 2021"),
            ("Explainable AI: A Review of Machine Learning Interpretability Methods", "ACM Computing Surveys, 2023"),
            ("A Survey of Explainable Artificial Intelligence", "IEEE Access, 2024"),
        ],
    },
    {
        "rank": 2,
        "slug": "self-supervised-learning",
        "name": "Self-supervised learning для научных данных",
        "area": "Машинное обучение",
        "description": "Обучение представлений без ручной разметки для сложных научных корпусов.",
        "confidence": 78,
        "direction": "растет",
        "signal": "Тема быстро переходит из лабораторных экспериментов в прикладные домены.",
        "color": "blue",
        "years": [22, 30, 38, 49, 65, 81, 103, 120, 139],
        "predictors": ["Рост публикаций в прикладных доменах", "Новые benchmark-наборы", "Снижение зависимости от labels"],
        "sources": [("A Survey on Self-supervised Learning", "IEEE TPAMI, 2022"), ("Representation Learning with Contrastive Methods", "ICML, 2023")],
    },
    {
        "rank": 3,
        "slug": "multimodal-models",
        "name": "Мультимодальные модели",
        "area": "Генеративный ИИ",
        "description": "Связка текста, изображений, графов и сигналов в едином научном представлении.",
        "confidence": 74,
        "direction": "растет",
        "signal": "Появляются устойчивые мосты между разными типами научных данных.",
        "color": "violet",
        "years": [20, 25, 34, 46, 59, 72, 89, 111, 128],
        "predictors": ["Конвергенция типов данных", "Рост качества zero-shot моделей", "Новые foundation models"],
        "sources": [("Multimodal Foundation Models", "Communications of the ACM, 2024"), ("Connecting Vision and Language", "NeurIPS, 2023")],
    },
    {
        "rank": 4,
        "slug": "synthetic-data",
        "name": "Синтетические данные для обучения",
        "area": "AI infrastructure",
        "description": "Генерация контролируемых наборов для обучения и проверки научных моделей.",
        "confidence": 71,
        "direction": "растет",
        "signal": "Синтетические корпуса становятся частью воспроизводимого ML-пайплайна.",
        "color": "green",
        "years": [18, 23, 31, 42, 50, 64, 78, 92, 113],
        "predictors": ["Дефицит качественных labels", "Рост privacy-aware методов", "Повторяемость экспериментов"],
        "sources": [("The Future of Synthetic Data", "Nature Machine Intelligence, 2024"), ("Data Generation for Scientific ML", "JMLR, 2023")],
    },
    {
        "rank": 5,
        "slug": "ai-for-materials",
        "name": "AI for Materials Discovery",
        "area": "Материаловедение",
        "description": "Модели для ускоренного поиска новых материалов и свойств соединений.",
        "confidence": 68,
        "direction": "растет",
        "signal": "Стабильный рост публикаций на стыке химии, физики и ML.",
        "color": "coral",
        "years": [20, 28, 35, 40, 51, 62, 75, 89, 101],
        "predictors": ["Рост лабораторных validation-кейсов", "Автоматизация discovery", "Междисциплинарное авторство"],
        "sources": [("Machine Learning for Materials Discovery", "Nature, 2022"), ("Materials Acceleration Platforms", "Nature Reviews Materials, 2024")],
    },
]

for index in range(6, 16):
    TRENDS.append({
        "rank": index,
        "slug": f"emerging-direction-{index}",
        "name": ["Neuro-symbolic AI", "Federated learning", "Digital twins", "AI agents", "Quantum machine learning", "Robotics foundation models", "Scientific knowledge graphs", "Causal representation learning", "Green AI", "Edge intelligence"][index - 6],
        "area": "Перспективное направление",
        "description": "Компактный сигнал из публикационной динамики и структуры научного корпуса.",
        "confidence": 66 - (index - 6),
        "direction": "растет",
        "signal": "Направление показывает устойчивое расширение публикационной базы.",
        "color": "blue",
        "years": [19 + index, 25 + index, 31 + index, 42 + index, 51 + index, 62 + index, 77 + index, 91 + index, 108 + index],
        "predictors": ["Рост публикационной активности", "Расширение тематического ядра", "Новые междисциплинарные связи"],
        "sources": [("Representative research on the direction", "OpenAlex corpus, 2024"), ("Emerging methods and applications", "Selected publications, 2025")],
    })


def dashboard(request):
    query = request.GET.get("q", "ИИ")
    return render(request, "trends/dashboard.html", {"query": query, "trends": TRENDS, "publication_count": "4 239"})


def trend_detail(request, slug):
    trend = next((item for item in TRENDS if item["slug"] == slug), TRENDS[0])
    return render(request, "trends/detail.html", {"trend": trend, "query": request.GET.get("q", "ИИ")})
