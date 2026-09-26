import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import numpy as np
from django.test import SimpleTestCase

from trends import charts, evidence, jobs, statistics, views
from on_demand_parsing.time_budget import SearchBudget


class WebsiteTests(SimpleTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.corpus = root / 'corpus.parquet'
        self.dataset = root / 'dataset.parquet'
        pd.DataFrame([{'doc_id': 'W1', 'pub_year': 2025, 'title': 'Original',
                       'abstract_text': 'Text', 'article_url': 'javascript:alert(1)',
                       'doi': 'https://doi.org/10.1/example',
                       'authorships_json': json.dumps([
                           {'institutions': [
                               {'type': 'company', 'display_name': 'Sensor Lab'},
                               {'type': 'education', 'display_name': 'University'},
                           ]},
                           {'institutions': [
                               {'type': 'company', 'display_name': 'Sensor Lab'},
                           ]},
                       ])}]).to_parquet(self.corpus)
        pd.DataFrame([{'doc_id': 'W1', 'model_confidence': 91., 'shap_values': '{}',
                       'diversity_max_similarity': 0.0,
                       'model_threshold': 0.2436524675,
                       'model_version': 'catboost-random-split-2026-09-v1',
                       'ranking_version': 'intent-relevance-v1',
                       'llm_title': 'Перевод', 'llm_title_version': 'exact-v2',
                       'llm_description': 'Summary',
                        'llm_analysis_version': 'grounded-v2',
                       'llm_problem': 'Обнаружение глюкозы',
                       'llm_advantage': 'Более точный метод',
                       'llm_case_result': 'Показана работа сенсора',
                       'llm_weak_signal': 'Объяснение'}]).to_parquet(self.dataset)
        for name, value in [('get_built_dataset_path', self.dataset),
                            ('get_parsed_corpus_path', self.corpus)]:
            patcher = patch('website.main.' + name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_cached_results_and_details_do_not_start_work(self):
        with patch.object(jobs, 'start') as start, patch.object(evidence, 'start', return_value='queued'):
            response = self.client.get('/', {'q': 'test'})
            self.assertContains(response, 'Перевод')
            self.assertNotContains(response, 'area-tag')
            self.assertNotContains(response, 'trend-state')
            self.assertNotContains(response, 'insight-panel')
            self.assertContains(response, 'Оценка сигнала')
            self.assertFalse(response.context['pending'])
            detail = self.client.get('/trend/result-1/', {'q': 'test'})
            self.assertNotContains(detail, 'area-tag')
            self.assertNotContains(detail, 'не вероятность успеха')
            self.assertContains(detail, 'https://doi.org/10.1/example')
            self.assertNotContains(detail, 'javascript:')
            self.assertEqual(self.client.get('/trend/unknown/', {'q': 'test'}).status_code, 404)
            start.assert_not_called()

    def test_deadline_keeps_completed_card_and_marks_partial_result(self):
        corpus = pd.read_parquet(self.corpus)
        second = corpus.iloc[0].copy()
        second["doc_id"] = "W2"
        corpus = pd.concat([corpus, second.to_frame().T], ignore_index=True)
        corpus.to_parquet(self.corpus, index=False)

        predictions = pd.read_parquet(self.dataset)
        second_prediction = predictions.iloc[0].copy()
        second_prediction["doc_id"] = "W2"
        predictions = pd.concat(
            [predictions, second_prediction.to_frame().T], ignore_index=True,
        )
        predictions.to_parquet(self.dataset, index=False)

        budget = SearchBudget(deadline=0)
        trends = views._load_trends(
            self.dataset, self.corpus, generate_llm_texts=True, budget=budget,
        )
        saved = pd.read_parquet(self.dataset)
        self.assertEqual([trend["doc_id"] for trend in trends], ["W1"])
        self.assertEqual(saved["doc_id"].tolist(), ["W1"])
        self.assertTrue(saved["search_partial"].all())
        self.assertTrue(budget.truncated)
        self.assertTrue(views._dataset_is_ready(self.dataset, self.corpus))
        response = self.client.get("/", {"q": "test"})
        self.assertTrue(response.context["partial"])
        self.assertContains(response, "19 мин 30 сек")

    def test_information_pages_are_reachable_from_navigation(self):
        home = self.client.get('/')
        self.assertContains(home, 'href="/how-it-works/"')
        self.assertContains(home, 'href="/about/"')
        how = self.client.get('/how-it-works/')
        about = self.client.get('/about/')
        self.assertContains(how, 'Четыре шага анализа')
        self.assertContains(how, 'От статьи к направлению')
        self.assertNotContains(how, 'не вероятность')
        self.assertNotContains(how, 'info-clarity-callout')
        self.assertContains(about, 'Газпромбанк.Тех')
        self.assertNotContains(about, 'info-limits')
        self.assertNotContains(about, 'info-hero-symbol')

    def test_detail_displays_verified_years_and_related_source(self):
        related = {
            'years': [
                {'year': 2026, 'count': 2, 'exhaustive': True},
                {'year': 2025, 'count': 4, 'exhaustive': False},
                {'year': 2024, 'count': 1, 'exhaustive': True},
                {'year': 2023, 'count': 0, 'exhaustive': True},
            ],
            'complete': False,
            'search_phrase': 'photonic crystal fiber',
            'sources': [{'title': 'Related paper', 'date': '2025-01-01',
                         'source_type': 'article', 'venue': 'Journal',
                         'url': 'https://doi.org/10.1/related'}],
        }
        with patch.object(evidence, 'load_cached', return_value=related), patch.object(evidence, 'start') as start:
            response = self.client.get('/trend/result-1/', {'q': 'test'})
        self.assertContains(response, '2023–2026')
        self.assertContains(response, 'Найдено не менее 4 публикаций за 2025 год')
        self.assertNotContains(response, '≥4')
        self.assertContains(response, 'class="publication-axis-value"')
        self.assertContains(response, 'Related paper')
        self.assertContains(response, 'Суть исследования')
        self.assertContains(response, 'Что предлагается')
        self.assertContains(response, 'Обнаружение глюкозы')
        self.assertContains(response, 'Компании среди организаций авторов')
        self.assertContains(response, 'Sensor Lab')
        self.assertNotContains(response, 'University')
        self.assertContains(response, 'class="source-link-label"')
        self.assertContains(response, 'Показана работа сенсора')
        self.assertContains(response, 'class="publication-chart"')
        self.assertContains(response, '<polyline')
        self.assertNotContains(response, 'year-stat-track')
        self.assertNotContains(response, 'Поисковая формулировка')
        self.assertNotContains(response, 'Самая ранняя найденная работа')
        self.assertContains(response, '2024')
        start.assert_not_called()

    def test_detail_prefers_grounded_signal_and_preserves_tabular_fallback(self):
        related = {
            'years': [], 'sources': [],
            'signal_text': 'Две работы применяют сенсор для обнаружения глюкозы.',
        }
        with patch.object(evidence, 'load_cached', return_value=related):
            response = self.client.get('/trend/result-1/', {'q': 'test'})
        self.assertContains(response, related['signal_text'])
        self.assertContains(response, 'Дополнительных работ, прошедших строгую проверку близости, не найдено.')
        self.assertNotContains(response, 'Объяснение')

        related['signal_text'] = None
        with patch.object(evidence, 'load_cached', return_value=related):
            response = self.client.get('/trend/result-1/', {'q': 'test'})
        self.assertContains(response, 'Объяснение')

    def test_detail_keeps_sources_when_thematic_counts_are_unavailable(self):
        related = {
            'years': None, 'sources': [{
                'title': 'Related work', 'date': '2025-01-01',
                'source_type': 'article', 'venue': None,
                'url': 'https://doi.org/10.1/related',
            }],
        }
        with patch.object(evidence, 'load_cached', return_value=related):
            response = self.client.get('/trend/result-1/', {'q': 'test'})
        self.assertContains(response, 'Не удалось получить годовую статистику')
        self.assertContains(response, 'Related work')

    def test_evidence_status_endpoint(self):
        with patch.object(evidence, 'load_cached', return_value=None), patch.object(evidence, 'start', return_value='queued'):
            self.client.get('/trend/result-1/', {'q': 'test'})
        with patch.object(evidence, 'status', return_value='complete'):
            response = self.client.get('/trend/result-1/evidence-status/', {'q': 'test'})
        self.assertEqual(response.json(), {'status': 'complete'})

    def test_company_line_is_hidden_when_no_company_is_affiliated(self):
        corpus = pd.read_parquet(self.corpus)
        corpus.loc[0, 'authorships_json'] = '[]'
        corpus.to_parquet(self.corpus, index=False)
        with patch.object(evidence, 'load_cached', return_value=None), \
             patch.object(evidence, 'start', return_value='queued'):
            response = self.client.get('/trend/result-1/', {'q': 'test'})
        self.assertNotContains(response, 'Компании среди организаций авторов')

    def test_new_query_returns_pending_without_blocking(self):
        self.dataset.unlink()
        with patch.object(jobs, 'start', return_value='queued') as start:
            response = self.client.get('/', {'q': 'new'})
        self.assertTrue(response.context['pending'])
        self.assertContains(response, 'data-status-url=')
        start.assert_called_once_with('new')

    def test_invalid_and_empty_queries_do_not_start_work(self):
        with patch.object(jobs, 'start') as start:
            self.assertEqual(self.client.get('/').status_code, 200)
            self.assertEqual(self.client.get('/', {'q': 'x' * 201}).status_code, 400)
            start.assert_not_called()

    def test_corrupt_cache_is_not_ready(self):
        self.dataset.write_bytes(b'broken')
        self.assertFalse(views._dataset_is_ready(self.dataset, self.corpus))

    def test_empty_search_is_a_finished_result(self):
        from website.main import build_dataset_for_query

        pd.read_parquet(self.corpus).iloc[:0].to_parquet(self.corpus, index=False)
        with patch('on_demand_parsing.parser.run_parser', return_value=([], {})):
            build_dataset_for_query('rare')

        self.assertTrue(views._dataset_is_ready(self.dataset, self.corpus))
        with patch.object(jobs, 'start') as start:
            response = self.client.get('/', {'q': 'rare'})
        self.assertFalse(response.context['pending'])
        self.assertContains(response, 'Подходящие кандидаты в тренды по этому запросу не найдены')
        start.assert_not_called()

    def test_missing_explanation_and_untranslated_title_invalidate_cache(self):
        data = pd.read_parquet(self.dataset)
        data.loc[0, 'llm_weak_signal'] = None
        data.to_parquet(self.dataset)
        self.assertFalse(views._dataset_is_ready(self.dataset, self.corpus))
        data.loc[0, 'llm_weak_signal'] = 'Объяснение'
        data.loc[0, 'llm_title'] = 'Original'
        data.to_parquet(self.dataset)
        self.assertFalse(views._dataset_is_ready(self.dataset, self.corpus))

    def test_status_failure_and_complete(self):
        with patch.object(jobs, 'status', return_value='failed'):
            self.assertEqual(self.client.get('/search/status/', {'q': 'test'}).json()['status'], 'failed')
        with patch.object(jobs, 'status', return_value='missing'):
            self.assertEqual(self.client.get('/search/status/', {'q': 'test'}).json()['status'], 'complete')

    def test_duplicate_jobs_are_not_submitted_twice(self):
        with patch.object(jobs, '_jobs', {}), patch.object(jobs._executor, 'submit') as submit:
            jobs.start('same')
            jobs.start('same')
            submit.assert_called_once()

    def test_empty_predictions_render_and_cache(self):
        df = pd.read_parquet(self.dataset).iloc[:0]
        df.to_parquet(self.dataset)
        self.assertTrue(views._dataset_is_ready(self.dataset, self.corpus))
        response = self.client.get('/', {'q': 'empty'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['trends'], [])

    def test_invalid_llm_content_is_reported(self):
        from website import query_llm
        with patch.object(query_llm, 'API_KEY', 'test'), patch.object(query_llm.requests, 'post') as post:
            post.return_value.json.return_value = {'choices': [{'message': {'content': None}}]}
            with self.assertRaises(RuntimeError):
                query_llm.summarize_abstract('text')

    def test_structured_analysis_uses_one_request_and_rejects_invalid_fields(self):
        from website import query_llm

        payload = ('{"summary":"Исследован сенсор.","problem":"Контроль глюкозы.",'
                   '"advantage":"None","case_result":"Получен прототип.",'
                   '"weak_signal":"Прототип измеряет глюкозу оптическим методом."}')
        factors = {'novelty_raw': {'shap': 0.4, 'value': 0.8}}
        with patch.object(query_llm, '_query_llm', return_value=payload) as call:
            result = query_llm.analyze_abstract('An abstract about a sensor.', factors)
        self.assertEqual(call.call_count, 1)
        self.assertIn('An abstract about a sensor.', call.call_args.args[1])
        self.assertIn('отличие от более ранних работ темы', call.call_args.args[1])
        self.assertEqual(result['problem'], 'Контроль глюкозы.')
        self.assertEqual(result['advantage'], 'None')
        self.assertEqual(result['weak_signal'], 'Прототип измеряет глюкозу оптическим методом.')
        with self.assertRaises(ValueError):
            query_llm._parse_article_analysis('{"summary":"Only one field"}')

    def test_article_analysis_retries_empty_and_transient_responses(self):
        from website import query_llm

        empty = json.dumps(dict.fromkeys(
            ('summary', 'problem', 'advantage', 'case_result', 'weak_signal'), 'None'
        ))
        valid = json.dumps({
            'summary': 'Sensor measures glucose.',
            'problem': 'Glucose monitoring.',
            'advantage': 'Optical measurement.',
            'case_result': 'Prototype demonstrated.',
            'weak_signal': 'Optical sensor prototype was demonstrated.',
        })
        with patch.object(
            query_llm, '_query_llm',
            side_effect=[empty, query_llm.requests.Timeout(), valid],
        ) as request:
            result = query_llm.analyze_abstract('A sensor abstract.')
        self.assertEqual(request.call_count, 3)
        self.assertEqual(result['summary'], 'Sensor measures glucose.')

    def test_article_analysis_accepts_scientific_units(self):
        from website import query_llm

        payload = json.dumps({
            'summary': 'Представлен перестраиваемый генератор сигналов.',
            'problem': 'Снизить фазовый шум.',
            'advantage': 'Широкий диапазон частот.',
            'case_result': 'Показаны 10 кГц, 5 мСм/см, ток 2 нА и барьер 0,7 кэВ.',
            'weak_signal': 'Прототип работает при 10 кГц и снижает фазовый шум.',
        })
        with patch.object(query_llm, '_query_llm', return_value=payload) as request:
            result = query_llm.analyze_abstract('A microwave oscillator is measured.')
        request.assert_called_once()
        self.assertIn('10 кГц', result['case_result'])
        self.assertIn('мСм/см', result['case_result'])
        self.assertIn('нА', result['case_result'])
        self.assertIn('кэВ', result['case_result'])
        self.assertTrue(query_llm.has_corrupt_text('Показан пиролЦельный результат.'))

    def test_scientific_camelcase_name_is_not_corrupt_text(self):
        from website import query_llm

        summary = 'Схема ФитцХью—Нагумо регулирует активность нейронной цепи.'
        self.assertFalse(query_llm.has_corrupt_text(summary))
        self.assertTrue(query_llm.is_valid_description(summary, 'Neural circuit abstract.'))
        payload = json.dumps({
            'summary': summary,
            'problem': 'Управление активностью цепи.',
            'advantage': 'Регулируемое шунтирование тока.',
            'case_result': 'Показана работа схемы.',
            'weak_signal': 'Проверена мемристивная нейронная схема.',
        }, ensure_ascii=False)
        self.assertEqual(query_llm._parse_article_analysis(payload)['summary'], summary)
        self.assertTrue(query_llm.has_corrupt_text('Обнаружен пиролЦельный результат.'))

    def test_cached_empty_analysis_is_repaired_without_model_rebuild(self):
        from website import query_llm

        cached = pd.read_parquet(self.dataset)
        for field in ('llm_description', 'llm_problem', 'llm_advantage', 'llm_case_result'):
            cached.loc[0, field] = 'None'
        cached.to_parquet(self.dataset)
        self.assertFalse(views._dataset_is_ready(self.dataset, self.corpus))
        analysis = {
            'summary': 'Sensor measures glucose.',
            'problem': 'Glucose monitoring.',
            'advantage': 'Optical measurement.',
            'case_result': 'Prototype demonstrated.',
            'weak_signal': 'Optical sensor prototype was demonstrated.',
        }
        with patch('website.main.model_dataset_is_ready', return_value=True), \
             patch('website.main.build_dataset_for_query') as build, \
             patch.object(query_llm, 'analyze_abstract', return_value=analysis) as analyze, \
             patch.object(jobs, '_jobs', {}):
            jobs._run('test')
            self.assertEqual(jobs.status('test'), 'complete')
        build.assert_not_called()
        analyze.assert_called_once()
        self.assertTrue(views._dataset_is_ready(self.dataset, self.corpus))
        result = pd.read_parquet(self.dataset)
        self.assertEqual(result.loc[0, 'llm_case_result'], 'Prototype demonstrated.')

    def test_failed_analysis_is_not_marked_as_completed(self):
        from website import query_llm

        cached = pd.read_parquet(self.dataset)
        cached.loc[0, 'llm_description'] = 'None'
        cached.to_parquet(self.dataset)
        empty = dict.fromkeys(
            ('summary', 'problem', 'advantage', 'case_result', 'weak_signal'), 'None'
        )
        with patch('website.main.model_dataset_is_ready', return_value=True), \
             patch('website.main.build_dataset_for_query') as build, \
             patch.object(query_llm, 'analyze_abstract', return_value=empty), \
             patch.object(jobs, '_jobs', {}):
            jobs._run('test')
            self.assertEqual(jobs.status('test'), 'failed')
        build.assert_not_called()
        self.assertFalse(views._dataset_is_ready(self.dataset, self.corpus))
        self.assertTrue(pd.isna(pd.read_parquet(self.dataset).loc[0, 'llm_analysis_version']))

    def test_old_text_cache_is_enriched_without_rebuilding_model(self):
        from website import query_llm

        cached = pd.read_parquet(self.dataset).drop(columns=[
            'llm_analysis_version', 'llm_problem', 'llm_advantage', 'llm_case_result',
        ])
        cached.to_parquet(self.dataset)
        analysis = {
            'summary': 'Краткое описание.', 'problem': 'Задача.',
            'advantage': 'Преимущество.', 'case_result': 'Результат.',
            'weak_signal': 'Конкретное обоснование статьи.',
        }
        with patch.object(query_llm, 'analyze_abstract', return_value=analysis) as call:
            views._load_trends(self.dataset, self.corpus, generate_llm_texts=True)
        self.assertEqual(call.call_count, 1)
        self.assertTrue(views._dataset_is_ready(self.dataset, self.corpus))
        result = pd.read_parquet(self.dataset)
        self.assertEqual(result.loc[0, 'llm_case_result'], 'Результат.')
        self.assertEqual(result.loc[0, 'llm_weak_signal'], 'Конкретное обоснование статьи.')

    def test_old_generic_signal_is_replaced_without_rebuilding_model(self):
        from website import query_llm

        cached = pd.read_parquet(self.dataset)
        cached.loc[0, 'llm_analysis_version'] = 'grounded-v1'
        cached.loc[0, 'llm_weak_signal'] = 'На оценку статьи положительно повлияли общие факторы.'
        cached.to_parquet(self.dataset)
        self.assertFalse(views._dataset_is_ready(self.dataset, self.corpus))
        analysis = {
            'summary': 'Оптический сенсор измеряет концентрацию глюкозы.',
            'problem': 'Контроль глюкозы.', 'advantage': 'None',
            'case_result': 'Показана работа сенсора.',
            'weak_signal': 'Сенсор применяет конкретный оптический метод для измерения глюкозы.',
        }
        with patch.object(query_llm, 'analyze_abstract', return_value=analysis) as call:
            views._load_trends(self.dataset, self.corpus, generate_llm_texts=True)
        call.assert_called_once()
        updated = pd.read_parquet(self.dataset)
        self.assertEqual(updated.loc[0, 'llm_weak_signal'], analysis['weak_signal'])
        self.assertTrue(views._dataset_is_ready(self.dataset, self.corpus))

    def test_old_specific_signal_is_kept_without_another_llm_call(self):
        from website import query_llm

        cached = pd.read_parquet(self.dataset)
        cached.loc[0, 'llm_analysis_version'] = 'grounded-v1'
        cached.loc[0, 'llm_weak_signal'] = 'Сенсор использует фотонный кристалл для анализа глюкозы.'
        cached.to_parquet(self.dataset)
        with patch.object(query_llm, 'analyze_abstract') as analyze:
            views._load_trends(self.dataset, self.corpus, generate_llm_texts=True)
        analyze.assert_not_called()
        updated = pd.read_parquet(self.dataset)
        self.assertEqual(
            updated.loc[0, 'llm_weak_signal'],
            'Сенсор использует фотонный кристалл для анализа глюкозы.',
        )
        self.assertTrue(views._dataset_is_ready(self.dataset, self.corpus))

    def test_tabular_fallback_keeps_article_specific_context(self):
        from website.query_llm import fallback_weak_signal

        factors = {'topic_publication_growth': {'shap': 0.4, 'value': 1.2}}
        sensor = fallback_weak_signal(
            factors, summary='Сенсор измеряет концентрацию глюкозы.',
        )
        robot = fallback_weak_signal(
            factors, summary='Робот перемещается по сложной поверхности.',
        )
        self.assertNotEqual(sensor, robot)
        self.assertIn('глюкозы', sensor)
        self.assertIn('Робот', robot)

    def test_current_analysis_cache_does_not_call_llm(self):
        from website import query_llm

        with patch.object(query_llm, 'analyze_abstract') as analyze, \
             patch.object(query_llm, 'translate_article_title') as translate:
            views._load_trends(self.dataset, self.corpus, generate_llm_texts=True)
        analyze.assert_not_called()
        translate.assert_not_called()

    def test_old_signal_disclaimer_is_hidden_from_cached_result(self):
        data = pd.read_parquet(self.dataset)
        data.loc[0, 'llm_weak_signal'] = (
            'Положительно повлиял рост публикаций. '
            'Это признаки в данных, а не доказательство появления нового тренда.'
        )
        data.to_parquet(self.dataset)
        with patch.object(evidence, 'load_cached', return_value=None), \
             patch.object(evidence, 'start', return_value='queued'):
            detail = self.client.get('/trend/result-1/', {'q': 'test'})
        self.assertContains(detail, 'Положительно повлиял рост публикаций.')
        self.assertNotContains(detail, 'не доказательство появления нового тренда')

    def test_translated_title_is_exact_and_uses_no_abstract(self):
        from website import query_llm

        translated = 'Перевод: Точное переведённое название без сокращений'
        with patch.object(query_llm, '_query_llm', return_value=translated) as query:
            result = query_llm.translate_article_title('Original', 'Abstract')

        self.assertEqual(result, 'Точное переведённое название без сокращений')
        self.assertEqual(query.call_args.args[1], 'Original')
        self.assertNotIn('Abstract', query.call_args.args[1])
        self.assertEqual(query.call_args.kwargs['temperature'], 0.0)

    def test_llm_output_validation_detects_observed_corruption(self):
        from website import query_llm

        source = 'Evolution of Photonic Crystal Fiber-Based Smart Sensors'
        self.assertFalse(query_llm.is_valid_title(source, source))
        self.assertFalse(query_llm.is_valid_weak_signal(None))
        self.assertFalse(query_llm.is_valid_title('чиков на основе волокна', source))
        self.assertFalse(query_llm.is_valid_title('None', source))
        self.assertTrue(query_llm.is_valid_title(
            'Эволюция интеллектуальных датчиков на основе волокна', source,
        ))
        self.assertFalse(query_llm.is_valid_description(
            'Цель — оценить потенциал пиролЦель —иза масла.ения', 'Abstract',
        ))
        self.assertFalse(query_llm.is_valid_description(
            'енивает потенциал пиролиза.', 'Abstract',
        ))
        self.assertFalse(query_llm.is_valid_weak_signal(
            'Метод объединяет микроfluidics и сенсоры.',
        ))

    def test_title_translation_retries_and_falls_back_safely(self):
        from website import query_llm

        source = 'Complete source title'
        valid = 'Полный перевод названия'
        with patch.object(
            query_llm, '_query_llm', side_effect=['обрезанный перевод', valid]
        ) as query:
            self.assertEqual(query_llm.translate_article_title(source), valid)
            self.assertEqual(query.call_count, 2)

        with patch.object(
            query_llm, '_query_llm', return_value='обрезанный перевод'
        ) as query:
            self.assertEqual(query_llm.translate_article_title(source), source)
            self.assertEqual(query.call_count, 2)

        with patch.object(
            query_llm, '_query_llm',
            side_effect=[query_llm.requests.Timeout(), valid],
        ) as query:
            self.assertEqual(query_llm.translate_article_title(source), valid)
            self.assertEqual(query.call_count, 2)

    def test_semantic_duplicates_are_replaced_by_lower_ranked_articles(self):
        from website import diversity

        ranked = pd.DataFrame([
            {'doc_id': 'W1', 'model_confidence': 99.0},
            {'doc_id': 'W2', 'model_confidence': 98.0},
            {'doc_id': 'W3', 'model_confidence': 90.0},
        ])
        texts = pd.DataFrame([
            {'doc_id': 'W1', 'abstract_text': 'same event one'},
            {'doc_id': 'W2', 'abstract_text': 'same event two'},
            {'doc_id': 'W3', 'abstract_text': 'different discovery'},
        ])

        class FakeModel:
            def encode(self, values, **kwargs):
                return np.array([
                    [1.0, 0.0],
                    [0.99, 0.01],
                    [0.0, 1.0],
                ], dtype=np.float32)

        with patch.object(diversity, '_get_model', return_value=FakeModel()):
            indices, similarities = diversity.select_diverse_top(
                ranked, texts, limit=2, similarity_threshold=0.78,
            )

        self.assertEqual(indices, [0, 2])
        self.assertEqual(similarities, [0.0, 0.0])

    def test_diversity_stops_embedding_after_top_results_are_found(self):
        from website import diversity

        ranked = pd.DataFrame({
            'doc_id': [f'W{i}' for i in range(200)],
            'model_confidence': list(range(200, 0, -1)),
        })
        texts = pd.DataFrame({
            'doc_id': ranked['doc_id'],
            'abstract_text': [f'abstract {i}' for i in range(200)],
        })

        class FakeModel:
            def __init__(self):
                self.encoded = 0

            def encode(self, values, **kwargs):
                self.encoded += len(values)
                return np.tile([1.0, 0.0], (len(values), 1))

        model = FakeModel()
        with patch.object(diversity, '_get_model', return_value=model):
            indices, _ = diversity.select_diverse_top(ranked, texts, limit=1)
        self.assertEqual(indices, [0])
        self.assertEqual(model.encoded, 64)

    def test_random_model_input_reproduces_training_features(self):
        from website.main import _prepare_model_input

        dataset = pd.DataFrame([
            {
                'doc_id': 'W1', 'pub_year': 2025, 'source_tier': 'journal',
                'commercial_maturity_index': np.nan,
                'topic_historical_volume': 4.0, 'topic_local_volume': 2.0,
                'historical_author_count': 6.0,
            },
            {
                'doc_id': 'W2', 'pub_year': 2025, 'source_tier': None,
                'commercial_maturity_index': 0.5,
                'topic_historical_volume': 2.0, 'topic_local_volume': 1.0,
                'historical_author_count': 3.0,
            },
        ])

        class FakeModel:
            feature_names_ = [
                'split', 'source_tier', 'commercial_maturity_index',
                'topic_historical_share', 'topic_local_share',
                'historical_author_share',
            ]

            def get_cat_feature_indices(self):
                return [0, 1]

        model_input, categorical = _prepare_model_input(dataset, FakeModel())

        self.assertEqual(categorical, ['split', 'source_tier'])
        self.assertEqual(model_input['split'].tolist(), ['inference', 'inference'])
        self.assertEqual(model_input.loc[1, 'source_tier'], 'missing')
        self.assertEqual(model_input.loc[0, 'commercial_maturity_index'], -1.0)
        self.assertAlmostEqual(model_input.loc[0, 'topic_historical_share'], 2.0, places=4)
        self.assertAlmostEqual(model_input.loc[0, 'topic_local_share'], 1.0, places=4)
        self.assertAlmostEqual(model_input.loc[0, 'historical_author_share'], 3.0, places=4)

    def test_reviews_do_not_enter_top_ranked_articles(self):
        from website import main
        from website.article_type import is_review_article

        self.assertTrue(is_review_article('A new method', 'review'))
        self.assertTrue(is_review_article('A systematic review of sensors', 'article'))
        self.assertTrue(is_review_article('Обзор методов диагностики', None))
        self.assertFalse(is_review_article('A new sensor for glucose monitoring', 'article'))
        self.assertTrue(is_review_article(
            'Applications and Advances of Machine Learning', 'article',
            'This paper reviews recent progress in the application of ML techniques.',
        ))
        self.assertTrue(is_review_article(
            'Toward AI ecosystems', 'article',
            'Here, we critically review the progress of AI applications.',
        ))
        self.assertFalse(is_review_article(
            'Guidelines for imaging', 'article',
            'We demonstrate atomic-resolution imaging with a new method.',
        ))

        pd.DataFrame([
            {'doc_id': 'W1', 'feature': 1.0},
            {'doc_id': 'W2', 'feature': 2.0},
            {'doc_id': 'W3', 'feature': 3.0},
        ]).to_parquet(self.dataset, index=False)
        pd.DataFrame([
            {'doc_id': 'W1', 'title': 'New technology', 'abstract_text': 'Text', 'work_type': 'review'},
            {'doc_id': 'W2', 'title': 'A survey of sensors', 'abstract_text': 'Text', 'work_type': 'article'},
            {'doc_id': 'W3', 'title': 'New glucose sensor', 'abstract_text': 'Text', 'work_type': 'article'},
        ]).to_parquet(self.corpus, index=False)

        class FakeModel:
            feature_names_ = ['feature']
            classes_ = [0, 1]

            def load_model(self, path):
                pass

            def predict_proba(self, values):
                return np.array([[.1, .9], [.2, .8], [.3, .7]])

            def get_feature_importance(self, pool, type):
                return np.array([[.1, .0]])

        def select(ranked, texts, limit):
            self.assertEqual(ranked['doc_id'].tolist(), ['W3'])
            return ranked.index.tolist(), [0.0]

        with patch.object(main, 'CatBoostClassifier', return_value=FakeModel()), \
             patch.object(main, '_prepare_model_input', side_effect=lambda df, model: (df[['feature']], [])), \
             patch.object(main, '_load_model_threshold', return_value=.5), \
             patch.object(main, 'Pool', return_value=None), \
             patch('website.diversity.select_diverse_top', side_effect=select):
            main.add_model_targets(self.dataset, self.corpus)

        result = pd.read_parquet(self.dataset)
        self.assertEqual(result['doc_id'].tolist(), ['W3'])
        self.assertEqual(result['ranking_version'].tolist(), [main.RANKING_VERSION])


class RelevanceGateTests(SimpleTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.dataset = Path(self.directory.name) / 'dataset.parquet'

    def test_original_intent_is_sent_with_article_and_response_is_validated(self):
        from website import relevance_gate

        works = [
            {'id': 'W1', 'title': 'AI-powered mammography screening', 'abstract': 'AI detects cancer.'},
            {'id': 'W2', 'title': 'Protecting language models from prompt injection',
             'abstract': 'The method detects attacks against AI systems.'},
        ]
        with patch.object(relevance_gate, '_query_llm',
                          return_value='{"relevant_ids":["W2"]}') as llm:
            self.assertEqual(relevance_gate._classify_batch('Защита ИИ', works), {'W2'})
        self.assertIn('Защита ИИ', llm.call_args.args[1])
        self.assertIn('AI-powered mammography', llm.call_args.args[1])
        with patch.object(relevance_gate, '_query_llm',
                          return_value='{"relevant_ids":["W999"]}'):
            with self.assertRaises(RuntimeError):
                relevance_gate._classify_batch('Защита ИИ', works)

        with patch.object(relevance_gate, '_query_llm', side_effect=[
            RuntimeError('invalid response'), '{"relevant_ids":["W2"]}',
        ]) as llm:
            self.assertEqual(relevance_gate._classify_batch('Защита ИИ', works), {'W2'})
        self.assertEqual(llm.call_count, 2)

    def test_rejected_works_are_replaced_and_decisions_cached(self):
        from website import relevance_gate

        ranked = pd.DataFrame({
            'doc_id': [f'W{i}' for i in range(23)],
            'model_confidence': list(range(23, 0, -1)),
        })
        texts = pd.DataFrame({
            'doc_id': [f'W{i}' for i in range(23)],
            'title': [f'Title {i}' for i in range(23)],
            'abstract_text': [f'Abstract {i}' for i in range(23)],
        })
        cache = self.dataset.with_name('relevance.json')

        def classify(query, works):
            return {work['id'] for work in works if work['id'] in {'W1', 'W21'}}

        def select(rows, article_texts, limit):
            indices = rows.index.tolist()[:limit]
            return indices, [0.0] * len(indices)

        with patch.object(relevance_gate, '_classify_batch', side_effect=classify) as llm, \
             patch('website.diversity.select_diverse_top', side_effect=select):
            indices, similarities = relevance_gate.select_relevant_diverse_top(
                ranked, texts, 'Защита ИИ', cache, limit=2,
            )
            self.assertEqual(indices, [1, 21])
            self.assertEqual(similarities, [0.0, 0.0])
            self.assertEqual(llm.call_count, 2)
            relevance_gate.select_relevant_diverse_top(
                ranked, texts, 'Защита ИИ', cache, limit=2,
            )
            self.assertEqual(llm.call_count, 2)

    def test_empty_or_unclear_result_is_not_filled_with_unrelated_articles(self):
        from website import relevance_gate

        ranked = pd.DataFrame({'doc_id': ['W1'], 'model_confidence': [90.0]})
        texts = pd.DataFrame({'doc_id': ['W1'], 'title': ['Mammography with AI'],
                              'abstract_text': ['AI detects cancer.']})
        cache = self.dataset.with_name('relevance-empty.json')
        with patch.object(relevance_gate, '_classify_batch', return_value=set()):
            self.assertEqual(
                relevance_gate.select_relevant_diverse_top(ranked, texts, 'Защита ИИ', cache),
                ([], []),
            )

    def test_expired_budget_uses_first_completed_relevance_batch(self):
        from website import relevance_gate

        ranked = pd.DataFrame({
            'doc_id': [f'W{i}' for i in range(25)],
            'model_confidence': list(range(25, 0, -1)),
        })
        texts = pd.DataFrame({
            'doc_id': [f'W{i}' for i in range(25)],
            'title': [f'Title {i}' for i in range(25)],
            'abstract_text': [f'Abstract {i}' for i in range(25)],
        })
        budget = SearchBudget(deadline=0)
        cache = self.dataset.with_name('relevance-budget.json')

        def select(rows, _texts, limit):
            indices = rows.index.tolist()[:limit]
            return indices, [0.0] * len(indices)

        with patch.object(relevance_gate, '_classify_batch', return_value={'W1', 'W21'}) as classify, \
             patch('website.diversity.select_diverse_top', side_effect=select):
            indices, _ = relevance_gate.select_relevant_diverse_top(
                ranked, texts, 'test', cache, limit=2, budget=budget,
            )
        self.assertEqual(indices, [1])
        self.assertEqual(classify.call_count, 1)
        self.assertTrue(budget.truncated)


class EvidenceTests(SimpleTestCase):
    def test_related_signal_requires_two_valid_supporting_sources(self):
        from website import query_llm

        anchor = {'doc_id': 'W1', 'title': 'New glucose sensor',
                  'abstract_text': 'A sensor uses a new optical method.', 'pub_year': '2025'}
        articles = [
            {'doc_id': 'W2', 'year': 2025, 'title': 'Optical glucose detector',
             'abstract_text': 'The method detects glucose.'},
            {'doc_id': 'W3', 'year': 2026, 'title': 'Optical glucose measurement',
             'abstract_text': 'The approach measures glucose.'},
        ]
        signal = 'Две работы применяют оптический метод для измерения глюкозы.'
        good = json.dumps({'signal': signal, 'supporting_ids': ['W2', 'W3']})
        with patch.object(query_llm, '_query_llm', return_value=good) as call:
            self.assertEqual(query_llm.summarize_related_signal(anchor, articles), signal)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs['temperature'], 0.0)
        self.assertIn('Optical glucose detector', call.call_args.args[1])

        unsupported = json.dumps({'signal': signal, 'supporting_ids': ['W2', 'W9']})
        with patch.object(query_llm, '_query_llm', return_value=unsupported):
            self.assertIsNone(query_llm.summarize_related_signal(anchor, articles))
        with patch.object(query_llm, '_query_llm', return_value=good) as call:
            self.assertIsNone(query_llm.summarize_related_signal(anchor, articles[:1]))
        call.assert_not_called()

    def test_evidence_uses_strict_sources_without_caching_abstracts(self):
        anchor = {'doc_id': 'W1', 'title': 'Optical glucose sensor',
                  'abstract_text': 'A new optical glucose sensor.',
                  'primary_topic_id': 'T123', 'pub_year': '2025'}

        def work(doc_id, title):
            return {
                'id': 'https://openalex.org/' + doc_id, 'title': title,
                'doi': 'https://doi.org/10.1/' + doc_id, 'type': 'article',
                'abstract_inverted_index': {'Optical': [0], 'glucose': [1], 'sensor': [2]},
            }

        pages = {
            2026: ([work('W2', 'Optical glucose sensor prototype'),
                    work('W3', 'Optical glucose sensor testing')], 2, True),
            2025: ([], 0, True), 2024: ([], 0, True), 2023: ([], 0, True),
        }

        class FakeEncoder:
            def encode(self, texts, **kwargs):
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        broad_years = [{'year': year, 'count': 20, 'exhaustive': True} for year in evidence.YEARS]
        with patch.object(evidence, 'publication_counts', return_value=('"optical glucose"', broad_years)), \
             patch.object(evidence, '_fetch_year', side_effect=lambda phrase, topic, year, session: pages[year]), \
             patch.object(evidence, 'summarize_related_signal', return_value='Конкретный вывод.') as summarize:
            result = evidence.build_evidence(anchor, session=object(), encoder=FakeEncoder())
        self.assertEqual(result['signal_text'], 'Конкретный вывод.')
        self.assertEqual([item['count'] for item in result['years']], [20] * 4)
        self.assertEqual([item['count'] for item in result['strict_years']], [2, 0, 0, 0])
        self.assertEqual(len(summarize.call_args.args[1]), 2)
        self.assertEqual([source['doc_id'] for source in result['sources']], ['W2', 'W3'])
        self.assertNotIn('abstract_text', result['sources'][0])

    def test_thematic_count_failure_does_not_remove_strict_sources(self):
        anchor = {'doc_id': 'W1', 'title': 'Optical glucose sensor',
                  'abstract_text': 'Optical glucose measurements.',
                  'primary_topic_id': 'T123'}
        work = {'id': 'https://openalex.org/W2', 'title': 'Optical glucose sensor test',
                'doi': 'https://doi.org/10.1/W2', 'type': 'article',
                'abstract_inverted_index': {'Optical': [0], 'glucose': [1], 'sensor': [2]}}

        class FakeEncoder:
            def encode(self, texts, **kwargs):
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        with patch.object(evidence.logger, 'exception'), \
             patch.object(evidence, 'publication_counts', side_effect=evidence.requests.Timeout), \
             patch.object(evidence, '_fetch_year', side_effect=lambda phrase, topic, year, session: (
                 [work] if year == 2025 else [], 1 if year == 2025 else 0, True,
             )):
            result = evidence.build_evidence(anchor, session=object(), encoder=FakeEncoder())
        self.assertIsNone(result['years'])
        self.assertFalse(result['complete'])
        self.assertEqual([source['doc_id'] for source in result['sources']], ['W2'])

    def test_company_affiliations_are_deduplicated_and_optional(self):
        from trends.organizations import company_names

        self.assertEqual(company_names('[{"institutions": [{"type": "education", "display_name": "University"}]}]'), [])
        self.assertEqual(company_names('[{"institutions": [{"type": "company", "display_name": "Acme"}]}, {"institutions": [{"type": "company", "display_name": "acme"}]}]'), ['Acme'])
        self.assertEqual(company_names('not json'), [])

    def test_publication_chart_sorts_years_and_builds_left_axis(self):
        years = [
            {'year': 2023, 'count': 0, 'exhaustive': True},
            {'year': 2024, 'count': 2, 'exhaustive': True},
            {'year': 2025, 'count': 5, 'exhaustive': True},
            {'year': 2026, 'count': 7, 'exhaustive': False},
        ]
        chart = charts.publication_timeline(list(reversed(years)))
        self.assertEqual([point['year'] for point in chart['points']], [2023, 2024, 2025, 2026])
        self.assertEqual([tick['value'] for tick in chart['ticks']], [0, 2, 4, 6, 8])
        self.assertIn('не менее 7', chart['points'][-1]['tooltip'])
        self.assertEqual(len(chart['line'].split()), 4)

    def test_publication_chart_uses_readable_ticks_for_thirteen_articles(self):
        chart = charts.publication_timeline([
            {'year': 2023, 'count': 1, 'exhaustive': True},
            {'year': 2026, 'count': 13, 'exhaustive': False},
        ])
        self.assertEqual([tick['value'] for tick in chart['ticks']], [0, 5, 10, 15])
        self.assertEqual(chart['points'][-1]['y'], 52)

    def test_search_phrase_preserves_application_and_short_ai_term(self):
        self.assertEqual(
            evidence._search_phrase('A systematic review of AI, VR, and LLM applications in special education'),
            'special education ai',
        )

    def test_thematic_query_uses_article_subject_not_misclassified_topic(self):
        self.assertEqual(
            statistics.thematic_query(
                'Enhancing IoT Security Using Machine Learning for Port Scan Attack Detection',
                'IoT networks face port scanning attacks. Port scanning requires early detection.',
            ),
            '"Port Scan" AND IoT',
        )
        self.assertEqual(
            statistics.thematic_query(
                'Playing the Fool: Jailbreaking LLMs with Out-of-Distribution Strategy',
                'LLMs remain vulnerable to jailbreaking. Jailbreaking bypasses guardrails.',
            ),
            '"Jailbreaking LLMs"',
        )

    def test_thematic_counts_use_one_grouped_request_without_topic_or_scan_limit(self):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {'group_by': [
                    {'key': '2026', 'count': 27}, {'key': '2025', 'count': 19},
                    {'key': '2023', 'count': 4},
                ]}

        class FakeSession:
            def __init__(self):
                self.calls = []

            def get(self, url, params, timeout):
                self.calls.append(params)
                return FakeResponse()

        session = FakeSession()
        phrase, years = statistics.publication_counts(
            'Enhancing IoT Security Using Machine Learning for Port Scan Attack Detection',
            'IoT networks face port scanning attacks. Port scanning requires early detection.',
            session=session,
        )
        self.assertEqual(phrase, '"Port Scan" AND IoT')
        self.assertEqual([item['count'] for item in years], [27, 19, 0, 4])
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0]['group_by'], 'publication_year')
        self.assertIn('title_and_abstract.search:', session.calls[0]['filter'])
        self.assertNotIn('topics.id:', session.calls[0]['filter'])

    def test_v4_cache_upgrade_reuses_strict_sources_and_signal(self):
        anchor = {'doc_id': 'W1', 'title': 'Optical glucose sensor',
                  'abstract_text': 'Optical glucose measurements.',
                  'primary_topic_id': 'T123'}
        old = {
            'version': 'openalex-related-v4', 'anchor_id': 'W1',
            'anchor_title': anchor['title'], 'topic_id': 'T123',
            'years': [{'year': 2025, 'count': 1, 'exhaustive': True}],
            'sources': [{'doc_id': 'W2', 'title': 'Related work'}],
            'signal_text': 'Две работы показывают конкретный метод.',
            'complete': True,
            'generated_at': evidence.dt.datetime.now(evidence.dt.timezone.utc).isoformat(),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'cached.json'
            path.write_text(json.dumps(old), encoding='utf-8')
            with patch.object(evidence, 'cache_path', return_value=path), \
                 patch.object(evidence, 'publication_counts', return_value=(
                     '"optical glucose"',
                     [{'year': 2025, 'count': 25, 'exhaustive': True}],
                 )) as counts:
                upgraded = evidence._upgrade_cached_statistics('test', anchor)
        counts.assert_called_once()
        self.assertEqual(upgraded['years'][0]['count'], 25)
        self.assertEqual(upgraded['strict_years'][0]['count'], 1)
        self.assertEqual(upgraded['sources'], old['sources'])
        self.assertEqual(upgraded['signal_text'], old['signal_text'])
        self.assertEqual(upgraded['version'], evidence.EVIDENCE_VERSION)

    def test_application_qualifier_is_kept_for_matching(self):
        self.assertEqual(
            evidence._focus_terms('Fiber sensors with emphasis on biomedical applications: a review'),
            {'biomedical'},
        )

    def test_fetch_year_marks_truncated_results_as_incomplete(self):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {'results': [{'id': 'https://openalex.org/W1'}],
                        'meta': {'count': 3, 'next_cursor': None}}

        class FakeSession:
            def get(self, url, params, timeout):
                self.params = params
                return FakeResponse()

        session = FakeSession()
        works, count, exhaustive = evidence._fetch_year('fiber sensor', 'T123', 2025, session)
        self.assertEqual((len(works), count, exhaustive), (1, 3, False))
        self.assertIn('topics.id:T123', session.params['filter'])

    def test_build_evidence_filters_unrelated_works_and_anchor(self):
        anchor = {'doc_id': 'W1', 'title': 'Photonic crystal fiber biomedical sensors',
                  'abstract_text': 'Optical glucose detection.', 'primary_topic_id': 'T123'}

        def work(doc_id, title):
            return {'id': 'https://openalex.org/' + doc_id, 'title': title,
                    'doi': 'https://doi.org/10.1/' + doc_id,
                    'abstract_inverted_index': {}, 'type': 'article'}

        pages = {
            2026: ([work('W1', anchor['title']),
                    work('W2', 'Photonic crystal fiber biomedical sensors for glucose'),
                    work('W3', 'Photonic crystal fiber biomedical sensors overview')], 3, True),
            2025: ([work('W4', 'Photonic crystal fiber biomedical sensors in diagnosis')], 4, False),
            2024: ([], 0, True),
            2023: ([], 0, True),
        }

        class FakeEncoder:
            def encode(self, texts, **kwargs):
                return np.asarray([
                    [0.1, 0.995] if 'overview' in text else [1.0, 0.0]
                    for text in texts
                ], dtype=np.float32)

        broad_years = [
            {'year': year, 'count': count, 'exhaustive': True}
            for year, count in zip(evidence.YEARS, (30, 25, 20, 15))
        ]
        with patch.object(evidence, 'publication_counts', return_value=('"photonic crystal"', broad_years)), \
             patch.object(evidence, '_fetch_year', side_effect=lambda phrase, topic, year, session: pages[year]):
            result = evidence.build_evidence(anchor, session=object(), encoder=FakeEncoder())

        self.assertEqual([year['count'] for year in result['years']], [30, 25, 20, 15])
        self.assertEqual([year['count'] for year in result['strict_years']], [2, 1, 0, 0])
        self.assertTrue(result['complete'])
        self.assertFalse(result['strict_complete'])
        self.assertEqual([source['doc_id'] for source in result['sources']], ['W2', 'W4'])
