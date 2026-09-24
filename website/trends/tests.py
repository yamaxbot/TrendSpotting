from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import numpy as np
from django.test import SimpleTestCase

from . import jobs, views


class WebsiteTests(SimpleTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.corpus = root / 'corpus.parquet'
        self.dataset = root / 'dataset.parquet'
        pd.DataFrame([{'doc_id': 'W1', 'pub_year': 2025, 'title': 'Original',
                       'abstract_text': 'Text', 'article_url': 'javascript:alert(1)',
                       'doi': 'https://doi.org/10.1/example'}]).to_parquet(self.corpus)
        pd.DataFrame([{'doc_id': 'W1', 'model_confidence': 91., 'shap_values': '{}',
                       'diversity_max_similarity': 0.0,
                       'llm_title': 'Translation', 'llm_title_version': 'exact-v1',
                       'llm_description': 'Summary',
                       'llm_weak_signal': 'Explanation'}]).to_parquet(self.dataset)
        for name, value in [('get_built_dataset_path', self.dataset),
                            ('get_parsed_corpus_path', self.corpus)]:
            patcher = patch('website.main.' + name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_cached_results_and_details_do_not_start_work(self):
        with patch.object(jobs, 'start') as start:
            response = self.client.get('/', {'q': 'test'})
            self.assertContains(response, 'Translation')
            self.assertFalse(response.context['pending'])
            detail = self.client.get('/trend/result-1/', {'q': 'test'})
            self.assertContains(detail, 'https://doi.org/10.1/example')
            self.assertNotContains(detail, 'javascript:')
            self.assertEqual(self.client.get('/trend/unknown/', {'q': 'test'}).status_code, 404)
            start.assert_not_called()

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

    def test_translated_title_is_exact_and_uses_no_abstract(self):
        from website import query_llm

        translated = 'Перевод: Точное переведённое название без сокращений'
        with patch.object(query_llm, '_query_llm', return_value=translated) as query:
            result = query_llm.translate_article_title('Original', 'Abstract')

        self.assertEqual(result, 'Точное переведённое название без сокращений')
        self.assertEqual(query.call_args.args[1], 'Original')
        self.assertNotIn('Abstract', query.call_args.args[1])
        self.assertEqual(query.call_args.kwargs['temperature'], 0.0)

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
