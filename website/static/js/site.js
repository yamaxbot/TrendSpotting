document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('.search-form');
  const search = form?.querySelector('input');
  const submitButton = form?.querySelector('button[type="submit"]');
  const loadingOverlay = document.querySelector('[data-loading-overlay]');
  const statusUrl = loadingOverlay?.dataset.statusUrl;
  let timer;
  let stopped = false;
  let submitting = false;

  const poll = async () => {
    if (stopped) return;
    try {
      const response = await fetch(statusUrl, { cache: 'no-store', signal: AbortSignal.timeout(15000) });
      if (!response.ok) throw new Error('status unavailable');
      const result = await response.json();
      document.getElementById('search-error').textContent = '';
      if (result.status === 'complete') {
        stopped = true;
        window.location.reload();
        return;
      }
      if (['failed', 'missing', 'busy'].includes(result.status)) {
        stopped = true;
        loadingOverlay.hidden = true;
        document.getElementById('search-error').textContent =
          'Не удалось завершить поиск. Нажмите «Найти», чтобы повторить.';
        return;
      }
    } catch (error) {
      document.getElementById('search-error').textContent =
        'Нет связи с сервером. Проверяем соединение…';
    }
    if (!stopped) timer = window.setTimeout(poll, 3000);
  };

  const resetLoadingState = () => {
    if (loadingOverlay) {
      loadingOverlay.hidden = !statusUrl || stopped;
    }
    if (submitButton) {
      submitButton.disabled = false;
    }
  };

  resetLoadingState();
  if (statusUrl) poll();

  if (search) {
    search.addEventListener('focus', () => search.select());
  }

  if (form && search && loadingOverlay) {
    form.addEventListener('submit', (event) => {
      if (submitting) {
        event.preventDefault();
        return;
      }
      if (!search.value.trim()) {
        event.preventDefault();
        search.focus();
        return;
      }

      event.preventDefault();
      submitting = true;
      stopped = true;
      window.clearTimeout(timer);
      loadingOverlay.hidden = false;
      if (submitButton) {
        submitButton.disabled = true;
      }

      window.setTimeout(() => {
        form.submit();
      }, 100);
    });
  }

  window.addEventListener('pageshow', (event) => {
    submitting = false;
    resetLoadingState();
    if (event.persisted && statusUrl) {
      stopped = false;
      loadingOverlay.hidden = false;
      window.clearTimeout(timer);
      poll();
    }
  });

  const evidenceNotice = document.querySelector('[data-evidence-status-url]');
  const evidenceUrl = evidenceNotice?.dataset.evidenceStatusUrl;
  if (evidenceUrl) {
    const pollEvidence = async () => {
      try {
        const response = await fetch(evidenceUrl, {
          cache: 'no-store', signal: AbortSignal.timeout(15000),
        });
        if (!response.ok) throw new Error('evidence status unavailable');
        const result = await response.json();
        if (result.status === 'complete') {
          window.location.reload();
          return;
        }
        if (result.status === 'failed') {
          evidenceNotice.textContent = 'Не удалось получить статистику OpenAlex. Обновите страницу, чтобы повторить поиск.';
          return;
        }
      } catch (error) {
        evidenceNotice.textContent = 'Проверяем связь с OpenAlex…';
      }
      window.setTimeout(pollEvidence, 3000);
    };
    window.setTimeout(pollEvidence, 3000);
  }

  window.addEventListener('pagehide', () => {
    stopped = true;
    window.clearTimeout(timer);
  });
});
