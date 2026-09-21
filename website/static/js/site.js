document.addEventListener('DOMContentLoaded', () => {
  const search = document.querySelector('.search-form input');
  if (search) {
    search.addEventListener('focus', () => search.select());
  }
});
