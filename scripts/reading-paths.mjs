export function readingPaths(index, englishIndex) {
  const recent = index.filter(p => p.section === 'posts').slice(0, 5).map(p => p.permalink);
  const series = index.find(p => p.categories.includes('code-series'))?.permalink;
  const originals = [...new Set([...recent, ...(series ? [series] : [])])];
  const available = new Set(englishIndex.map(p => p.permalink));
  const translations = originals.map(p => `/en${p}`).filter(p => available.has(p));
  return { recent, series, paths: ['/', '/en/', ...originals, ...translations] };
}
