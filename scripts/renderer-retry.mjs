import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

// Dedicated exit status for CDN fetch failures during renderer initialization.
const FETCH_FAILURE = 75;

export function rendererLoadExitCode(error, loadingRenderer) {
  const message = String(error);
  const fetchFailure = /Failed to fetch dynamically imported module:\s*https:\/\/cdn\.jsdelivr\.net\/npm\/mermaid@/.test(message)
    || /Failed to load https:\/\/cdn\.jsdelivr\.net\/npm\/katex@/.test(message);
  return loadingRenderer && fetchFailure ? FETCH_FAILURE : 1;
}

export async function retryRenderer(run, {
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
  log = message => console.error(message),
} = {}) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    const code = await run();
    if (code !== FETCH_FAILURE || attempt === 3) return code;
    log(`Renderer CDN fetch failed; retrying in ${attempt}s (attempt ${attempt + 1}/3).`);
    await sleep(attempt * 1000);
  }
}

// Each retry starts a fresh Chrome process/profile. Failed ES module imports are
// cached in a document's module map, so repeating import() there cannot recover.
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  if (!args.length) {
    console.error('Usage: node scripts/renderer-retry.mjs <checker.mjs> [build-directory]');
    process.exitCode = 2;
  } else {
    process.exitCode = await retryRenderer(() => new Promise((resolve, reject) => {
      const child = spawn(process.execPath, args, { stdio: 'inherit' });
      child.once('error', reject);
      child.once('exit', code => resolve(code ?? 1));
    }));
  }
}
