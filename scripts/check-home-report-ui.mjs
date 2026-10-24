// Dependency-free smoke checks against a running Hugo build. Requires Chrome and Node 22+.
// Usage: node scripts/check-home-report-ui.mjs http://127.0.0.1:8765 [screenshot-directory]
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';

const base = process.argv[2] || 'http://127.0.0.1:8765';
const screenshots = process.argv[3];
const profile = await mkdtemp(join(tmpdir(), 'blog-reading-check-'));
const chrome = spawn(process.env.CHROME_BIN || 'google-chrome', [
  '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = '';
    const timeout = setTimeout(() => reject(new Error('Chrome startup timed out')), 20000);
    chrome.once('error', error => { clearTimeout(timeout); reject(error); });
    chrome.stderr.on('data', chunk => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) { clearTimeout(timeout); resolve(match[1]); }
    });
    chrome.once('exit', code => { clearTimeout(timeout); reject(new Error(`Chrome exited: ${code}`)); });
  });
  const port = new URL(endpoint).port;
  const target = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' }).then(r => r.json());
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let sequence = 0;
  const pending = new Map();
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    const request = pending.get(message.id);
    if (!request) return;
    clearTimeout(request.timeout);
    pending.delete(message.id);
    if (message.error) request.reject(new Error(JSON.stringify(message.error)));
    else request.resolve(message.result);
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 30000);
    pending.set(id, { resolve, reject, timeout });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await send('Page.enable');

  const index = await fetch(new URL('/index.json', base)).then(r => r.json());
  const reports = ['일간', '주간', '월간'].map(kind => index.find(p => p.title.startsWith('서울·경기 아파트') && p.title.includes(kind)));
  assert(reports.every(Boolean), 'Expected daily, weekly, monthly report pages');
  if (screenshots) await mkdir(screenshots, { recursive: true });
  for (const dark of [false, true]) {
    for (const width of [360, 1440]) {
      await send('Emulation.setDeviceMetricsOverride', {width, height:1000, deviceScaleFactor:1, mobile:width < 600});
      for (const [number, report] of reports.entries()) {
        const url = new URL(report.permalink, base).href;
        await send('Page.navigate', {url});
        for (let i=0;i<100;i++) {
          if(await evaluate(`document.readyState==='complete' && location.href===${JSON.stringify(url)}`)) break;
          await pause(150);
        }
        await evaluate(`document.documentElement.classList.toggle('dark',${dark}); document.fonts.ready.then(()=>true)`);
        await pause(150);
        const result = await evaluate(`(() => {
          const article=document.querySelector('article.home-report');
          const citations=article?[...article.querySelectorAll('a[href^="#fn:"]')]:[];
          const text=article?.textContent || '';
          const rows=article?[...article.querySelectorAll('.report-price-row')]:[];
          const samples=rows.map(e=>e.textContent.match(/거래 (\\d+) → (\\d+)건/)).filter(Boolean).map(m=>[+m[1],+m[2]]);
          // Research is optional and selected per report; validate every displayed attribution.
          const researchSources=[];
          const researchHeading=article?[...article.querySelectorAll('h2')].find(h=>h.textContent.includes('참고 리포트와 해석의 범위')):null;
          for(let node=researchHeading?.nextElementSibling;node && !['H2','HR'].includes(node.tagName);node=node.nextElementSibling){
            if(node.tagName!=='H3')continue;
            const attribution=node.nextElementSibling;
            const source=attribution?[...attribution.querySelectorAll('a[href]')].find(a=>a.protocol==='https:'):null;
            const citation=attribution?.querySelector('a[href^="#fn:"]');
            const note=citation?document.getElementById(decodeURIComponent(citation.hash.slice(1))):null;
            researchSources.push({organization:node.textContent.trim(),
              valid:!!source && !!note && [...note.querySelectorAll('a[href]')].some(a=>a.href===source.href)
                && attribution.textContent.split('·').length>=3 && /[0-9]{4}-[0-9]{2}-[0-9]{2}/.test(attribution.textContent)});
          }
          const clips=article?[...article.querySelectorAll('.report-kpi,.report-price-row,.report-volume-row,.report-history-row')].filter(e=>e.getBoundingClientRect().right>innerWidth+1 || e.getBoundingClientRect().left<0).length:99;
          function rgb(color){const c=document.createElement('canvas');c.width=c.height=1;const x=c.getContext('2d');x.fillStyle=color;x.fillRect(0,0,1,1);return [...x.getImageData(0,0,1,1).data].slice(0,3);}
          function lum(v){return v.map(n=>{n/=255;return n<=.04045?n/12.92:Math.pow((n+.055)/1.055,2.4)}).reduce((a,v,i)=>a+v*[.2126,.7152,.0722][i],0);}
          const contrast=article?[...article.querySelectorAll('.report-up,.report-down,.report-kpi-label,.report-kpi-value')].map(e=>{let p=e,bg='transparent';while(p){bg=getComputedStyle(p).backgroundColor;if(!['rgba(0, 0, 0, 0)','transparent'].includes(bg))break;p=p.parentElement;}const f=lum(rgb(getComputedStyle(e).color)),b=lum(rgb(bg));return (Math.max(f,b)+.05)/(Math.min(f,b)+.05);}):[];
          return {article:!!article,kpis:article?.querySelectorAll('.report-kpi').length, priceRows:rows.length, samples,
            volumeRows:article?.querySelectorAll('.report-volume-row').length, historyRows:article?.querySelectorAll('.report-history-row').length,clipped:clips,
            overflow:document.documentElement.scrollWidth>innerWidth+1,
            badReferences:citations.filter(a=>!document.getElementById(decodeURIComponent(a.hash.slice(1)))).map(a=>a.hash),
            rawFootnotes:/\\[\\^\\d+\\]/.test(text),
            researchSources,
            minimumContrast:Math.min(...contrast),
            priceTop:article?.querySelector('.report-price-list')?.getBoundingClientRect().top+scrollY};
        })()`);
        assert(result.article && result.kpis>=1 && result.kpis<=3, `${report.title}: expected report overview`);
        assert(!result.overflow && !result.clipped, `${report.title} ${width}: page or component overflow`);
        assert(result.priceRows>=0 && result.priceRows<=4, 'Primary price cards must be readable and bounded');
        assert(result.samples.length===result.priceRows && result.samples.every(v=>Math.min(...v)>=5), 'Price cards need ample sample counts');
        assert(result.volumeRows>=0 && result.volumeRows<=8, 'Volume selection');
        assert(result.historyRows>=0 && result.historyRows<=24, 'Monthly history must stay bounded');
        assert.deepEqual(result.badReferences, [], 'All HTML citation anchors must resolve');
        assert(!result.rawFootnotes, 'Footnotes must be rendered');
        assert(result.researchSources.every(source=>source.valid), 'Displayed research needs author, date, original URL and matching citation');
        assert(result.minimumContrast>=4.5, `Report contrast ${result.minimumContrast.toFixed(2)}`);
        if(screenshots){
          const stem=`${['daily','weekly','monthly'][number]}-${width}-${dark?'dark':'light'}`;
          const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:1000,scale:1}});
          await writeFile(join(screenshots,stem+'.png'),Buffer.from(shot.data,'base64'));
          if(number===1){const price=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:Math.max(0,result.priceTop-110),width,height:1000,scale:1}});await writeFile(join(screenshots,stem+'-prices.png'),Buffer.from(price.data,'base64'));}
        }
        if (await evaluate(`!!document.querySelector('.report-background summary')`)) {
          assert(!await evaluate(`document.querySelector('.report-background').open`),'Daily historical background starts folded');
          await evaluate(`document.querySelector('.report-background summary').focus()`);
          await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
          await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
          await pause(100);
          assert(await evaluate(`document.querySelector('.report-background').open`),'Historical background must open with keyboard');
          assert(!await evaluate(`document.documentElement.scrollWidth>innerWidth+1`),'Expanded background must not overflow');
        }
        if (await evaluate(`!!document.querySelector('.report-data-table summary')`)) {
        await evaluate(`document.querySelector('.report-data-table summary').focus()`);
        await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
        await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
        await pause(100);
        assert(await evaluate(`document.querySelector('.report-data-table').open`),'Data tables must open with keyboard');
        }
        console.log(JSON.stringify({title:report.title,width,dark,...result}));
      }
    }
  }
} finally {
  socket?.close();
  chrome.kill('SIGTERM');
  await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }).catch(() => {});
}
