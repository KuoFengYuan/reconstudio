// One editor at a time. Remote decode runs in the editor's transferable Worker.
(function () {
  'use strict';
  let request = null;
  let generation = 0;
  let metadata = null;
  let ready = false;

  function stop() {
    generation++;
    if (request) request.abort();
    request = null;
    const frame = document.getElementById('ss-frame');
    if (frame) frame.remove(); // unload also terminates its decoding Worker
    ready = false;
  }
  function button(text, callback, className = 'ghost') {
    const element = document.createElement('button');
    element.type = 'button'; element.className = className;
    element.textContent = text; element.onclick = callback;
    return element;
  }
  function shell(jobId) {
    const main = document.getElementById('main');
    main.replaceChildren();
    const bar = document.createElement('div'); bar.className = 'ss-toolbar';
    bar.append(button('← 返回任務', () => { stop(); window.loadJob(jobId); }));
    const cancel = button('停止載入／關閉模型', () => {
      stop(); window.loadJob(jobId);
    });
    bar.append(cancel);
    const label = document.createElement('label'); label.textContent = '編輯畫質';
    const select = document.createElement('select'); select.id = 'ss-quality';
    for (const [value, text] of [['performance','流暢（大場景）'],['balanced','均衡'],['native','原始解析度']]) {
      const option = document.createElement('option');option.value=value;option.textContent=text;select.append(option);
    }
    select.value = metadata && (metadata.large || metadata.size >= 100 * 1048576) ? 'performance' : 'balanced';
    select.disabled = true;
    select.onchange = () => {
      const frame = document.getElementById('ss-frame');
      if (frame) frame.contentWindow.postMessage({type:'reconstudio:quality',quality:select.value}, location.origin);
    };
    label.append(select);bar.append(label);
    const send = button('送回去背點雲', () => window.superSplatSendBack(jobId));
    send.id = 'ss-send';send.disabled = true;bar.append(send);
    main.append(bar);
    const status = document.createElement('div');status.className='ss-load-status';status.setAttribute('role','status');
    status.id='ss-load-status';main.append(status);
    const exportStatus = document.createElement('div');exportStatus.id='ss-status';exportStatus.className='hint';main.append(exportStatus);
    return main;
  }
  function status(message, progress) {
    const box = document.getElementById('ss-load-status');
    if (!box) return;
    box.replaceChildren();
    const text = document.createElement('span');text.textContent=message;box.append(text);
    if (!ready) {
      const meter = document.createElement('progress');meter.max=100;
      meter.setAttribute('aria-label','模型載入進度');
      if (Number.isFinite(progress)) meter.value=progress;
      box.append(meter);
    }
  }
  window.openSuperSplat = async function(jobId) {
    stop(); const token=generation;metadata=null;
    if (window.rsBus) { window.rsBus.unwatchLog();window._logJobId=null; }
    window.showTab('run');shell(jobId);status('正在檢查模型…');
    if (window.revealWorkspaceContent) window.revealWorkspaceContent();
    request = new AbortController();
    try {
      const response=await fetch('/api/jobs/'+encodeURIComponent(jobId)+'/splat_info',{signal:request.signal});
      if (!response.ok) throw new Error('無法取得模型資訊（HTTP '+response.status+'）');
      const info=await response.json();
      if(token!==generation)return;
      metadata=info;window.mountSuperSplat(jobId,info.filename);
    } catch(error) {
      if(token!==generation || error.name==='AbortError')return;
      ready=true;status(error.message);ready=false;
    } finally { if(token===generation)request=null; }
  };
  window.mountSuperSplat = function(jobId,filename) {
    const main=shell(jobId);
    const source=new URL('/api/jobs/'+encodeURIComponent(jobId)+'/splat/'+encodeURIComponent(filename),location.origin);
    if(metadata && metadata.revision)source.searchParams.set('v',metadata.revision);
    const url=new URL('/static/supersplat/index.html',location.origin);
    url.searchParams.set('load',source.href);
    url.searchParams.set('filename',filename);
    url.searchParams.set('rsQuality',document.getElementById('ss-quality').value);
    // Version the entry URL too: older installed service workers must not mask a patched bundle.
    url.searchParams.set('rsBuild','worker-v1');
    const details=metadata ? (metadata.size/1048576).toFixed(1)+' MB'+(metadata.count?' · '+metadata.count.toLocaleString()+' splats':'') : filename;
    status(details+' · 正在啟動編輯器…');
    const hint=document.createElement('p');hint.className='hint';
    hint.textContent='流暢模式僅降低編輯畫面的解析度，完整點數與模型資料保留。框選 → Ctrl+I 反選 → Delete 去背景 → 送回。';
    main.append(hint);
    const frame=document.createElement('iframe');frame.id='ss-frame';frame.title='SuperSplat 點雲編輯器';frame.src=url.href;
    main.append(frame);
    if (window.revealWorkspaceContent) window.revealWorkspaceContent();
  };
  window.addEventListener('message', event => {
    const frame=document.getElementById('ss-frame');
    if(!frame || event.source!==frame.contentWindow || event.origin!==location.origin)return;
    const data=event.data;
    if(!data || data.type!=='reconstudio:splat-load')return;
    if(data.phase==='loading') {
      const percent=data.total && Number.isFinite(data.loaded) ? Math.min(99,100*data.loaded/data.total):undefined;
      status(percent===undefined?'背景讀取與解析模型…':'背景讀取與解析模型 · '+percent.toFixed(0)+'%',percent);
    } else if(data.phase==='packing') {
      status('背景整理 GPU 紋理 · 完整保留球諧與幾何資料…');
    } else if(data.phase==='gpu') {
      status('解析完成 · 正在建立 GPU 資料（'+Number(data.count).toLocaleString()+' splats）…');
    } else if(data.phase==='ready') {
      ready=true;frame.dataset.ready='true';document.getElementById('ss-send').disabled=false;document.getElementById('ss-quality').disabled=false;
      status('已載入 '+Number(data.count).toLocaleString()+' splats · 可以開始編輯');
    } else if(data.phase==='error') {
      ready=true;status('載入失敗：'+String(data.message));ready=false;
    }
  });
  // Leaving the editor must invalidate a still-pending metadata request, too.
  document.body.addEventListener('htmx:beforeSwap', event => {
    if(event.detail.target.id==='main') {stop();}
  });
})();
