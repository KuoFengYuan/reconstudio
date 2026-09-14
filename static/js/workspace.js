// Workspace guidance, accessible dialogs and request feedback. No pipeline state.
(function () {
  'use strict';
  const descriptions = {
    gcs: ['雲端資料', '下載或上傳素材，再到需要的功能選取本機資料夾。'],
    frames: ['影片抽幀', '選擇影片與輸出位置，保留清晰影格作為重建素材。'],
    colmap: ['照片重建', '選擇照片與工作目錄，計算相機位置並建立稀疏點雲。'],
    fusion: ['多視角融合', '將不同方向拍攝的素材與遮罩整合，準備共同重建。'],
    train: ['3DGS 訓練', '使用 COLMAP 資料集訓練模型；選擇後端後可調整細部參數。'],
    mesh: ['Mesh 輸出', '從已訓練模型抽取三角網格，供檢視、量測或匯出使用。'],
    depth: ['深度與法線', '為照片產生深度或法線圖，提供訓練時的幾何資訊。'],
    matte: ['影像去背', '先選擇照片資料夾，再框選或點選要保留的物體。'],
    blocksplit: ['場景分塊', '將大型場景分成較小的訓練區域，也可只裁切指定範圍。'],
    gallery: ['圖片檢視', '以縮圖牆瀏覽伺服器上的照片，快速檢查拍攝素材。'],
    viewer: ['Mesh 檢視', '開啟模型檢查外觀與尺寸；也可留空後選擇電腦上的檔案。'],
    measure: ['模型量測', '選擇 Mesh 檔案，檢查模型的幾何尺寸。'],
    supersplat: ['點雲編輯', '開啟 3DGS 或點雲，使用 SuperSplat 檢視與編輯。'],
  };
  function describe(which) {
    const text = descriptions[which];
    if (!text) return;
    document.getElementById('form-title').textContent = text[0];
    document.getElementById('form-description').textContent = text[1];
  }
  document.addEventListener('rs:formchange', (e) => {
    describe(e.detail.which);
    if (document.querySelector('.wrap').classList.contains('nleft')) window.toggleLeftPanel();
  });
  const current = document.querySelector('.tabs.steps button.active');
  describe(current ? current.id.slice(2) : 'gcs');

  window.startWorkflow = function (which) {
    if (document.querySelector('.wrap').classList.contains('nleft')) window.toggleLeftPanel();
    window.showForm(which);
    const field = document.querySelector('#form-' + which + ' input:not([type=hidden]):not([disabled])');
    if (field) field.focus();
  };

  // Give existing standalone labels and inline help controls keyboard semantics.
  document.querySelectorAll('.col.left label:not([for])').forEach((label, i) => {
    if (label.querySelector('input,select,textarea')) return;
    const sibling = label.nextElementSibling;
    const field = sibling && (sibling.matches('input,select,textarea') ? sibling : sibling.querySelector('input,select,textarea'));
    if (!field) return;
    if (!field.id) field.id = 'rs-field-' + i;
    label.htmlFor = field.id;
  });
  document.querySelectorAll('.info[onclick]').forEach((info) => {
    info.tabIndex = 0;
    info.setAttribute('role', 'button');
    info.setAttribute('aria-label', '參數說明');
    info.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); info.click(); }
    });
  });
  document.querySelector('.workspace-tabs').addEventListener('keydown', (e) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    const tabs = Array.from(document.querySelectorAll('.workspace-tabs [role=tab]'));
    const selected = tabs.findIndex(tab => tab.getAttribute('aria-selected') === 'true');
    const index = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 :
      (selected + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    const which = tabs[index].id.slice('view-'.length);
    window.showTab(which);
    document.getElementById('view-' + which).focus();
  });

  let noticeTimer;
  window.notifyUI = function (message, tone = "error") {
    const notice = document.getElementById('ui-notice');
    notice.dataset.tone = tone;
    notice.textContent = message;
    notice.hidden = false;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => { notice.hidden = true; }, 12000);
  };

  // Scope loading state to the submitted form; do not disable unrelated inputs.
  const requests = new WeakMap();
  document.body.addEventListener('htmx:beforeRequest', (e) => {
    const form = e.detail.elt;
    if (!form || !form.matches('.col.left form[hx-post]')) return;
    if (requests.has(form)) { e.preventDefault(); return; }
    const buttons = Array.from(form.querySelectorAll('button[type=submit]')).map((button) =>
      ({ button, text: button.textContent, disabled: button.disabled }));
    requests.set(form, buttons);
    buttons.forEach(({button}) => { button.disabled = true; button.textContent = '正在送出…'; });
    form.setAttribute('aria-busy', 'true');
  });
  document.body.addEventListener('htmx:afterRequest', (e) => {
    const form = e.detail.elt;
    if (!requests.has(form)) return;
    requests.get(form).forEach(({button, text, disabled}) => { button.textContent = text; button.disabled = disabled; });
    requests.delete(form);
    form.removeAttribute('aria-busy');
    if (form.id === 'form-matte' && window.matteGate) window.matteGate();
  });
  for (const event of ['htmx:responseError', 'htmx:sendError', 'htmx:timeout']) {
    document.body.addEventListener(event, (e) => {
      const status = e.detail.xhr && e.detail.xhr.status;
      window.notifyUI(status ? '請求失敗（' + status + '）。請稍後重試，或查看環境檢查。' :
        '暫時無法連線。若剛送出任務，請先查看任務紀錄確認是否已建立。');
    });
  }
  // Replacing a tall job/editor must not inherit a scroll position that hides
  // the new controls above the visible content area.
  window.revealWorkspaceContent = function () {
    const output = document.getElementById('workspace-output');
    if (output.dataset.view !== 'run') return;
    document.getElementById('workspace-content').scrollTop = 0;
    if (window.matchMedia('(max-width:760px)').matches) output.scrollIntoView({block:'start'});
  };

  let selectedJob = new URLSearchParams(location.search).get('job');
  window.syncWorkspaceLocation = function () {
    const url = new URL(location.href);
    if (document.getElementById('workspace-output').dataset.view === 'run' && selectedJob) {
      url.searchParams.set('job', selectedJob);
    } else {
      url.searchParams.delete('job');
    }
    history.replaceState(history.state, '', url);
  };
  // A request may finish after the user has returned home or to the history.
  // Keep its result under Current work, without stealing the selected view.
  const navigationAtRequest = new WeakMap();
  document.body.addEventListener('htmx:beforeRequest', (e) => {
    if (e.detail.target?.id === 'main' && e.detail.xhr) {
      navigationAtRequest.set(e.detail.xhr, window._workspaceNavigation || 0);
    }
  });
  document.body.addEventListener('htmx:afterSwap', (e) => {
    if (e.detail.target.id !== 'main') return;
    const job = e.detail.target.querySelector('[data-current-job]');
    selectedJob = job ? job.dataset.currentJob : null;
    const requestNavigation = navigationAtRequest.get(e.detail.xhr);
    if (requestNavigation !== undefined && requestNavigation === (window._workspaceNavigation || 0)) {
      window.showTab('run');
    }
    window.syncWorkspaceLocation();
    if (document.getElementById('workspace-output').dataset.view !== 'run') return;
    window.revealWorkspaceContent();
    const error = e.detail.target.querySelector('[data-ui-error]');
    if (error) error.focus({preventScroll:true});
    if (window.matchMedia('(max-width:760px)').matches) {
      document.getElementById('workspace-output').scrollIntoView({block:'start'});
    }
  });

  window.copyJobLink = async function () {
    const job = document.querySelector('[data-current-job]');
    if (!job) return;
    const url = new URL(location.pathname, location.origin);
    url.searchParams.set('job', job.dataset.currentJob);
    try {
      await navigator.clipboard.writeText(url.href);
      window.notifyUI('已複製任務連結，可用來返回此任務。', 'success');
    } catch (_) {
      window.prompt('複製此連結即可返回任務：', url.href);
    }
  };
  const linkedJob = new URLSearchParams(location.search).get('job');
  if (linkedJob) window.loadJob(linkedJob);

  // Existing picker entry points all toggle .open; observe this single contract.
  const picker = document.getElementById('picker');
  let returnFocus = null;
  let wasOpen = false;
  const focusable = () => Array.from(picker.querySelectorAll('button, a[href], input, select, textarea, [tabindex="0"]'))
    .filter((el) => !el.disabled && el.getClientRects().length);
  const observer = new MutationObserver(() => {
    const open = picker.classList.contains('open');
    if (open === wasOpen) return;
    wasOpen = open;
    if (open) {
      returnFocus = document.activeElement;
      document.getElementById('picker-title').textContent = window._pickerMode === 'file' ? '選擇檔案' :
        window._pickerMode === 'gcs' ? '選擇雲端資料夾' : '選擇資料夾';
      document.querySelector('header').inert = true;
      document.querySelector('.wrap').inert = true;
      (focusable()[0] || picker).focus();
    } else {
      document.querySelector('header').inert = false;
      document.querySelector('.wrap').inert = false;
      if (returnFocus && returnFocus.isConnected) returnFocus.focus({preventScroll:true});
    }
  });
  observer.observe(picker, {attributes:true, attributeFilter:['class']});
  picker.addEventListener('click', (e) => { if (e.target === picker) picker.classList.remove('open'); });
  picker.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); picker.classList.remove('open'); }
    if (e.key !== 'Tab') return;
    const items = focusable();
    const first = items[0], last = items[items.length - 1];
    if (!first) { e.preventDefault(); picker.focus(); }
    else if (e.shiftKey && (document.activeElement === first || document.activeElement === picker)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
})();
