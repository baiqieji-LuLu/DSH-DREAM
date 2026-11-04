/**
 * DSH-DREAM settings page for the DSH Web client.
 *
 * The page is a plain-JS Client plugin: it registers a `settings.section`
 * contribution while the Host serves the `dsh-dream-settings` entry, and
 * reads/writes that entry through the shared `configForms` service. No build
 * step and no import of another Harness Client package is required.
 */
window.__ModuleLoader__.load({
  id: 'dsh-dream',
  factory(require) {
    const React = require('react');
    const h = React.createElement;

    const NS = 'settings.agentMemory';
    const ENTRY_ID = 'dsh-dream-settings';
    const inject = ['slots', 'locale', 'configForms'];

    const en = {
      nav: 'DSH-DREAM Long-Term Memory',
      title: 'DSH-DREAM Long-Term Memory',
      description:
        'Configure persistent long-term memory for the DSH-DREAM Python package.',
      enable: 'Enable long-term memory',
      enableHint:
        'When off, memory_search returns no results without opening Chroma.',
      maxWindow: 'Window token budget',
      maxWindowHint:
        'Token budget used to initialize WindowManager (128–32768).',
      topK: 'Retrieval top-k',
      topKHint:
        'Number of RAG memories returned by memory_search (1–20).',
      ragPath: 'Chroma persistence path',
      ragPathHint:
        "Leave blank to use the DSH user directory; enter an absolute path to customize, or set AGENT_MEMORY_RAG_PATH to override.",
      embeddingTitle: 'Embedding model (advanced)',
      embeddingWarning:
        '⚠️ 警告：更换 Embedding 模型会改变向量维度！修改此项后，系统将尝试无损重嵌入旧数据（或要求您手动清空旧 RAG 数据库），否则新记忆将无法与旧记忆一起被检索！',
      embeddingProvider: 'Embedding provider',
      embeddingProviderHint:
        'local uses the bundled BAAI Chinese model; openai_compatible calls a cloud /embeddings API.',
      embeddingModel: 'Embedding model',
      embeddingModelHint:
        'Cloud model name. Leave blank for local BAAI or cloud default text-embedding-3-small.',
      embeddingApiKey: 'Embedding API key',
      embeddingApiKeyHint:
        'Bearer token for the cloud endpoint. Stored locally and masked in this panel.',
      embeddingBaseUrl: 'Embedding base URL',
      embeddingBaseUrlHint:
        'For example https://api.deepseek.com/v1 (confirm it matches your API key).',
      autoRebuild:
        'On dimension mismatch, try lossless re-embedding of old data (cloud APIs may consume tokens)',
      autoRebuildHint:
        'When checked, old documents are cached and re-embedded with the new model instead of being discarded.',
      holographic: 'Holographic mode (forced full memory)',
      holographicHint:
        'When enabled, every turn writes the complete User and Assistant originals to the vector store, no matter what the conversation is about. Nothing is omitted.',
      holographicBadge: 'FORCE MEMORY',
      providerLocal: 'local (bundled Chinese model)',
      providerCloud: 'openai_compatible (cloud API)',
      invalidApiKey: 'Cloud embedding requires an API key.',
      invalidBaseUrl:
        'Cloud embedding requires an http(s) base URL.',
      save: 'Save',
      saving: 'Saving…',
      discard: 'Discard',
      saved: 'Saved',
      unsaved: 'Unsaved changes',
      readOnly: 'This deployment serves settings read-only.',
      unavailable:
        'DSH-DREAM is not loaded, so its settings are unavailable right now.',
      loading: 'Loading settings…',
      saveFailed:
        'The deployment did not accept these values. Your draft was kept.',
      invalidMax:
        'Window token budget must be an integer between 128 and 32768.',
      invalidTopK: 'Retrieval top-k must be an integer between 1 and 20.',
    };

    const zh = {
      nav: 'DSH-DREAM 长期记忆',
      title: 'DSH-DREAM 长期记忆',
      description: '配置 DSH-DREAM Python 包的持久化长期记忆。',
      enable: '启用长期记忆',
      enableHint: '关闭后 memory_search 不会打开 Chroma，直接返回空结果。',
      maxWindow: '窗口 Token 预算',
      maxWindowHint: '用于初始化 WindowManager 的 Token 预算（128–32768）。',
      topK: 'RAG 召回数量',
      topKHint: 'memory_search 返回的记忆条数（1–20）。',
      ragPath: 'Chroma 持久化路径',
      ragPathHint:
        '留空则默认存储在 DSH 用户目录下；如需自定义，请填入绝对路径，或通过环境变量 AGENT_MEMORY_RAG_PATH 覆盖。',
      embeddingTitle: 'Embedding 模型（高阶）',
      embeddingWarning:
        '⚠️ 警告：更换 Embedding 模型会改变向量维度！修改此项后，系统将尝试无损重嵌入旧数据（或要求您手动清空旧 RAG 数据库），否则新记忆将无法与旧记忆一起被检索！',
      embeddingProvider: 'Embedding 提供方',
      embeddingProviderHint:
        'local 使用内置 BAAI 中文模型；openai_compatible 调用云端 /embeddings API。',
      embeddingModel: 'Embedding 模型',
      embeddingModelHint:
        '云端模型名。留空时 local 使用 BAAI，云端使用 text-embedding-3-small。',
      embeddingApiKey: 'Embedding API Key',
      embeddingApiKeyHint:
        '云端接口的 Bearer Token。保存在本机桥接配置中，面板输入框会脱敏显示。',
      embeddingBaseUrl: 'Embedding Base URL',
      embeddingBaseUrlHint:
        '例如 https://api.deepseek.com/v1（请确认与您的 API Key 匹配）。',
      autoRebuild:
        '维度不匹配时，尝试无损重嵌入旧数据（注：若使用云端 API 会产生 Token 消耗）',
      autoRebuildHint:
        '勾选后，旧文档会先缓存，再用新模型重新计算向量写回，而不是直接删除。',
      holographic: '全息模式（强制全量记忆）',
      holographicHint:
        '开启后，无论对话内容如何，每轮对话的 User 和 Assistant 原话都会被强制写入向量库，绝不遗漏。',
      holographicBadge: '强制记忆',
      providerLocal: 'local（内置中文模型）',
      providerCloud: 'openai_compatible（云端 API）',
      invalidApiKey: '云端 Embedding 必须填写 API Key。',
      invalidBaseUrl: '云端 Embedding 必须填写 http(s) Base URL。',
      save: '保存',
      saving: '保存中…',
      discard: '放弃修改',
      saved: '已保存',
      unsaved: '有未保存的修改',
      readOnly: '当前部署的设置为只读。',
      unavailable: 'DSH-DREAM 当前未加载，暂时无法配置。',
      loading: '正在加载设置…',
      saveFailed: '当前部署没有接受这些值，草稿已保留。',
      invalidMax: '窗口 Token 预算必须是 128–32768 之间的整数。',
      invalidTopK: 'RAG 召回数量必须是 1–20 之间的整数。',
    };

    const CSS = `
      .am-page {
        box-sizing: border-box;
        max-width: 760px;
        padding: 8px 0 32px;
        color: var(--dsw-alias-label-primary);
        font-size: 13px;
        line-height: 22px;
      }
      .am-head { margin-bottom: 20px; }
      .am-title {
        margin: 0 0 6px;
        font-size: 20px;
        line-height: 28px;
        font-weight: 600;
      }
      .am-desc {
        max-width: 620px;
        margin: 0;
        color: var(--dsw-alias-label-secondary);
      }
      .am-notice {
        margin: 0 0 12px;
        padding: 8px 12px;
        border: 0.5px solid var(--dsw-alias-border-l2);
        border-radius: var(--dsw-radius-md, 8px);
        color: var(--dsw-alias-label-secondary);
        background: var(--dsw-alias-settings-card-fill, var(--dsw-alias-bg-base));
      }
      .am-card {
        overflow: hidden;
        border: 0.5px solid var(--dsw-alias-settings-card-stroke, var(--dsw-alias-border-l2));
        border-radius: var(--dsw-radius-xl, 12px);
        background: var(--dsw-alias-settings-card-fill, var(--dsw-alias-bg-base));
      }
      .am-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        padding: 16px;
        border-bottom: 0.5px solid var(--dsw-alias-border-l2);
      }
      .am-row-last { border-bottom: 0; }
      .am-copy { min-width: 0; }
      .am-label {
        color: var(--dsw-alias-label-primary);
        font-weight: 500;
      }
      .am-hint {
        margin-top: 2px;
        color: var(--dsw-alias-label-tertiary);
        font-size: 12px;
        line-height: 18px;
      }
      .am-input {
        box-sizing: border-box;
        width: 180px;
        height: 36px;
        padding: 0 10px;
        border: 0.5px solid var(--dsw-alias-border-l3);
        border-radius: var(--dsw-radius-md, 8px);
        outline: none;
        background: var(--dsw-alias-bg-base);
        color: var(--dsw-alias-label-primary);
        font: inherit;
      }
      .am-input:focus-visible {
        outline: 2px solid var(--dsw-focus-ring-color, var(--dsw-alias-state-business-primary));
        outline-offset: 1px;
      }
      .am-input:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      .am-input-readonly {
        width: 280px;
        background: transparent;
        color: var(--dsw-alias-label-secondary);
        cursor: text;
      }
      .am-toggle {
        position: relative;
        flex: none;
        width: 44px;
        height: 24px;
        padding: 0;
        border: 0;
        border-radius: 999px;
        background: var(--dsw-alias-border-l3);
        cursor: pointer;
        transition: background 0.16s ease;
      }
      .am-toggle-on { background: var(--dsw-alias-state-business-primary); }
      .am-toggle:disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }
      .am-toggle:focus-visible {
        outline: 2px solid var(--dsw-focus-ring-color, var(--dsw-alias-state-business-primary));
        outline-offset: 2px;
      }
      .am-knob {
        position: absolute;
        top: 3px;
        left: 3px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #fff;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
        transition: transform 0.16s ease;
      }
      .am-toggle-on .am-knob { transform: translateX(20px); }
      .am-actions {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 10px;
        min-height: 36px;
        margin-top: 16px;
      }
      .am-status {
        flex: 1;
        color: var(--dsw-alias-label-tertiary);
        font-size: 12px;
        text-align: right;
      }
      .am-status:empty { display: none; }
      .am-button {
        height: 36px;
        padding: 0 14px;
        border-radius: var(--dsw-radius-md, 8px);
        color: var(--dsw-alias-label-primary);
        font: inherit;
        cursor: pointer;
      }
      .am-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .am-button:focus-visible {
        outline: 2px solid var(--dsw-focus-ring-color, var(--dsw-alias-state-business-primary));
        outline-offset: 2px;
      }
      .am-button-primary {
        border: 0.5px solid var(--dsw-alias-state-business-primary);
        background: var(--dsw-alias-state-business-primary);
        color: #fff;
      }
      .am-button-secondary {
        border: 0.5px solid var(--dsw-alias-border-l3);
        background: transparent;
        color: var(--dsw-alias-label-primary);
      }
      .am-holo {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin: 0 0 16px;
        padding: 14px 16px;
        border: 1.5px solid var(--dsw-alias-state-error-primary, #f85149);
        border-radius: var(--dsw-radius-xl, 12px);
        background: color-mix(
          in srgb,
          var(--dsw-alias-state-error-primary, #f85149) 8%,
          transparent
        );
      }
      .am-holo-badge {
        display: inline-block;
        margin-left: 8px;
        padding: 1px 6px;
        border-radius: 999px;
        background: var(--dsw-alias-state-error-primary, #f85149);
        color: #fff;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.04em;
        vertical-align: 1px;
      }
      .am-holo .am-label { font-weight: 700; }
      .am-warning {
        margin: 16px 0 10px;
        color: var(--dsw-alias-state-error-primary, #f85149);
        font-size: 13px;
        font-weight: 700;
        line-height: 20px;
      }
      .am-subhead {
        margin: 22px 0 8px;
        color: var(--dsw-alias-label-primary);
        font-size: 15px;
        font-weight: 600;
      }
      .am-select {
        width: 220px;
      }
      .am-check-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
      }
      .am-check-row input[type='checkbox'] {
        width: 16px;
        height: 16px;
        margin-top: 3px;
        accent-color: var(--dsw-alias-state-business-primary);
      }
      .am-muted { color: var(--dsw-alias-label-tertiary); }
      @media (max-width: 560px) {
        .am-row {
          align-items: flex-start;
          flex-direction: column;
          gap: 10px;
        }
        .am-input, .am-input-readonly, .am-select { width: 100%; }
        .am-actions { flex-wrap: wrap; }
        .am-status { flex-basis: 100%; text-align: left; }
      }
    `;

    function asDraft(value) {
      const current = value && typeof value === 'object' ? value : {};
      return {
        enable: current.enable_memory !== false,
        ragPath: String(current.rag_path ?? ''),
        max: String(current.max_window_tokens ?? 1500),
        top: String(current.retrieval_top_k ?? 5),
        embeddingProvider:
          current.embedding_provider === 'openai_compatible'
            ? 'openai_compatible'
            : 'local',
        embeddingModel: String(current.embedding_model ?? ''),
        embeddingApiKey: String(current.embedding_api_key ?? ''),
        embeddingBaseUrl: String(current.embedding_base_url ?? ''),
        autoRebuild: current.auto_rebuild_collection !== false,
        holographic: current.holographic_mode === true,
      };
    }

    function parseDraft(draft) {
      const max = Number(draft.max);
      const top = Number(draft.top);
      if (!Number.isInteger(max) || max < 128 || max > 32768) {
        return { error: 'invalidMax' };
      }
      if (!Number.isInteger(top) || top < 1 || top > 20) {
        return { error: 'invalidTopK' };
      }
      const provider =
        draft.embeddingProvider === 'openai_compatible'
          ? 'openai_compatible'
          : 'local';
      const apiKey = String(draft.embeddingApiKey || '').trim();
      const baseUrl = String(draft.embeddingBaseUrl || '').trim();
      if (provider === 'openai_compatible' && !apiKey) {
        return { error: 'invalidApiKey' };
      }
      if (
        provider === 'openai_compatible' &&
        !/^https?:\/\//i.test(baseUrl)
      ) {
        return { error: 'invalidBaseUrl' };
      }
      return {
        max,
        top,
        provider,
        model: String(draft.embeddingModel || '').trim(),
        apiKey,
        baseUrl,
        autoRebuild: Boolean(draft.autoRebuild),
        holographic: Boolean(draft.holographic),
      };
    }

    function Toggle(props) {
      const { checked, disabled, onChange, label } = props;
      return h(
        'button',
        {
          type: 'button',
          role: 'switch',
          'aria-checked': checked,
          'aria-label': label,
          className: `am-toggle${checked ? ' am-toggle-on' : ''}`,
          disabled,
          onClick: () => onChange(!checked),
        },
        h('span', { className: 'am-knob', 'aria-hidden': true }),
      );
    }

    function PlainNotice(props) {
      return h(
        'section',
        { className: 'am-page', 'aria-label': props.label },
        h('style', null, CSS),
        h('h2', { className: 'am-title' }, props.title),
        h('p', { className: 'am-muted' }, props.message),
      );
    }

    function AgentMemorySection(props) {
      const { t, scope } = props;
      const tr = typeof t === 'function' ? t : (key) => key;
      const snapshot = React.useSyncExternalStore(
        React.useCallback((listener) => scope.subscribe(listener), [scope]),
        React.useCallback(() => scope.getSnapshot(), [scope]),
      );
      const [draft, setDraft] = React.useState(null);
      const [dirty, setDirty] = React.useState(false);
      const [saving, setSaving] = React.useState(false);
      const [saved, setSaved] = React.useState(false);
      const [error, setError] = React.useState('');

      React.useEffect(() => {
        if (snapshot.status === 'ready' && !dirty) {
          setDraft(asDraft(snapshot.value));
        }
      }, [snapshot.status, snapshot.revision, dirty]);

      React.useEffect(() => {
        if (!saved) return undefined;
        const timer = setTimeout(() => setSaved(false), 2500);
        return () => clearTimeout(timer);
      }, [saved]);

      if (snapshot.status === 'loading') {
        return h(PlainNotice, {
          label: tr('nav'),
          title: tr('title'),
          message: tr('loading'),
        });
      }
      if (snapshot.status === 'unavailable') {
        return h(PlainNotice, {
          label: tr('nav'),
          title: tr('title'),
          message: tr('unavailable'),
        });
      }
      if (draft === null) {
        return h(PlainNotice, {
          label: tr('nav'),
          title: tr('title'),
          message: tr('loading'),
        });
      }

      const writable = snapshot.writable !== false;
      const disabled = !writable || saving;
      const ragPath = String(draft.ragPath || '');

      const pushBridge = async (payload) => {
        console.log('🔥 [Client] 推送配置到 Host', {
          ...payload,
          embedding_api_key: payload.embedding_api_key ? '***' : '',
        });
        try {
          const response = await fetch('/dsh-dream/config', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(payload),
          });
          const result = await response.json().catch(() => ({}));
          if (!response.ok) {
            throw new Error(
              result.error || `HTTP ${response.status}`,
            );
          }
          console.log('🔥 [Client] Host 已写入配置', {
            ok: result && result.ok,
          });
          return true;
        } catch (error) {
          console.error('🔥 [Client] 配置推送失败', error);
          return false;
        }
      };

      const edit = (patch) => {
        setDraft((current) => ({ ...(current || draft), ...patch }));
        setDirty(true);
        setSaved(false);
        setError('');
      };

      const discard = () => {
        setDraft(asDraft(snapshot.value));
        setDirty(false);
        setError('');
        setSaved(false);
      };

      const save = async () => {
        const parsed = parseDraft(draft);
        if (parsed.error) {
          setError(tr(parsed.error));
          return;
        }
        setSaving(true);
        setError('');
        const bridgePayload = {
          enable_memory: draft.enable,
          rag_path: String(draft.ragPath || '').trim(),
          max_window_tokens: parsed.max,
          retrieval_top_k: parsed.top,
          embedding_provider: parsed.provider,
          embedding_model: parsed.model,
          embedding_api_key: parsed.apiKey,
          embedding_base_url: parsed.baseUrl,
          auto_rebuild_collection: parsed.autoRebuild,
          holographic_mode: parsed.holographic,
        };
        const bridgePromise = pushBridge(bridgePayload);
        try {
          const accepted = await scope.mutate(
            [
              {
                op: 'set',
                path: ['enable_memory'],
                value: draft.enable,
              },
              {
                op: 'set',
                path: ['rag_path'],
                value: String(draft.ragPath || '').trim(),
              },
              {
                op: 'set',
                path: ['max_window_tokens'],
                value: parsed.max,
              },
              {
                op: 'set',
                path: ['retrieval_top_k'],
                value: parsed.top,
              },
              {
                op: 'set',
                path: ['embedding_provider'],
                value: parsed.provider,
              },
              {
                op: 'set',
                path: ['embedding_model'],
                value: parsed.model,
              },
              {
                op: 'set',
                path: ['embedding_api_key'],
                value: parsed.apiKey,
              },
              {
                op: 'set',
                path: ['embedding_base_url'],
                value: parsed.baseUrl,
              },
              {
                op: 'set',
                path: ['auto_rebuild_collection'],
                value: parsed.autoRebuild,
              },
              {
                op: 'set',
                path: ['holographic_mode'],
                value: parsed.holographic,
              },
            ],
            snapshot.revision,
          );
          console.log('🔥 [Client] scope.mutate accepted=', accepted);
          const bridgeOk = await bridgePromise;
          if (!accepted) {
            setError(tr('saveFailed'));
            return;
          }
          if (!bridgeOk) {
            console.warn(
              '🔥 [Client] settings saved but HTTP bridge failed; Host event fallback may run',
            );
          }
          // scope.mutate commits to the Host settings document. The HTTP
          // bridge above is the explicit same-origin write that guarantees
          // agent_memory_config.json is updated even if no Host event fires.
          setDirty(false);
          setSaved(true);
        } catch (error) {
          console.error('🔥 [Client] scope.mutate failed', error);
          await bridgePromise;
          setError(tr('saveFailed'));
        } finally {
          setSaving(false);
        }
      };

      return h(
        'section',
        { className: 'am-page', 'aria-label': tr('nav') },
        h('style', null, CSS),
        h(
          'header',
          { className: 'am-head' },
          h('h2', { className: 'am-title' }, tr('title')),
          h('p', { className: 'am-desc' }, tr('description')),
        ),
        !writable &&
          h(
            'p',
            { className: 'am-notice', role: 'status' },
            tr('readOnly'),
          ),
        h(
          'div',
          { className: 'am-holo' },
          h(
            'div',
            { className: 'am-copy' },
            h(
              'div',
              { className: 'am-label' },
              tr('holographic'),
              h('span', { className: 'am-holo-badge' }, tr('holographicBadge')),
            ),
            h('div', { className: 'am-hint' }, tr('holographicHint')),
          ),
          h(Toggle, {
            checked: draft.holographic,
            disabled,
            label: tr('holographic'),
            onChange: (value) => edit({ holographic: value }),
          }),
        ),
        h(
          'div',
          { className: 'am-card' },
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h('div', { className: 'am-label' }, tr('enable')),
              h('div', { className: 'am-hint' }, tr('enableHint')),
            ),
            h(Toggle, {
              checked: draft.enable,
              disabled,
              label: tr('enable'),
              onChange: (value) => edit({ enable: value }),
            }),
          ),
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-max-window' },
                tr('maxWindow'),
              ),
              h('div', { className: 'am-hint' }, tr('maxWindowHint')),
            ),
            h('input', {
              id: 'am-max-window',
              className: 'am-input',
              type: 'number',
              min: 128,
              max: 32768,
              step: 1,
              value: draft.max,
              disabled,
              onChange: (event) => edit({ max: event.target.value }),
            }),
          ),
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-top-k' },
                tr('topK'),
              ),
              h('div', { className: 'am-hint' }, tr('topKHint')),
            ),
            h('input', {
              id: 'am-top-k',
              className: 'am-input',
              type: 'number',
              min: 1,
              max: 20,
              step: 1,
              value: draft.top,
              disabled,
              onChange: (event) => edit({ top: event.target.value }),
            }),
          ),
          h(
            'div',
            { className: 'am-row am-row-last' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-rag-path' },
                tr('ragPath'),
              ),
              h('div', { className: 'am-hint' }, tr('ragPathHint')),
            ),
            h('input', {
              id: 'am-rag-path',
              className: 'am-input',
              type: 'text',
              value: ragPath,
              placeholder: '',
              onChange: (event) => edit({ ragPath: event.target.value }),
            }),
          ),
        ),
        h(
          'p',
          { className: 'am-warning', role: 'alert' },
          tr('embeddingWarning'),
        ),
        h('div', { className: 'am-subhead' }, tr('embeddingTitle')),
        h(
          'div',
          { className: 'am-card' },
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-embedding-provider' },
                tr('embeddingProvider'),
              ),
              h('div', { className: 'am-hint' }, tr('embeddingProviderHint')),
            ),
            h(
              'select',
              {
                id: 'am-embedding-provider',
                className: 'am-input am-select',
                value: draft.embeddingProvider,
                disabled,
                onChange: (event) => {
                  const next = event.target.value;
                  edit({
                    embeddingProvider: next,
                    ...(next === 'local' ? { embeddingModel: '' } : {}),
                  });
                },
              },
              h('option', { value: 'local' }, tr('providerLocal')),
              h(
                'option',
                { value: 'openai_compatible' },
                tr('providerCloud'),
              ),
            ),
          ),
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-embedding-model' },
                tr('embeddingModel'),
              ),
              h('div', { className: 'am-hint' }, tr('embeddingModelHint')),
            ),
            h('input', {
              id: 'am-embedding-model',
              className: 'am-input',
              type: 'text',
              value: draft.embeddingModel,
              placeholder: 'text-embedding-3-small',
              disabled,
              onChange: (event) =>
                edit({ embeddingModel: event.target.value }),
            }),
          ),
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-embedding-api-key' },
                tr('embeddingApiKey'),
              ),
              h('div', { className: 'am-hint' }, tr('embeddingApiKeyHint')),
            ),
            h('input', {
              id: 'am-embedding-api-key',
              className: 'am-input',
              type: 'password',
              value: draft.embeddingApiKey,
              placeholder: 'sk-…',
              autoComplete: 'new-password',
              spellCheck: false,
              disabled,
              onChange: (event) =>
                edit({ embeddingApiKey: event.target.value }),
            }),
          ),
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-embedding-base-url' },
                tr('embeddingBaseUrl'),
              ),
              h('div', { className: 'am-hint' }, tr('embeddingBaseUrlHint')),
            ),
            h('input', {
              id: 'am-embedding-base-url',
              className: 'am-input',
              type: 'text',
              value: draft.embeddingBaseUrl,
              placeholder:
                '例如 https://api.deepseek.com/v1（请确认与您的 API Key 匹配）',
              disabled,
              onChange: (event) =>
                edit({ embeddingBaseUrl: event.target.value }),
            }),
          ),
          h(
            'div',
            { className: 'am-row am-row-last' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-auto-rebuild' },
                tr('autoRebuild'),
              ),
              h('div', { className: 'am-hint' }, tr('autoRebuildHint')),
            ),
            h(
              'div',
              { className: 'am-check-row' },
              h('input', {
                id: 'am-auto-rebuild',
                type: 'checkbox',
                checked: draft.autoRebuild,
                disabled,
                onChange: (event) =>
                  edit({ autoRebuild: event.target.checked }),
              }),
            ),
          ),
        ),
        h(
          'footer',
          { className: 'am-actions' },
          h(
            'span',
            { className: 'am-status', role: 'status', 'aria-live': 'polite' },
            error || (saved ? tr('saved') : dirty ? tr('unsaved') : ''),
          ),
          h(
            'button',
            {
              type: 'button',
              className: 'am-button am-button-secondary',
              disabled: !dirty || saving || !writable,
              onClick: discard,
            },
            tr('discard'),
          ),
          h(
            'button',
            {
              type: 'button',
              className: 'am-button am-button-primary',
              disabled: !dirty || saving || !writable,
              onClick: save,
            },
            saving ? tr('saving') : tr('save'),
          ),
        ),
      );
    }

    function apply(ctx) {
      const t = ctx.locale.bind(NS);
      ctx.effect(
        () => ctx.locale.register(NS, { zh, en }),
        'DSH-DREAM: locale',
      );

      const scope = ctx.configForms.get(ENTRY_ID);
      ctx.effect(
        () =>
          ctx.configForms.whileServed([ENTRY_ID], () =>
            ctx.slots.inject('settings.section', () =>
              ctx.slots.register(
                {
                  name: 'settings.section',
                  id: 'DSH-DREAM',
                  order: 40,
                  label: () => t('nav'),
                  locale: NS,
                  inject: () => ({ scope }),
                },
                AgentMemorySection,
              ),
            ),
          ),
        'DSH-DREAM: settings section',
      );
    }

    return { inject, apply };
  },
});
