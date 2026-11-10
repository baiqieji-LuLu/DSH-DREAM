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
      autoRecall: 'Auto-recall before each turn',
      autoRecallHint:
        'Inject optional memory hints only when a highly relevant memory exists. Disable to skip automatic retrieval; memory_search remains available.',
      maxWindow: 'Window token budget',
      maxWindowHint:
        'Token budget used to initialize WindowManager (128–32768).',
      topK: 'Retrieval top-k',
      topKHint:
        'Number of RAG memories returned by memory_search (1–20).',
      ragPath: 'Chroma persistence path',
      ragPathHint:
        "Leave blank to use the DSH user directory; enter an absolute path to customize, or set AGENT_MEMORY_RAG_PATH to override.",
      pythonPath: 'Python interpreter path',
      pythonPathHint:
        '(Optional / advanced) Leave this blank — the plugin assembles its own isolated environment. Fill it in only if automatic setup failed, or you want to force a specific Python.',
      pythonPathPlaceholder: 'Auto-detect .venv or PATH',
      hfHome: 'Model cache directory',
      hfHomeHint:
        '(Optional) Leave blank to store models in your user directory. If your C: drive is full, point this at an empty folder on a D:/E: drive instead.',
      hfHomePlaceholder: 'Auto-detect a writable cache',
      runtimeCheck: 'Runtime check',
      runtimeChecking: 'Checking Python runtime…',
      runtimeProvisioning:
        'Initializing the memory engine (first run may take a few minutes)…',
      runtimeInstalled: 'Dependencies are installed. Restart DSH to continue.',
      runtimeOk: 'Python runtime is ready.',
      runtimeMissing: 'Missing Python packages:',
      runtimeCacheError:
        'Model cache directory is not writable; specify one in Settings.',
      runtimeUnknown: 'Python runtime could not be checked.',
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
      holoIndicator: 'Show holographic indicator',
      holoIndicatorHint:
        'A UI hint only — not the holographic switch. Hiding it does not stop recording. Drag the badge to move it.',
      holoIndicatorOn: 'HOLO · RECORDING',
      holoIndicatorAria:
        'Holographic mode is on. Every turn is being written to long-term memory.',
      holoIndicatorDrag: 'Drag to reposition',
      routerTitle: 'Smart routing (Router)',
      routerEnabled: 'Smart intent cutoff',
      routerEnabledHint:
        'When on, the AI first decides whether this topic needs to look through old memories. Small talk is answered directly, without wasting time on a database lookup.',
      routerMode: 'Decision priority',
      routerModeHint:
        'The default zero-config local judge takes about 5 ms and downloads nothing new. When a topic is ambiguous it lets retrieval happen: better to look too often than to forget.',
      routerModeAuto: 'auto (default)',
      routerModeCloud: 'cloud (prefer cloud)',
      routerModeLocal: 'local (fast local only)',
      routerModeRegex: 'regex (keywords only)',
      routerAdvancedTitle: 'Advanced: bring your own decision model (optional)',
      routerCloudBadge: 'ADVANCED',
      routerAdvancedNotice:
        'Regular users: leave this empty! Empty means the built-in fast local judge is used. Advanced users can connect an external cloud model (e.g. Alibaba Bailian, Jev) here to improve accuracy on complex wording.',
      routerCloudBaseUrl: 'Cloud endpoint URL',
      routerCloudBaseUrlHint:
        'Full endpoint URL including https://. Leave blank to keep the built-in fast local judge.',
      routerCloudBaseUrlPlaceholder:
        'e.g. https://ws-xxx.cn-beijing.maas.aliyuncs.com',
      routerCloudPath: 'Request path',
      routerCloudPathHint: 'Leave blank to use the provider default path.',
      routerCloudPathPlaceholder: 'Default /compatible-mode/v1/systemone',
      routerCloudModel: 'Cloud model name',
      routerCloudModelHint:
        'Leave blank to use the preset model for this provider.',
      routerCloudModelPlaceholder: 'e.g. decision-model-preview or jev-latest',
      routerCloudApiKeyEnv: 'API key env var name (keeps the key safe)',
      routerCloudApiKeyEnvHint:
        'Enter only the environment-variable name, never the key itself. Put the key in your system environment variables.',
      routerCloudApiKeyEnvPlaceholder:
        'e.g. DASHSCOPE_API_KEY (the env var you configured on this machine)',
      invalidRouterBaseUrl:
        'The cloud endpoint URL must start with http:// or https://, or be left blank.',
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
      autoRecall: '每轮自动预检索',
      autoRecallHint:
        '仅在记忆高度相关时注入可选的记忆提示。关闭后不再自动检索；memory_search 工具仍然可用。',
      maxWindow: '窗口 Token 预算',
      maxWindowHint: '用于初始化 WindowManager 的 Token 预算（128–32768）。',
      topK: 'RAG 召回数量',
      topKHint: 'memory_search 返回的记忆条数（1–20）。',
      ragPath: 'Chroma 持久化路径',
      ragPathHint:
        '留空则默认存储在 DSH 用户目录下；如需自定义，请填入绝对路径，或通过环境变量 AGENT_MEMORY_RAG_PATH 覆盖。',
      pythonPath: 'Python 解释器路径',
      pythonPathHint:
        '（选填/进阶向）一般留空即可！插件会自动帮你装配隔离环境。仅当自动装配失败，或你想强制指定某个 Python 时再填写。',
      pythonPathPlaceholder: '自动查找 .venv 或 PATH',
      hfHome: '模型缓存目录',
      hfHomeHint:
        '（选填）留空则自动存放到系统用户目录。如果你 C 盘爆满，可以在这里填个 D 盘或 E 盘的空文件夹路径。',
      hfHomePlaceholder: '自动寻找可写缓存目录',
      runtimeCheck: '运行环境检查',
      runtimeChecking: '正在检查 Python 运行环境…',
      runtimeProvisioning: '正在初始化记忆引擎（首次运行可能需要几分钟）…',
      runtimeInstalled: '依赖装配完成，请重启 DSH。',
      runtimeOk: 'Python 运行环境正常。',
      runtimeMissing: '缺少 Python 依赖：',
      runtimeCacheError: '模型缓存目录不可写，请手动在设置中指定目录',
      runtimeUnknown: '无法检查 Python 运行环境。',
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
      holoIndicator: '显示全息提示',
      holoIndicatorHint:
        '这只是界面提示，不是全息开关；关闭提示不会停止落库。小灯可在页面上拖动。',
      holoIndicatorOn: '全息 · 落库中',
      holoIndicatorAria:
        '全息模式已开启，每轮对话都会自动写入长期记忆。',
      holoIndicatorDrag: '拖动可调整位置',
      routerTitle: '智能路由（Router）',
      routerEnabled: '智能意图截断',
      routerEnabledHint:
        '开启后，AI 会先判断话题是否需要翻阅旧记忆。闲聊直接回复，不浪费时间查库。',
      routerMode: '判断优先级',
      routerModeHint:
        '默认采用零配置的本地极速判断（约 5 毫秒，不需下载新模型）。如果遇到拿不准的话题，默认放行去查库，宁可多查也不失忆。',
      routerModeAuto: 'auto（自动-默认）',
      routerModeCloud: 'cloud（优先云端）',
      routerModeLocal: 'local（仅本地极速）',
      routerModeRegex: 'regex（仅关键词规则）',
      routerAdvancedTitle: '高级：自定义外接判断模型（选填）',
      routerCloudBadge: '进阶',
      routerAdvancedNotice:
        '普通用户请留空！留空即代表使用内置极速本地判断。如果你是进阶用户，可在此接入外部云端模型（如阿里百炼、Jev）来提升复杂语境的判断准确率。',
      routerCloudBaseUrl: '云端接口地址',
      routerCloudBaseUrlHint:
        '填写完整的接口地址（含 https://）；留空则继续使用内置的本地极速判断。',
      routerCloudBaseUrlPlaceholder:
        '例如 https://ws-xxx.cn-beijing.maas.aliyuncs.com',
      routerCloudPath: '请求路径',
      routerCloudPathHint: '留空则使用该服务商的默认路径。',
      routerCloudPathPlaceholder: '默认 /compatible-mode/v1/systemone',
      routerCloudModel: '云端模型名称',
      routerCloudModelHint: '留空则使用该服务商预设的模型名。',
      routerCloudModelPlaceholder: '例如 decision-model-preview 或 jev-latest',
      routerCloudApiKeyEnv: 'API Key 环境变量名（保护密钥）',
      routerCloudApiKeyEnvHint:
        '只填环境变量的名字，不要填密钥本身；密钥请配置在系统环境变量里。',
      routerCloudApiKeyEnvPlaceholder:
        '例如 DASHSCOPE_API_KEY（填入你在电脑里配的环境变量名）',
      invalidRouterBaseUrl:
        '云端接口地址必须以 http:// 或 https:// 开头，或直接留空。',
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
      .am-holo-indicator {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 0.5px dashed var(--dsw-alias-border-l3);
      }
      .am-holo-indicator-label {
        color: var(--dsw-alias-label-primary);
        font-weight: 500;
      }
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
      .am-advanced {
        margin: 0 0 8px;
        padding: 12px 14px;
        border: 0.5px dashed var(--dsw-alias-border-l3);
        border-radius: var(--dsw-radius-xl, 12px);
        background: color-mix(
          in srgb,
          var(--dsw-alias-bg-base) 92%,
          var(--dsw-alias-label-primary)
        );
      }
      .am-advanced-summary {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        color: var(--dsw-alias-label-primary);
        font-weight: 500;
        list-style: none;
      }
      .am-advanced-summary::-webkit-details-marker { display: none; }
      .am-advanced-summary::before {
        content: '▸';
        color: var(--dsw-alias-label-tertiary);
        font-size: 11px;
        transition: transform 0.16s ease;
      }
      .am-advanced[open] .am-advanced-summary::before {
        transform: rotate(90deg);
      }
      .am-advanced-badge {
        padding: 1px 6px;
        border-radius: 999px;
        border: 0.5px solid var(--dsw-alias-border-l3);
        color: var(--dsw-alias-label-secondary);
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.04em;
      }
      .am-advanced-notice {
        margin: 10px 0 6px;
        padding: 8px 10px;
        border-radius: var(--dsw-radius-md, 8px);
        background: var(--dsw-alias-settings-card-fill, var(--dsw-alias-bg-base));
        color: var(--dsw-alias-label-secondary);
        font-size: 12px;
        line-height: 18px;
      }
      .am-advanced-card { margin-top: 6px; }
      .am-advanced-card .am-row { padding: 12px 14px; }
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

    /*
     * Holographic indicator (frame overlay).
     *
     * The badge lives in `shell.overlay`, the frame-wide click-through layer,
     * so it can float at the top of the conversation box and be dragged
     * anywhere without participating in the composer layout. Its position is
     * frame-relative and remembered in localStorage; a frame resize clamps a
     * dragged badge and re-anchors a never-dragged one to the conversation box.
     */
    const HOLO_POSITION_KEY = 'dsh-dream.holo-indicator.position.v1';
    const HOLO_DEFAULT_INSET = 12;

    const HOLO_CSS = `
      .dsh-dream-holo-indicator {
        position: absolute;
        z-index: 30;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 2px 8px;
        border: 0.5px solid var(--dsw-alias-border-l3, rgba(128, 128, 128, 0.45));
        border-radius: 999px;
        background: var(--dsw-alias-bg-layer-2, rgba(127, 127, 127, 0.18));
        color: var(--dsw-alias-label-secondary, #666666);
        font-size: 11px;
        line-height: 16px;
        font-weight: 600;
        letter-spacing: 0.02em;
        white-space: nowrap;
        user-select: none;
        cursor: grab;
        touch-action: none;
        pointer-events: auto;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.16);
      }
      .dsh-dream-holo-indicator[data-dragging='true'] {
        cursor: grabbing;
      }
      .dsh-dream-holo-indicator:focus-visible {
        outline: 2px solid var(--dsw-focus-ring-color, var(--dsw-alias-state-business-primary));
        outline-offset: 2px;
      }
      .dsh-dream-holo-indicator-dot {
        flex: none;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #ff453a;
        box-shadow:
          0 0 0 1px #ffffff,
          0 0 0 2px rgba(0, 0, 0, 0.55);
      }
    `;

    const EMPTY_HOLO_SNAPSHOT = { status: 'unavailable', value: {} };

    function findHoloChatHost() {
      if (typeof document === 'undefined') return null;
      const hosts = Array.from(
        document.querySelectorAll('[data-conversation-region="chat"]'),
      );
      let best = null;
      let bestArea = 0;
      for (const host of hosts) {
        const rect = host.getBoundingClientRect?.();
        if (!rect || rect.width <= 0 || rect.height <= 0) continue;
        const area = rect.width * rect.height;
        if (area > bestArea) {
          best = host;
          bestArea = area;
        }
      }
      return best;
    }

    function holoOverlayRect(node) {
      const overlay = node?.closest?.('[data-shell-overlay]');
      const rect = overlay?.getBoundingClientRect?.();
      if (rect && rect.width > 0 && rect.height > 0) return rect;
      if (typeof window === 'undefined') {
        return { left: 0, top: 0, width: 0, height: 0 };
      }
      return {
        left: 0,
        top: 0,
        width: window.innerWidth,
        height: window.innerHeight,
      };
    }

    function clampHoloPosition(left, top, node) {
      const frame = holoOverlayRect(node);
      const width = node?.offsetWidth || 150;
      const height = node?.offsetHeight || 28;
      return {
        left: Math.min(Math.max(0, left), Math.max(0, frame.width - width)),
        top: Math.min(Math.max(0, top), Math.max(0, frame.height - height)),
      };
    }

    function defaultHoloPosition(node) {
      const frame = holoOverlayRect(node);
      const chat = findHoloChatHost();
      const chatRect = chat?.getBoundingClientRect?.();
      if (!chatRect) {
        return clampHoloPosition(
          HOLO_DEFAULT_INSET,
          HOLO_DEFAULT_INSET,
          node,
        );
      }
      return clampHoloPosition(
        chatRect.left - frame.left + HOLO_DEFAULT_INSET,
        chatRect.top - frame.top + HOLO_DEFAULT_INSET,
        node,
      );
    }

    function readHoloPosition() {
      try {
        const raw = window.localStorage.getItem(HOLO_POSITION_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (
          Number.isFinite(parsed?.left) &&
          Number.isFinite(parsed?.top)
        ) {
          return { left: parsed.left, top: parsed.top };
        }
      } catch {
        // A private/blocked localStorage must not break the badge.
      }
      return null;
    }

    function saveHoloPosition(position) {
      try {
        window.localStorage.setItem(
          HOLO_POSITION_KEY,
          JSON.stringify(position),
        );
      } catch {
        // Position persistence is a convenience, never a requirement.
      }
    }

    function HoloIndicator(props) {
      const { t, scope } = props;
      const tr = typeof t === 'function' ? t : (key) => key;
      const subscribe = React.useCallback(
        (listener) =>
          scope && typeof scope.subscribe === 'function'
            ? scope.subscribe(listener)
            : () => {},
        [scope],
      );
      const getSnapshot = React.useCallback(
        () =>
          scope && typeof scope.getSnapshot === 'function'
            ? scope.getSnapshot()
            : EMPTY_HOLO_SNAPSHOT,
        [scope],
      );
      const snapshot = React.useSyncExternalStore(subscribe, getSnapshot);
      const value =
        snapshot && snapshot.value && typeof snapshot.value === 'object'
          ? snapshot.value
          : {};
      const holographic = value.holographic_mode === true;
      const indicator =
        value.holo_indicator && typeof value.holo_indicator === 'object'
          ? value.holo_indicator
          : {};
      const active = holographic && indicator.enabled !== false;
      const badgeRef = React.useRef(null);
      const positionRef = React.useRef(null);
      const customRef = React.useRef(false);
      const dragRef = React.useRef(null);
      const [position, setPosition] = React.useState(null);
      const [dragging, setDragging] = React.useState(false);

      const applyPosition = React.useCallback((next, custom) => {
        positionRef.current = next;
        customRef.current = custom;
        setPosition(next);
      }, []);

      React.useLayoutEffect(() => {
        if (!active) return undefined;
        const node = badgeRef.current;
        if (!node) return undefined;
        const saved = readHoloPosition();
        if (saved) {
          applyPosition(clampHoloPosition(saved.left, saved.top, node), true);
        } else {
          applyPosition(defaultHoloPosition(node), false);
        }
        return undefined;
      }, [active, applyPosition]);

      React.useEffect(() => {
        if (!active) return undefined;
        const onResize = () => {
          const node = badgeRef.current;
          if (!node) return;
          const current = positionRef.current;
          if (!current) return;
          applyPosition(
            customRef.current
              ? clampHoloPosition(current.left, current.top, node)
              : defaultHoloPosition(node),
            customRef.current,
          );
        };
        window.addEventListener('resize', onResize);
        const overlay = badgeRef.current?.closest?.('[data-shell-overlay]');
        const observer =
          typeof ResizeObserver !== 'undefined' && overlay
            ? new ResizeObserver(onResize)
            : null;
        observer?.observe(overlay);
        if (observer && badgeRef.current) observer.observe(badgeRef.current);
        return () => {
          window.removeEventListener('resize', onResize);
          observer?.disconnect();
        };
      }, [active, applyPosition]);

      const onPointerDown = (event) => {
        if (event.button !== undefined && event.button !== 0) return;
        const node = badgeRef.current;
        if (!node) return;
        event.preventDefault();
        event.currentTarget.focus?.();
        const start = positionRef.current ?? defaultHoloPosition(node);
        positionRef.current = start;
        dragRef.current = {
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          originLeft: start.left,
          originTop: start.top,
        };
        try {
          event.currentTarget.setPointerCapture?.(event.pointerId);
        } catch {
          // Pointer capture is best-effort; the badge still moves on move.
        }
        setDragging(true);
      };

      const onPointerMove = (event) => {
        const drag = dragRef.current;
        const node = badgeRef.current;
        if (!drag || !node || drag.pointerId !== event.pointerId) return;
        event.preventDefault();
        applyPosition(
          clampHoloPosition(
            drag.originLeft + event.clientX - drag.startX,
            drag.originTop + event.clientY - drag.startY,
            node,
          ),
          true,
        );
      };

      const finishDrag = (event) => {
        const drag = dragRef.current;
        if (!drag) return;
        dragRef.current = null;
        setDragging(false);
        try {
          event.currentTarget.releasePointerCapture?.(drag.pointerId);
        } catch {
          // Ignore a capture already released by the browser.
        }
        if (positionRef.current) saveHoloPosition(positionRef.current);
      };

      const onKeyDown = (event) => {
        const step = event.shiftKey ? 24 : 8;
        let dx = 0;
        let dy = 0;
        if (event.key === 'ArrowLeft') dx = -step;
        else if (event.key === 'ArrowRight') dx = step;
        else if (event.key === 'ArrowUp') dy = -step;
        else if (event.key === 'ArrowDown') dy = step;
        if (!dx && !dy) return;
        const node = badgeRef.current;
        if (!node) return;
        event.preventDefault();
        const current = positionRef.current ?? defaultHoloPosition(node);
        const next = clampHoloPosition(
          current.left + dx,
          current.top + dy,
          node,
        );
        applyPosition(next, true);
        saveHoloPosition(next);
      };

      if (!active) return null;

      const ariaLabel = `${tr('holoIndicatorAria')} ${tr('holoIndicatorDrag')}`;

      return h(
        React.Fragment,
        null,
        h('style', { key: 'dsh-dream-holo-css' }, HOLO_CSS),
        h(
          'span',
          {
            ref: badgeRef,
            className: 'dsh-dream-holo-indicator',
            role: 'status',
            tabIndex: 0,
            'aria-live': 'polite',
            'aria-label': ariaLabel,
            'aria-keyshortcuts': 'ArrowLeft ArrowRight ArrowUp ArrowDown',
            title: ariaLabel,
            'data-dsh-dream-holo-indicator': '',
            'data-dragging': dragging ? 'true' : undefined,
            style: position
              ? {
                  left: `${position.left}px`,
                  top: `${position.top}px`,
                  visibility: 'visible',
                }
              : { left: '0px', top: '0px', visibility: 'hidden' },
            onPointerDown,
            onPointerMove,
            onPointerUp: finishDrag,
            onPointerCancel: finishDrag,
            onLostPointerCapture: finishDrag,
            onKeyDown,
          },
          h('span', {
            className: 'dsh-dream-holo-indicator-dot',
            'aria-hidden': true,
          }),
          h(
            'span',
            { className: 'dsh-dream-holo-indicator-text' },
            tr('holoIndicatorOn'),
          ),
        ),
      );
    }

    function normalizeRouterMode(value) {
      const text = String(value ?? 'auto')
        .trim()
        .toLowerCase();
      return ['auto', 'cloud', 'local', 'regex'].includes(text) ? text : 'auto';
    }

    function asDraft(value) {
      const current = value && typeof value === 'object' ? value : {};
      const router =
        current.router && typeof current.router === 'object'
          ? current.router
          : {};
      const cloud =
        router.cloud && typeof router.cloud === 'object' ? router.cloud : {};
      const holoIndicator =
        current.holo_indicator && typeof current.holo_indicator === 'object'
          ? current.holo_indicator
          : {};
      const activeProvider = String(cloud.active ?? '');
      const providerConfig =
        activeProvider &&
        cloud.providers &&
        typeof cloud.providers === 'object' &&
        cloud.providers[activeProvider] &&
        typeof cloud.providers[activeProvider] === 'object'
          ? cloud.providers[activeProvider]
          : {};
      // The DSH settings form stores flat router_* fields; the custom page and
      // the bridge file use the nested block. Accept both.
      const flat = (key, fallback) => {
        if (
          Object.prototype.hasOwnProperty.call(current, key) &&
          current[key] !== undefined
        ) {
          return current[key];
        }
        return fallback;
      };
      return {
        enable: current.enable_memory !== false,
        autoRecall: current.enable_auto_recall !== false,
        ragPath: String(current.rag_path ?? ''),
        pythonPath: String(current.python_path ?? ''),
        hfHomePath: String(current.hf_home_path ?? ''),
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
        holoIndicatorEnabled: holoIndicator.enabled !== false,
        routerEnabled: flat('router_enabled', router.enabled) !== false,
        routerMode: normalizeRouterMode(flat('router_mode', router.mode)),
        routerCloudBaseUrl: String(
          flat('router_cloud_base_url', providerConfig.base_url ?? '') ?? '',
        ),
        routerCloudPath: String(
          flat('router_cloud_path', providerConfig.path ?? '') ?? '',
        ),
        routerCloudModel: String(
          flat('router_cloud_model', providerConfig.model ?? '') ?? '',
        ),
        routerCloudApiKeyEnv: String(
          flat('router_cloud_api_key_env', providerConfig.api_key_env ?? '') ??
            '',
        ),
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
      const routerBaseUrl = String(draft.routerCloudBaseUrl || '').trim();
      if (routerBaseUrl && !/^https?:\/\//i.test(routerBaseUrl)) {
        return { error: 'invalidRouterBaseUrl' };
      }
      return {
        max,
        top,
        provider,
        model: String(draft.embeddingModel || '').trim(),
        apiKey,
        baseUrl,
        autoRebuild: Boolean(draft.autoRebuild),
        autoRecall: Boolean(draft.autoRecall),
        holographic: Boolean(draft.holographic),
        holoIndicatorEnabled: draft.holoIndicatorEnabled !== false,
        routerEnabled: Boolean(draft.routerEnabled),
        routerMode: normalizeRouterMode(draft.routerMode),
        routerBaseUrl,
        routerPath: String(draft.routerCloudPath || '').trim(),
        routerModel: String(draft.routerCloudModel || '').trim(),
        routerApiKeyEnv: String(draft.routerCloudApiKeyEnv || '').trim(),
        // A filled endpoint is the user's opt-in for the external judge;
        // leaving it blank means "use the built-in local judge".
        routerCloudEnabled: Boolean(routerBaseUrl),
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
      // Open the advanced cloud block automatically when a cloud endpoint is
      // already configured, so an existing setup is never hidden.
      const [advancedOpen, setAdvancedOpen] = React.useState(false);
      const advancedInitialized = React.useRef(false);
      const [runtime, setRuntime] = React.useState({
        state: 'checking',
        missing: [],
        cacheError: '',
        provisionStatus: '',
      });

      React.useEffect(() => {
        let active = true;
        let timer = null;
        const load = () => {
          fetch('/dsh-dream/runtime', { credentials: 'same-origin' })
            .then((response) => response.json())
            .then((result) => {
              if (!active) return;
              const provisionStatus = result?.provision?.status || '';
              setRuntime({
                state: result?.ok ? 'ok' : 'missing',
                missing: Array.isArray(result?.missing) ? result.missing : [],
                cacheError: result?.cache?.error || '',
                provisionStatus,
              });
              if (
                provisionStatus === 'provisioning' ||
                provisionStatus === 'checking'
              ) {
                timer = setTimeout(load, 3000);
              }
            })
            .catch(() => {
              if (active) {
                setRuntime({
                  state: 'unknown',
                  missing: [],
                  cacheError: '',
                  provisionStatus: '',
                });
              }
            });
        };
        load();
        return () => {
          active = false;
          if (timer) clearTimeout(timer);
        };
      }, []);

      React.useEffect(() => {
        if (snapshot.status === 'ready' && !dirty) {
          setDraft(asDraft(snapshot.value));
        }
      }, [snapshot.status, snapshot.revision, dirty]);

      React.useEffect(() => {
        if (!draft || advancedInitialized.current) return;
        advancedInitialized.current = true;
        if (String(draft.routerCloudBaseUrl || '').trim()) {
          setAdvancedOpen(true);
        }
      }, [draft]);

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
      const pythonPath = String(draft.pythonPath || '');
      const hfHomePath = String(draft.hfHomePath || '');
      const runtimeText =
        runtime.provisionStatus === 'provisioning'
          ? tr('runtimeProvisioning')
          : runtime.provisionStatus === 'installed'
            ? tr('runtimeInstalled')
            : runtime.state === 'checking'
              ? tr('runtimeChecking')
              : runtime.cacheError
                ? tr('runtimeCacheError')
                : runtime.state === 'ok'
                  ? tr('runtimeOk')
                  : runtime.state === 'missing'
                    ? `${tr('runtimeMissing')} ${runtime.missing.join(', ')}`.trim()
                    : tr('runtimeUnknown');

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
          enable_auto_recall: parsed.autoRecall,
          rag_path: String(draft.ragPath || '').trim(),
          python_path: String(draft.pythonPath || '').trim(),
          hf_home_path: String(draft.hfHomePath || '').trim(),
          max_window_tokens: parsed.max,
          retrieval_top_k: parsed.top,
          embedding_provider: parsed.provider,
          embedding_model: parsed.model,
          embedding_api_key: parsed.apiKey,
          embedding_base_url: parsed.baseUrl,
          auto_rebuild_collection: parsed.autoRebuild,
          holographic_mode: parsed.holographic,
          holo_indicator: {
            enabled: parsed.holoIndicatorEnabled,
          },
          // Router fields stay flat here; the Host assembles the nested
          // `router` block and preserves any advanced setting the page did
          // not render (budget, threshold, circuit, audit).
          router_enabled: parsed.routerEnabled,
          router_mode: parsed.routerMode,
          router_cloud_enabled: parsed.routerCloudEnabled,
          router_cloud_provider: parsed.routerCloudEnabled ? 'custom' : '',
          router_cloud_base_url: parsed.routerBaseUrl,
          router_cloud_path: parsed.routerPath,
          router_cloud_model: parsed.routerModel,
          router_cloud_api_key_env: parsed.routerApiKeyEnv,
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
                path: ['enable_auto_recall'],
                value: parsed.autoRecall,
              },
              {
                op: 'set',
                path: ['rag_path'],
                value: String(draft.ragPath || '').trim(),
              },
              {
                op: 'set',
                path: ['python_path'],
                value: String(draft.pythonPath || '').trim(),
              },
              {
                op: 'set',
                path: ['hf_home_path'],
                value: String(draft.hfHomePath || '').trim(),
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
              {
                op: 'set',
                path: ['holo_indicator'],
                value: {
                  enabled: parsed.holoIndicatorEnabled,
                },
              },
              {
                op: 'set',
                path: ['router_enabled'],
                value: parsed.routerEnabled,
              },
              {
                op: 'set',
                path: ['router_mode'],
                value: parsed.routerMode,
              },
              {
                op: 'set',
                path: ['router_cloud_enabled'],
                value: parsed.routerCloudEnabled,
              },
              {
                op: 'set',
                path: ['router_cloud_provider'],
                value: parsed.routerCloudEnabled ? 'custom' : '',
              },
              {
                op: 'set',
                path: ['router_cloud_base_url'],
                value: parsed.routerBaseUrl,
              },
              {
                op: 'set',
                path: ['router_cloud_path'],
                value: parsed.routerPath,
              },
              {
                op: 'set',
                path: ['router_cloud_model'],
                value: parsed.routerModel,
              },
              {
                op: 'set',
                path: ['router_cloud_api_key_env'],
                value: parsed.routerApiKeyEnv,
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
            h(
              'div',
              { className: 'am-holo-indicator' },
              h(
                'span',
                { className: 'am-holo-indicator-label' },
                tr('holoIndicator'),
              ),
              h(Toggle, {
                checked: draft.holoIndicatorEnabled,
                disabled,
                label: tr('holoIndicator'),
                onChange: (value) => edit({ holoIndicatorEnabled: value }),
              }),
            ),
            h('div', { className: 'am-hint' }, tr('holoIndicatorHint')),
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
              h('div', { className: 'am-label' }, tr('autoRecall')),
              h('div', { className: 'am-hint' }, tr('autoRecallHint')),
            ),
            h(Toggle, {
              checked: draft.autoRecall,
              disabled: disabled || !draft.enable,
              label: tr('autoRecall'),
              onChange: (value) => edit({ autoRecall: value }),
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
            { className: 'am-row' },
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
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h(
                'label',
                { className: 'am-label', htmlFor: 'am-python-path' },
                tr('pythonPath'),
              ),
              h('div', { className: 'am-hint' }, tr('pythonPathHint')),
            ),
            h('input', {
              id: 'am-python-path',
              className: 'am-input',
              type: 'text',
              value: pythonPath,
              placeholder: tr('pythonPathPlaceholder'),
              disabled,
              onChange: (event) => edit({ pythonPath: event.target.value }),
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
                { className: 'am-label', htmlFor: 'am-hf-home' },
                tr('hfHome'),
              ),
              h('div', { className: 'am-hint' }, tr('hfHomeHint')),
            ),
            h('input', {
              id: 'am-hf-home',
              className: 'am-input',
              type: 'text',
              value: hfHomePath,
              placeholder: tr('hfHomePlaceholder'),
              disabled,
              onChange: (event) => edit({ hfHomePath: event.target.value }),
            }),
          ),
          h(
            'div',
            {
              className: 'am-row am-row-last',
              role: 'status',
              'aria-live': 'polite',
            },
            h(
              'div',
              { className: 'am-copy' },
              h('div', { className: 'am-label' }, tr('runtimeCheck')),
              h('div', { className: 'am-hint' }, runtimeText),
            ),
          ),
        ),
        h('div', { className: 'am-subhead' }, tr('routerTitle')),
        h(
          'div',
          { className: 'am-card' },
          h(
            'div',
            { className: 'am-row' },
            h(
              'div',
              { className: 'am-copy' },
              h('div', { className: 'am-label' }, tr('routerEnabled')),
              h('div', { className: 'am-hint' }, tr('routerEnabledHint')),
            ),
            h(Toggle, {
              checked: draft.routerEnabled,
              disabled,
              label: tr('routerEnabled'),
              onChange: (value) => edit({ routerEnabled: value }),
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
                { className: 'am-label', htmlFor: 'am-router-mode' },
                tr('routerMode'),
              ),
              h('div', { className: 'am-hint' }, tr('routerModeHint')),
            ),
            h(
              'select',
              {
                id: 'am-router-mode',
                className: 'am-input am-select',
                value: draft.routerMode,
                disabled: disabled || !draft.routerEnabled,
                onChange: (event) =>
                  edit({ routerMode: event.target.value }),
              },
              h('option', { value: 'auto' }, tr('routerModeAuto')),
              h('option', { value: 'cloud' }, tr('routerModeCloud')),
              h('option', { value: 'local' }, tr('routerModeLocal')),
              h('option', { value: 'regex' }, tr('routerModeRegex')),
            ),
          ),
        ),
        h(
          'details',
          {
            className: 'am-advanced',
            open: advancedOpen,
            onToggle: (event) => setAdvancedOpen(event.currentTarget.open),
          },
          h(
            'summary',
            { className: 'am-advanced-summary' },
            h('span', null, tr('routerAdvancedTitle')),
            h(
              'span',
              { className: 'am-advanced-badge' },
              tr('routerCloudBadge'),
            ),
          ),
          h(
            'p',
            { className: 'am-advanced-notice' },
            tr('routerAdvancedNotice'),
          ),
          h(
            'div',
            { className: 'am-card am-advanced-card' },
            h(
              'div',
              { className: 'am-row' },
              h(
                'div',
                { className: 'am-copy' },
                h(
                  'label',
                  { className: 'am-label', htmlFor: 'am-router-cloud-url' },
                  tr('routerCloudBaseUrl'),
                ),
                h('div', { className: 'am-hint' }, tr('routerCloudBaseUrlHint')),
              ),
              h('input', {
                id: 'am-router-cloud-url',
                className: 'am-input',
                type: 'text',
                value: draft.routerCloudBaseUrl,
                placeholder: tr('routerCloudBaseUrlPlaceholder'),
                disabled,
                spellCheck: false,
                onChange: (event) =>
                  edit({ routerCloudBaseUrl: event.target.value }),
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
                  { className: 'am-label', htmlFor: 'am-router-cloud-path' },
                  tr('routerCloudPath'),
                ),
                h('div', { className: 'am-hint' }, tr('routerCloudPathHint')),
              ),
              h('input', {
                id: 'am-router-cloud-path',
                className: 'am-input',
                type: 'text',
                value: draft.routerCloudPath,
                placeholder: tr('routerCloudPathPlaceholder'),
                disabled,
                spellCheck: false,
                onChange: (event) =>
                  edit({ routerCloudPath: event.target.value }),
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
                  { className: 'am-label', htmlFor: 'am-router-cloud-model' },
                  tr('routerCloudModel'),
                ),
                h('div', { className: 'am-hint' }, tr('routerCloudModelHint')),
              ),
              h('input', {
                id: 'am-router-cloud-model',
                className: 'am-input',
                type: 'text',
                value: draft.routerCloudModel,
                placeholder: tr('routerCloudModelPlaceholder'),
                disabled,
                spellCheck: false,
                onChange: (event) =>
                  edit({ routerCloudModel: event.target.value }),
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
                  {
                    className: 'am-label',
                    htmlFor: 'am-router-cloud-key-env',
                  },
                  tr('routerCloudApiKeyEnv'),
                ),
                h('div', { className: 'am-hint' }, tr('routerCloudApiKeyEnvHint')),
              ),
              h('input', {
                id: 'am-router-cloud-key-env',
                className: 'am-input',
                type: 'text',
                value: draft.routerCloudApiKeyEnv,
                placeholder: tr('routerCloudApiKeyEnvPlaceholder'),
                disabled,
                spellCheck: false,
                autoComplete: 'off',
                onChange: (event) =>
                  edit({ routerCloudApiKeyEnv: event.target.value }),
              }),
            ),
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

      // The badge reads the same live ConfigForm snapshot as the settings page
      // (the `dsh-dream-settings` entry, with its revision). It registers in the
      // frame-wide overlay so it can float and be dragged; it renders null while
      // holographic mode or the indicator is off, so the pre-feature interface
      // is preserved exactly.
      ctx.effect(
        () =>
          ctx.slots.inject('shell.overlay', () =>
            ctx.slots.register(
              {
                name: 'shell.overlay',
                id: 'dsh-dream-holo-indicator',
                order: 10,
                locale: NS,
                inject: () => ({ scope }),
              },
              HoloIndicator,
            ),
          ),
        'DSH-DREAM: holographic indicator',
      );
    }

    return { inject, apply };
  },
});
