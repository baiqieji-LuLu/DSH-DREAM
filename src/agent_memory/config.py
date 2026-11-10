"""Configuration for the DSH-DREAM plugin.

The DSH Host settings panel writes a small JSON bridge file from the native
``Config`` form. This module reads that file (or a compatible manifest default)
and returns a validated :class:`MemoryConfig`.

The bridge-file resolution order is:

1. explicitly supplied ``config_path``;
2. ``DSH_DREAM_CONFIG``;
3. ``AGENT_MEMORY_CONFIG``;
4. ``agent_memory_config.json`` in the project root;
5. defaults from ``manifest.json``.

Within the resolved config, ``rag_path`` priority is:

1. the value entered in the DSH UI / bridge file;
2. ``AGENT_MEMORY_RAG_PATH``.

There is deliberately no implicit ``.chroma`` fallback: a missing/blank value
raises :class:`ConfigError` instead of silently opening a stale database.

中文说明：
DSH 原生设置面板通过 Host 插件的 ``Config`` 表单把配置写入一个 JSON 桥接文件。
本模块读取该文件并返回校验后的 :class:`MemoryConfig`；``rag_path`` 优先级为：
UI/桥接文件显式值 → ``AGENT_MEMORY_RAG_PATH``。两者都没有时直接报错，
不再回退到任何隐式的 ``.chroma`` 数据库。
Host 使用 ``DSH_DREAM_CONFIG`` 指向当前 profile，避免旧的 ``AGENT_MEMORY_CONFIG`` 劫持。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MANIFEST_PATH: Final[Path] = PROJECT_ROOT / "manifest.json"


def _default_rag_path() -> Path:
    """Return a per-user DSH data path, never a checkout/package path."""
    profile = os.environ.get("DSH_PROFILE_DIR", "").strip()
    if profile:
        return (Path(profile).expanduser() / ".chroma").resolve()
    dsh_home = os.environ.get("DSH_HOME", "").strip()
    home = (
        Path(dsh_home).expanduser()
        if dsh_home
        else (Path.home() / ".dsh")
    )
    return (home / "profiles" / "desktop" / ".chroma").resolve()


# The DSH Host plugin writes this exact project-root path by default; an
# explicit AGENT_MEMORY_CONFIG still wins for non-default checkouts.
DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "agent_memory_config.json"
CONFIG_ENV: Final[str] = "AGENT_MEMORY_CONFIG"
DSH_CONFIG_ENV: Final[str] = "DSH_DREAM_CONFIG"
DEFAULT_RAG_PATH: Final[Path] = _default_rag_path()
DEFAULT_MAX_WINDOW_TOKENS: Final[int] = 1_500
DEFAULT_RETRIEVAL_TOP_K: Final[int] = 5
MAX_WINDOW_TOKENS: Final[int] = 32_768
MAX_TOP_K: Final[int] = 20
DEFAULT_STORAGE_BACKEND: Final[str] = "sqlite-vec"
DEFAULT_EMBEDDING_ENGINE: Final[str] = "onnx"
DEFAULT_EMBEDDING_PROVIDER: Final[str] = "local"
DEFAULT_EMBEDDING_MODEL: Final[str] = ""
DEFAULT_EMBEDDING_API_KEY: Final[str] = ""
DEFAULT_EMBEDDING_BASE_URL: Final[str] = ""

# --- RouterEngine defaults (v0.4.0) ---------------------------------------
#
# The user-facing requirement is "中性默认调用 / 模糊即查": ambiguity, the 200 ms
# deadline, and internal errors all resolve to ``retrieve``.  The pre-research
# document listed ``bypass_main`` as the timeout default; the approved task
# overrides that on purpose, so the defaults here are ``retrieve``.
DEFAULT_ROUTER_BUDGET_MS: Final[int] = 200
DEFAULT_ROUTER_THRESHOLD: Final[float] = 0.55
DEFAULT_ROUTER_UNCERTAINTY_BAND: Final[float] = 0.10
DEFAULT_ROUTER_TIMEOUT_POLICY: Final[str] = "retrieve"
DEFAULT_ROUTER_ERROR_POLICY: Final[str] = "fallback_regex"
DEFAULT_ROUTER_INTERNAL_POLICY: Final[str] = "retrieve"
DEFAULT_ROUTER_LOCAL_BACKEND: Final[str] = "bge-head"
DEFAULT_ROUTER_WARMUP_TIMEOUT_MS: Final[int] = 30_000
ROUTER_MODES: Final[tuple[str, ...]] = ("auto", "cloud", "local", "regex")
ROUTER_POLICIES: Final[tuple[str, ...]] = ("bypass_main", "retrieve", "skip")
ROUTER_DECISIONS: Final[tuple[str, ...]] = ("retrieve", "skip")
ROUTER_LOCAL_BACKENDS: Final[tuple[str, ...]] = ("bge-head", "onnx-intent", "off")


class ConfigError(ValueError):
    """Raised when the bridge config or manifest is invalid.

    中文说明：桥接配置或 manifest 内容非法时抛出。
    """


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Validated RouterEngine settings (v0.4.0).

    The Node host owns arbitration; this object only carries the values the
    Python bridge needs to build its local head and report status.

    中文说明：RouterEngine 配置。仲裁在 Node 侧完成，这里只保存本地头所需的
    参数与状态上报字段。
    """

    enabled: bool = True
    mode: str = "auto"
    budget_ms: int = DEFAULT_ROUTER_BUDGET_MS
    threshold: float = DEFAULT_ROUTER_THRESHOLD
    uncertainty_band: float = DEFAULT_ROUTER_UNCERTAINTY_BAND
    timeout_policy: str = DEFAULT_ROUTER_TIMEOUT_POLICY
    error_policy: str = DEFAULT_ROUTER_ERROR_POLICY
    on_internal_error: str = DEFAULT_ROUTER_INTERNAL_POLICY
    shadow_mode: bool = False
    regex_enabled: bool = True
    regex_default_decision: str = "retrieve"
    regex_explicit_fast_path: bool = False
    local_backend: str = DEFAULT_ROUTER_LOCAL_BACKEND
    local_hedge: bool = True
    local_model_dir: str = ""
    local_warmup_timeout_ms: int = DEFAULT_ROUTER_WARMUP_TIMEOUT_MS
    cloud_enabled: bool = False
    cloud_provider: str = ""
    cloud_budget_ms: int = 150
    cache_enabled: bool = True
    cache_ttl_ms: int = 120_000
    cache_max_entries: int = 256
    circuit_failure_threshold: int = 3
    circuit_cooldown_ms: int = 300_000
    audit_enabled: bool = True
    audit_store_query: str = "hash"
    audit_max_bytes: int = 4_194_304

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe nested configuration block."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "budget_ms": self.budget_ms,
            "threshold": self.threshold,
            "uncertainty_band": self.uncertainty_band,
            "timeout_policy": self.timeout_policy,
            "error_policy": self.error_policy,
            "on_internal_error": self.on_internal_error,
            "shadow_mode": self.shadow_mode,
            "regex": {
                "enabled": self.regex_enabled,
                "default_decision": self.regex_default_decision,
                "explicit_fast_path": self.regex_explicit_fast_path,
            },
            "local": {
                "backend": self.local_backend,
                "hedge": self.local_hedge,
                "model_dir": self.local_model_dir,
                "warmup_timeout_ms": self.local_warmup_timeout_ms,
            },
            "cloud": {
                "enabled": self.cloud_enabled,
                "active": self.cloud_provider,
                "budget_ms": self.cloud_budget_ms,
            },
            "cache": {
                "enabled": self.cache_enabled,
                "ttl_ms": self.cache_ttl_ms,
                "max_entries": self.cache_max_entries,
            },
            "circuit": {
                "failure_threshold": self.circuit_failure_threshold,
                "cooldown_ms": self.circuit_cooldown_ms,
            },
            "audit": {
                "enabled": self.audit_enabled,
                "store_query": self.audit_store_query,
                "max_bytes": self.audit_max_bytes,
            },
        }


def _coerce_float(
    value: Any,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(minimum, min(maximum, number))


def _coerce_choice(value: Any, default: str, allowed: tuple[str, ...]) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in allowed:
            return text
    return default


def parse_router_config(merged: Mapping[str, Any]) -> RouterConfig:
    """Build :class:`RouterConfig` from nested ``router`` plus flat ``router_*``.

    The DSH panel projects flat fields (``router_enabled`` …) while the bridge
    file stores the nested object; accepting both keeps old/new bridge files
    working without a migration.

    中文说明：同时接受嵌套 ``router`` 与扁平 ``router_*`` 字段，兼容不同版本
    的桥接文件，无需迁移。
    """
    nested = merged.get("router")
    source: dict[str, Any] = dict(nested) if isinstance(nested, Mapping) else {}
    regex = source.get("regex") if isinstance(source.get("regex"), Mapping) else {}
    local = source.get("local") if isinstance(source.get("local"), Mapping) else {}
    cloud = source.get("cloud") if isinstance(source.get("cloud"), Mapping) else {}
    cache = source.get("cache") if isinstance(source.get("cache"), Mapping) else {}
    circuit = source.get("circuit") if isinstance(source.get("circuit"), Mapping) else {}
    audit = source.get("audit") if isinstance(source.get("audit"), Mapping) else {}

    def pick(flat_key: str, nested_value: Any, default: Any) -> Any:
        """Flat UI field wins over the nested value when present."""
        if flat_key in merged:
            return merged[flat_key]
        return default if nested_value is None else nested_value

    return RouterConfig(
        enabled=_coerce_bool(pick("router_enabled", source.get("enabled"), True), True),
        mode=_coerce_choice(
            pick("router_mode", source.get("mode"), "auto"), "auto", ROUTER_MODES
        ),
        budget_ms=_coerce_int(
            pick("router_budget_ms", source.get("budget_ms"), DEFAULT_ROUTER_BUDGET_MS),
            DEFAULT_ROUTER_BUDGET_MS,
            minimum=50,
            maximum=200,
        ),
        threshold=_coerce_float(
            pick("router_threshold", source.get("threshold"), DEFAULT_ROUTER_THRESHOLD),
            DEFAULT_ROUTER_THRESHOLD,
            minimum=0.0,
            maximum=1.0,
        ),
        uncertainty_band=_coerce_float(
            pick(
                "router_uncertainty_band",
                source.get("uncertainty_band"),
                DEFAULT_ROUTER_UNCERTAINTY_BAND,
            ),
            DEFAULT_ROUTER_UNCERTAINTY_BAND,
            minimum=0.0,
            maximum=0.3,
        ),
        timeout_policy=_coerce_choice(
            pick(
                "router_timeout_policy",
                source.get("timeout_policy"),
                DEFAULT_ROUTER_TIMEOUT_POLICY,
            ),
            DEFAULT_ROUTER_TIMEOUT_POLICY,
            ROUTER_POLICIES,
        ),
        error_policy=_coerce_choice(
            pick("router_error_policy", source.get("error_policy"), DEFAULT_ROUTER_ERROR_POLICY),
            DEFAULT_ROUTER_ERROR_POLICY,
            ROUTER_POLICIES,
        ),
        on_internal_error=_coerce_choice(
            pick(
                "router_on_internal_error",
                source.get("on_internal_error"),
                DEFAULT_ROUTER_INTERNAL_POLICY,
            ),
            DEFAULT_ROUTER_INTERNAL_POLICY,
            ROUTER_POLICIES,
        ),
        shadow_mode=_coerce_bool(
            pick("router_shadow_mode", source.get("shadow_mode"), False), False
        ),
        regex_enabled=_coerce_bool(regex.get("enabled"), True),
        regex_default_decision=_coerce_choice(
            regex.get("default_decision"), "retrieve", ROUTER_DECISIONS
        ),
        regex_explicit_fast_path=_coerce_bool(regex.get("explicit_fast_path"), False),
        local_backend=_coerce_choice(
            pick("router_local_backend", local.get("backend"), DEFAULT_ROUTER_LOCAL_BACKEND),
            DEFAULT_ROUTER_LOCAL_BACKEND,
            ROUTER_LOCAL_BACKENDS,
        ),
        local_hedge=_coerce_bool(
            pick("router_local_hedge", local.get("hedge"), True), True
        ),
        local_model_dir=str(local.get("model_dir") or "").strip(),
        local_warmup_timeout_ms=_coerce_int(
            local.get("warmup_timeout_ms"),
            DEFAULT_ROUTER_WARMUP_TIMEOUT_MS,
            minimum=1_000,
            maximum=120_000,
        ),
        cloud_enabled=_coerce_bool(
            pick("router_cloud_enabled", cloud.get("enabled"), False), False
        ),
        cloud_provider=str(
            pick("router_cloud_provider", cloud.get("active"), "") or ""
        ).strip(),
        cloud_budget_ms=_coerce_int(
            cloud.get("budget_ms"), 150, minimum=40, maximum=180
        ),
        cache_enabled=_coerce_bool(cache.get("enabled"), True),
        cache_ttl_ms=_coerce_int(
            cache.get("ttl_ms"), 120_000, minimum=0, maximum=600_000
        ),
        cache_max_entries=_coerce_int(
            cache.get("max_entries"), 256, minimum=1, maximum=4_096
        ),
        circuit_failure_threshold=_coerce_int(
            circuit.get("failure_threshold"), 3, minimum=1, maximum=10
        ),
        circuit_cooldown_ms=_coerce_int(
            circuit.get("cooldown_ms"), 300_000, minimum=1_000, maximum=1_800_000
        ),
        audit_enabled=_coerce_bool(audit.get("enabled"), True),
        audit_store_query=_coerce_choice(
            audit.get("store_query"), "hash", ("hash", "full", "none")
        ),
        audit_max_bytes=_coerce_int(
            audit.get("max_bytes"), 4_194_304, minimum=65_536, maximum=67_108_864
        ),
    )


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Validated runtime configuration for the memory plugin.

    Attributes:
        enable_memory: Whether long-term memory is enabled.
        rag_path: Absolute path to the Chroma persistence directory.
        max_window_tokens: Token budget for :class:`WindowManager`.
        retrieval_top_k: Number of RAG hits returned by ``memory_search``.

    中文说明：
        记忆插件校验后的运行时配置，包含记忆开关、Chroma 绝对路径、滑动窗口 Token
        预算和 RAG 返回数量。
    """

    enable_memory: bool = True
    enable_auto_recall: bool = True
    rag_path: Path = DEFAULT_RAG_PATH
    max_window_tokens: int = DEFAULT_MAX_WINDOW_TOKENS
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K
    storage_backend: str = DEFAULT_STORAGE_BACKEND
    embedding_engine: str = DEFAULT_EMBEDDING_ENGINE
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    auto_rebuild_collection: bool = True
    holographic_mode: bool = False
    router: RouterConfig = field(default_factory=RouterConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary with an absolute ``rag_path``.

        中文说明：返回可 JSON 序列化的配置字典，``rag_path`` 为绝对路径。
        """
        return {
            "enable_memory": self.enable_memory,
            "enable_auto_recall": self.enable_auto_recall,
            "rag_path": str(self.rag_path),
            "max_window_tokens": self.max_window_tokens,
            "retrieval_top_k": self.retrieval_top_k,
            "storage_backend": self.storage_backend,
            "embedding_engine": self.embedding_engine,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_api_key": self.embedding_api_key,
            "embedding_base_url": self.embedding_base_url,
            "auto_rebuild_collection": self.auto_rebuild_collection,
            "holographic_mode": self.holographic_mode,
            "router": self.router.to_dict(),
        }


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the bridge-config location without reading it.

    中文说明：解析桥接配置文件路径，但不在此时读取文件。
    """
    if config_path is not None:
        return Path(config_path).expanduser().resolve()
    # The Host writes DSH_DREAM_CONFIG for the current profile; the legacy
    # AGENT_MEMORY_CONFIG remains a fallback for direct library use only.
    for env_name in (DSH_CONFIG_ENV, CONFIG_ENV):
        env_path = os.getenv(env_name, "").strip()
        if env_path:
            return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def _manifest_defaults() -> dict[str, Any]:
    """Read defaults from ``manifest.json`` when present.

    中文说明：读取项目根目录 ``manifest.json`` 中的字段默认值。
    """
    if not MANIFEST_PATH.exists():
        return {}
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {MANIFEST_PATH}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("manifest.json must contain a JSON object")
    config = raw.get("config")
    if not isinstance(config, Mapping):
        return {}
    properties = config.get("properties")
    if not isinstance(properties, Mapping):
        return {}
    return {
        key: value.get("default")
        for key, value in properties.items()
        if isinstance(value, Mapping) and "default" in value
    }


def _read_config_file(path: Path) -> dict[str, Any]:
    """Read one bridge-config JSON object, if it exists.

    中文说明：读取一个 JSON 桥接配置文件；文件不存在时返回空字典。
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} must contain a JSON object")
    return dict(raw)


def _coerce_text(value: Any, default: str) -> str:
    """Trim a user-entered string, falling back on invalid values."""
    if isinstance(value, str):
        return value.strip() or default
    return default


def _coerce_provider(value: Any) -> str:
    """Normalize the only two supported embedding providers."""
    if isinstance(value, str):
        provider = value.strip().lower()
        if provider in {"local", "openai_compatible"}:
            return provider
    return DEFAULT_EMBEDDING_PROVIDER


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return default


def _coerce_int(
    value: Any,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _coerce_storage_backend(value: object) -> str:
    text = str(value or DEFAULT_STORAGE_BACKEND).strip().lower()
    if text not in {"sqlite-vec", "chroma"}:
        raise ConfigError(
            "storage_backend must be 'sqlite-vec' or 'chroma'"
        )
    return text


def _coerce_embedding_engine(value: object) -> str:
    text = str(value or DEFAULT_EMBEDDING_ENGINE).strip().lower()
    if text not in {"onnx", "sentence-transformers"}:
        raise ConfigError(
            "embedding_engine must be 'onnx' or 'sentence-transformers'"
        )
    return text


def load_config(config_path: str | Path | None = None) -> MemoryConfig:
    """Load and validate the effective memory configuration.

    Args:
        config_path: Optional explicit bridge-config path.

    Returns:
        A validated :class:`MemoryConfig` with an absolute ``rag_path``.

    Raises:
        ConfigError: If a present manifest or bridge file is invalid JSON or not
            a JSON object.

    中文说明：
        加载并校验有效配置。桥接文件按显式路径、``DSH_DREAM_CONFIG``、
        ``AGENT_MEMORY_CONFIG``、项目根目录 ``agent_memory_config.json``、
        manifest 默认值解析。``rag_path`` 必须由 DSH UI/桥接配置或
        ``AGENT_MEMORY_RAG_PATH`` 显式提供；不再回退到任何隐式 ``.chroma``。
    """
    merged: dict[str, Any] = {}
    merged.update(_manifest_defaults())
    bridge_values = _read_config_file(resolve_config_path(config_path))
    merged.update(bridge_values)

    ui_rag = str(merged.get("rag_path") or "").strip()
    env_rag = os.environ.get("AGENT_MEMORY_RAG_PATH", "").strip()
    rag_text = ui_rag or env_rag
    if not rag_text:
        raise ConfigError(
            "rag_path is not configured; set it in the DSH settings panel "
            "or AGENT_MEMORY_RAG_PATH, then restart DSH."
        )
    rag_path = Path(rag_text).expanduser()
    if not rag_path.is_absolute():
        raise ConfigError(
            "rag_path must be an absolute path; refusing implicit fallback "
            f"to {PROJECT_ROOT / rag_path}."
        )
    rag_path = rag_path.resolve()

    return MemoryConfig(
        enable_memory=_coerce_bool(merged.get("enable_memory"), True),
        enable_auto_recall=_coerce_bool(
            merged.get("enable_auto_recall"),
            True,
        ),
        rag_path=rag_path,
        max_window_tokens=_coerce_int(
            merged.get("max_window_tokens"),
            DEFAULT_MAX_WINDOW_TOKENS,
            minimum=128,
            maximum=MAX_WINDOW_TOKENS,
        ),
        retrieval_top_k=_coerce_int(
            merged.get("retrieval_top_k"),
            DEFAULT_RETRIEVAL_TOP_K,
            minimum=1,
            maximum=MAX_TOP_K,
        ),
        storage_backend=_coerce_storage_backend(
            merged.get("storage_backend"),
        ),
        embedding_engine=_coerce_embedding_engine(
            merged.get("embedding_engine"),
        ),
        embedding_provider=_coerce_provider(merged.get("embedding_provider")),
        embedding_model=_coerce_text(
            merged.get("embedding_model"),
            DEFAULT_EMBEDDING_MODEL,
        ),
        embedding_api_key=_coerce_text(
            merged.get("embedding_api_key"),
            DEFAULT_EMBEDDING_API_KEY,
        ),
        embedding_base_url=_coerce_text(
            merged.get("embedding_base_url"),
            DEFAULT_EMBEDDING_BASE_URL,
        ),
        auto_rebuild_collection=_coerce_bool(
            merged.get("auto_rebuild_collection"),
            True,
        ),
        holographic_mode=_coerce_bool(
            merged.get("holographic_mode"),
            False,
        ),
        # Read router settings from the bridge file only: manifest UI defaults
        # are flat `router_*` keys that must not shadow a nested `router` block.
        router=parse_router_config(bridge_values),
    )


def load_config_dict(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return the effective configuration as a JSON-safe dictionary.

    中文说明：返回有效配置的 JSON 字典，便于工具或前端状态展示。
    """
    return load_config(config_path).to_dict()


__all__ = (
    "CONFIG_ENV",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_EMBEDDING_API_KEY",
    "DEFAULT_EMBEDDING_BASE_URL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_PROVIDER",
    "DEFAULT_MAX_WINDOW_TOKENS",
    "DEFAULT_RAG_PATH",
    "DEFAULT_RETRIEVAL_TOP_K",
    "DEFAULT_ROUTER_BUDGET_MS",
    "DEFAULT_ROUTER_ERROR_POLICY",
    "DEFAULT_ROUTER_INTERNAL_POLICY",
    "DEFAULT_ROUTER_LOCAL_BACKEND",
    "DEFAULT_ROUTER_THRESHOLD",
    "DEFAULT_ROUTER_TIMEOUT_POLICY",
    "DEFAULT_ROUTER_UNCERTAINTY_BAND",
    "DEFAULT_ROUTER_WARMUP_TIMEOUT_MS",
    "MANIFEST_PATH",
    "MAX_TOP_K",
    "MAX_WINDOW_TOKENS",
    "MemoryConfig",
    "PROJECT_ROOT",
    "ROUTER_DECISIONS",
    "ROUTER_LOCAL_BACKENDS",
    "ROUTER_MODES",
    "ROUTER_POLICIES",
    "RouterConfig",
    "load_config",
    "load_config_dict",
    "parse_router_config",
    "resolve_config_path",
)
