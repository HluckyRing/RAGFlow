# -*- coding: utf-8 -*-
"""pytest 公共配置：把项目根目录加进 sys.path，并提供不依赖真实 API Key 的环境变量。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须在 import src.config 之前设置，否则 OpenAI 客户端初始化会失败
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("BASE_URL", "https://example.invalid")

import pytest


@pytest.fixture(autouse=True)
def _isolate_upload_dir(tmp_path, monkeypatch):
    """所有测试都不许把上传原件写进真实的 uploads/。

    原文件预览要落盘原件，漏掉这层隔离就会在开发者工作区留下用户数据。
    """
    import src.server as server

    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path / "uploads"), raising=False)


@pytest.fixture(autouse=True)
def _never_load_the_reranker(monkeypatch):
    """测试套件绝不加载 reranker 模型（权重 1.1GB、首次还会联网下载）。

    要测重排的用例自己把开关打开、并注入替身（见 tests/test_retrieval.py 的 `_enable_rerank`），
    所以这里一律先关掉，不受 config 默认值影响。
    """
    import src.retrieval as retrieval

    monkeypatch.setattr(retrieval, "RERANK_ENABLED", False)


@pytest.fixture(autouse=True)
def _never_call_the_real_decomposer(monkeypatch):
    """测试套件绝不许真的去调拆问大模型。

    `DECOMPOSE_ENABLED` 2026-11-08 起默认是 **1**（第九轮定案），而 chat 链路会把它读进来 ——
    没有这层隔离，`tests/test_chat_flow.py` 里的每条提问都会去连真实的 API（或卡在
    `example.invalid` 上超时）。要测拆问接线的用例自己在用例里 monkeypatch 开关并注入替身
    （见 `tests/test_decompose_retrieval.py`），不受这里的默认值影响。
    """
    import src.config as config

    monkeypatch.setattr(config, "DECOMPOSE_ENABLED", False)


@pytest.fixture(autouse=True)
def _never_sample_coverage_by_default(monkeypatch):
    """测试套件里默认关掉覆盖取样（两个开关）。

    `COVERAGE_INTRA_FILE` 2026-11-15 起默认 **1**（质量门过了 + 覆盖说明已上线），而它会在
    「枚举措辞 + 点名章号」时**换一条检索路径**（`collection.get` 整章枚举）并往 context 前面
    插一句【覆盖说明】—— 这会让所有走真实检索链路的用例的 context/块数发生变化。
    要测取样接线的用例自己 monkeypatch 开关（见 `tests/test_coverage.py`），不受这里影响。
    """
    import src.llm as llm

    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", False)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", False)


@pytest.fixture(autouse=True)
def _clear_pairs_cache():
    """切片缓存是模块级的：用例之间清一次，免得上一条用例的正文影响这一条。"""
    import src.retrieval as retrieval

    retrieval.clear_pairs_cache()
    yield
    retrieval.clear_pairs_cache()


@pytest.fixture(autouse=True)
def _never_scope_filter_by_default(monkeypatch):
    """测试套件里默认关掉「范围词 → 文件过滤」并清零融合候选池。

    `SCOPE_FILTER` 与 `FUSION_CANDIDATES` 现在都默认关（新行为先默认关），但开发者本机的
    `.env` / 环境变量可能把它们打开 —— 那会让「结合第2章课件，…」这类用例悄悄换一条检索路径
    （多传 `where`、探针文本被改写、候选池变大），与测试意图无关。要测这两条的用例自己
    monkeypatch 开关（见 `tests/test_scope.py`），不受这里影响。
    """
    import src.retrieval as retrieval

    monkeypatch.setattr(retrieval, "SCOPE_FILTER", False)
    monkeypatch.setattr(retrieval, "FUSION_CANDIDATES", 0)


@pytest.fixture(autouse=True)
def _never_rate_limit_chat_requests(monkeypatch):
    """测试套件里不限流。

    `CHAT_RATE_LIMIT` 默认 20 次/分钟（按客户端 IP），而 `TestClient` 的 client.host 恒为
    `testclient` —— 一套用例里连着打几十次 `/api/chat` 会集体撞 429（实测挂了 7 条
    test_refusal）。要测限流的用例自己注入一个小的限流器（见
    `tests/test_security_runtime.py`），不受这里影响。
    """
    import src.server as server
    from src.security import SlidingWindowLimiter

    monkeypatch.setattr(server, "_chat_limiter", SlidingWindowLimiter(0))


@pytest.fixture(autouse=True)
def _never_require_auth_by_default(monkeypatch):
    """测试里默认**不要** `AUTH_TOKEN`：本机 `.env` 里一旦有真令牌，整个套件会大面积 401。

    为什么必须隔离（2026-11-25 真事故）：`.env` 被 `load_dotenv()` 读进 `config.AUTH_TOKEN`
    之后，鉴权中间件对所有请求要求 HTTP Basic —— 而除 `tests/test_auth.py` 之外没有一个用例
    带凭证。于是「往 `.env` 里填 AUTH_TOKEN 好去跑 Docker」这个**完全正常的操作**会让 87 个
    测试变红，报错还长得像业务 bug（`KeyError: 'session_id'`）。

    要测鉴权的用例自己在用例里 `monkeypatch.setattr(config, "AUTH_TOKEN", "s3cret-token")`
    —— 那发生在**本 fixture 之后**，会正常覆盖；`tests/test_auth.py` 就是这么写的。
    """
    import src.config as config

    monkeypatch.setattr(config, "AUTH_TOKEN", "")
