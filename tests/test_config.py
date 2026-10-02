# -*- coding: utf-8 -*-
"""配置层回归：路径锚定（P1-14）与启动/客户端参数（P1-16）。

改造前 VECTOR_DB_PATH 默认 "./chroma_db"，是相对当前工作目录的 —— 换个目录启动
就会另建一个向量库，状态目录同理。这里锁定「相对路径一律锚定项目根」。
"""
import logging
from pathlib import Path

import pytest

import src.config as config
import src.state as state


# ── 路径锚定 ──

def test_relative_path_is_anchored_to_project_root():
    assert config.resolve_path("./chroma_db", "chroma_db") == str(config.PROJECT_ROOT / "chroma_db")


def test_nested_relative_path_is_anchored():
    assert config.resolve_path("data/chroma", "chroma_db") == str(config.PROJECT_ROOT / "data" / "chroma")


def test_absolute_path_is_kept(tmp_path):
    assert config.resolve_path(str(tmp_path / "db"), "chroma_db") == str(tmp_path / "db")


def test_missing_value_falls_back_to_project_root_default():
    assert config.resolve_path(None, "chroma_db") == str(config.PROJECT_ROOT / "chroma_db")
    assert config.resolve_path("", "state") == str(config.PROJECT_ROOT / "state")


def test_vector_db_path_is_absolute_even_after_chdir(monkeypatch, tmp_path):
    """换工作目录启动时，向量库路径不能跟着漂。"""
    monkeypatch.chdir(tmp_path)
    assert Path(config.VECTOR_DB_PATH).is_absolute(), \
        f"VECTOR_DB_PATH 必须是绝对路径，实际 {config.VECTOR_DB_PATH!r}"


def test_state_paths_are_absolute():
    assert Path(state.STATE_DIR).is_absolute()
    assert Path(state.LEGACY_STATE_FILE).is_absolute()
    assert Path(state.PROJECT_ROOT) == config.PROJECT_ROOT, "项目根只该有一处定义"


# ── 启动参数与客户端 ──

def test_upload_limit_defaults_are_consistent():
    assert config.MAX_UPLOAD_MB > 0
    assert config.MAX_UPLOAD_BYTES == config.MAX_UPLOAD_MB * 1024 * 1024


def test_llm_timeout_default_is_positive():
    assert config.LLM_TIMEOUT > 0


def test_client_is_none_when_api_key_missing(caplog):
    """缺 API_KEY 不能抛 SDK 的英文异常把整个服务带崩，要给中文提示并降级。"""
    with caplog.at_level(logging.ERROR, logger="ai_rag"):
        built = config.build_client("", "https://example.invalid", 30.0)
    assert built is None
    assert any("API_KEY" in r.getMessage() for r in caplog.records), "缺 API_KEY 时应给出中文提示"


def test_client_gets_configured_timeout():
    built = config.build_client("sk-test", "https://example.invalid", 12.5)
    assert built is not None
    assert built.timeout == pytest.approx(12.5), "超时必须真的传到 OpenAI 客户端上"


# ── 启动绑定（P1-21）──

def test_host_defaults_to_loopback_and_port_to_8080():
    assert config.HOST == "127.0.0.1", "默认只监听回环，避免无鉴权服务暴露到局域网"
    assert config.PORT == 8080
