# -*- coding: utf-8 -*-
"""拆问（query 分解）的解析与降级纪律。

这里只测**纯函数与失败路径**：真模型好不好用只能靠 `scripts/eval_decomposition.py`
真跑（而且要按「拆出来的子问去检索，看 gold 目标片还在不在」那种机械口径算，
不能用词面相似度去猜 —— 本项目在这上面栽过三次）。

要钉住的东西：
- 模型输出脏（围栏、序号、项目符号、引号、多余解释）时还能解析；
- 单问绝不被拆、也绝不被改写成同义句（改写只会让检索变差）；
- 任何故障（缺 key、抛错、空内容、解析不出）都退回 `[question]`，并且**出声**；
- 廉价闸门先挡一层，不是复合问句就别花那 0.75 秒。
"""
import json
from types import SimpleNamespace

import pytest

import src.decompose as decompose_mod
from src.decompose import (
    MAX_PARTS,
    decompose_question,
    looks_compound,
    parse_parts,
)


class FakeClient:
    """替身：记下每次调用，按需返回内容或抛错。"""

    def __init__(self, content="", error=None):
        self.content = content
        self.error = error
        self.calls = []
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        self.response = SimpleNamespace(choices=[choice])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def fake_client(monkeypatch):
    def _install(content="", error=None):
        client = FakeClient(content=content, error=error)
        monkeypatch.setattr(decompose_mod, "client", client)
        return client
    return _install


# ── 解析：模型输出干净时 ──

def test_parse_parts_reads_a_json_array():
    assert parse_parts('["年假有几天", "迟到怎么算"]') == ["年假有几天", "迟到怎么算"]


def test_parse_parts_strips_a_code_fence():
    assert parse_parts('```json\n["年假有几天", "迟到怎么算"]\n```') == ["年假有几天", "迟到怎么算"]


def test_parse_parts_reads_an_object_with_sub_questions():
    raw = json.dumps({"sub_questions": ["年假有几天", "迟到怎么算"], "count": 2},
                     ensure_ascii=False)
    assert parse_parts(raw) == ["年假有几天", "迟到怎么算"]


# ── 解析：模型输出脏时 ──

def test_parse_parts_falls_back_to_lines_when_json_fails():
    raw = "1. 年假有几天\n2. 迟到怎么算\n3. 加班能折算成调休吗"
    assert parse_parts(raw) == ["年假有几天", "迟到怎么算", "加班能折算成调休吗"]


def test_parse_parts_strips_bullets_quotes_and_blank_lines():
    raw = '- 「年假有几天」\n\n* 迟到怎么算\n• “加班能折算成调休吗”'
    assert parse_parts(raw) == ["年假有几天", "迟到怎么算", "加班能折算成调休吗"]


def test_parse_parts_drops_duplicates_and_degenerate_entries():
    raw = '["年假有几天", "年假有几天", "", "。", "' + "很长" * 80 + '"]'
    assert parse_parts(raw) == ["年假有几天"]


def test_parse_parts_caps_the_number_of_parts():
    raw = json.dumps([f"第{i}个问题是什么" for i in range(10)], ensure_ascii=False)
    assert len(parse_parts(raw)) == MAX_PARTS
    assert len(parse_parts(raw, max_parts=2)) == 2


def test_parse_parts_of_garbage_is_empty():
    assert parse_parts("") == []
    assert parse_parts("   \n  ") == []
    assert parse_parts("我不知道该怎么拆") == ["我不知道该怎么拆"]


# ── 廉价闸门：不是复合问句就别花钱 ──

@pytest.mark.parametrize("question", [
    "年假有几天？迟到怎么算？",
    "年假有几天，迟到怎么算",
    "我想知道年假有几天还有迟到怎么算",
    "备份数据用什么加密算法以及全量备份保留多久",
    "出差住宿一线城市限额多少，境外城市呢？",   # 省略式追问：第二问只有「呢」
])
def test_looks_compound_says_yes_to_real_compounds(question):
    assert looks_compound(question) is True


@pytest.mark.parametrize("question", [
    "年假有几天",
    "公司为什么把数据库迁移安排到夜间",
    "报销金额超过 1000 元，要附什么？",
    "今天天气怎么样",
    "年假有几天呢",          # 单问带「呢」，不该被当成两问
    "",
])
def test_looks_compound_says_no_to_single_questions(question):
    assert looks_compound(question) is False


# ── 主函数：正常路径 ──

def test_single_question_never_calls_the_model(fake_client):
    client = fake_client('["年假有几天"]')

    assert decompose_question("年假有几天") == ["年假有几天"]
    assert client.calls == [], "单问连闸门都过不去，一次 API 都不该花"


def test_compound_question_returns_the_model_parts(fake_client):
    client = fake_client('["年假有几天", "迟到怎么算", "加班能折算成调休吗"]')

    parts = decompose_question("年假有几天？迟到怎么算？加班能折算成调休吗？")

    assert parts == ["年假有几天", "迟到怎么算", "加班能折算成调休吗"]
    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent["temperature"] == 0
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}, \
        "推理模型不关思考会把小预算吃光、返回空内容"


def test_one_part_result_returns_the_original_question_not_the_paraphrase(fake_client):
    fake_client('["年假可以休几天呢？"]')

    assert decompose_question("年假有几天？") == ["年假有几天？"], \
        "只拆出一条时用原问句 —— 模型的同义改写只会让检索变差"


def test_gate_can_be_bypassed_for_measurement(fake_client):
    client = fake_client('["年假有几天", "迟到怎么算"]')

    parts = decompose_question("年假有几天，迟到怎么算", use_gate=False)

    assert parts == ["年假有几天", "迟到怎么算"]
    assert len(client.calls) == 1


# ── 失败路径：一律退回原问句，而且要出声 ──

def test_missing_client_falls_back_to_the_question(monkeypatch, caplog):
    monkeypatch.setattr(decompose_mod, "client", None)

    assert decompose_question("年假有几天？迟到怎么算？") == ["年假有几天？迟到怎么算？"]
    assert "跳过拆问" in caplog.text


def test_request_error_falls_back_to_the_question(fake_client, caplog):
    fake_client(error=RuntimeError("boom"))

    assert decompose_question("年假有几天？迟到怎么算？") == ["年假有几天？迟到怎么算？"]
    assert "拆问失败" in caplog.text


def test_empty_content_falls_back_to_the_question(fake_client, caplog):
    fake_client("   ")

    assert decompose_question("年假有几天？迟到怎么算？") == ["年假有几天？迟到怎么算？"]
    assert "空内容" in caplog.text


def test_unparsable_content_falls_back_to_the_question(fake_client, caplog):
    fake_client("```json\n[这里不是 JSON\n```")

    assert decompose_question("年假有几天？迟到怎么算？") == ["年假有几天？迟到怎么算？"]


def test_empty_question_stays_empty(fake_client):
    client = fake_client('["x"]')

    assert decompose_question("") == []
    assert client.calls == []
