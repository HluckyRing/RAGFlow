# -*- coding: utf-8 -*-
"""文件加载器回归：中文 CSV 的各种常见编码都要能读进来。"""
import io

import pytest

import src.loaders as loaders


class FakeUpload:
    """模拟 FastAPI 的 UploadFile（构造后交给 _unwrap 处理）。"""

    def __init__(self, data: bytes, name: str):
        self.name = name
        self._buf = io.BytesIO(data)

    def seek(self, *args):
        return self._buf.seek(*args)

    def read(self, *args):
        return self._buf.read(*args)


CSV_CASES = [
    # 简体样本
    ("utf-8", "姓名,分数\n张三,90\n", "姓名 | 分数"),
    ("utf-8-sig", "姓名,分数\n张三,90\n", "姓名 | 分数"),
    ("gbk", "姓名,分数\n张三,90\n", "姓名 | 分数"),
    ("gb2312", "姓名,分数\n张三,90\n", "姓名 | 分数"),
    # 繁体样本：Big5 编不出简体「数」字，所以单独准备一份
    ("big5", "姓名,分數\n張三,90\n", "姓名 | 分數"),
]


@pytest.mark.parametrize("encoding,text,first_line", CSV_CASES)
def test_load_csv_supports_common_encodings(encoding, text, first_line):
    out = loaders.load_file(FakeUpload(text.encode(encoding), "成绩.csv"))
    assert "90" in out
    # BOM 不能粘在第一个单元格上
    assert out.splitlines()[0] == first_line


def test_load_txt_still_supports_gbk():
    out = loaders.load_file(FakeUpload("中文内容\n".encode("gbk"), "a.txt"))
    assert "中文内容" in out


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        loaders.load_file(FakeUpload(b"x", "a.bin"))
