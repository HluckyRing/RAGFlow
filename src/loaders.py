import os
import io
import csv
from pypdf import PdfReader
from src.config import logger

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False


def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


# 依次尝试的编码：utf-8-sig 处理 Excel 导出的带 BOM 文件；latin-1 永不抛错，作兜底。
ENCODINGS = ('utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'big5', 'latin-1')


def _decode_bytes(content):
    """按常见中文编码依次尝试解码，保证不抛 UnicodeDecodeError。"""
    for encoding in ENCODINGS:
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode('utf-8', errors='replace')


def load_txt(uploaded_file):
    return _decode_bytes(uploaded_file.read())


def load_docx(uploaded_file):
    if not HAS_DOCX:
        raise ImportError("请安装 python-docx: pip install python-docx")
    doc = Document(uploaded_file)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)
    return "\n".join(paragraphs)


def load_xlsx(uploaded_file):
    if not HAS_OPENPYXL:
        raise ImportError("请安装 openpyxl: pip install openpyxl")
    wb = openpyxl.load_workbook(uploaded_file, read_only=True, data_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"=== {sheet_name} ===")
        for row in ws.iter_rows(values_only=True):
            row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
            if row_text.strip():
                parts.append(row_text)
    wb.close()
    return "\n".join(parts)


def _pptx_shape_text(shape, parts):
    """递归收集一个形状里的文字。

    组合形状（group）必须下钻，否则组合里的文字会整块丢掉。
    """
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in shape.shapes:
            _pptx_shape_text(child, parts)
        return
    if shape.has_table:
        for row in shape.table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)
        return
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)


def load_pptx(uploaded_file):
    if not HAS_PPTX:
        raise ImportError("请安装 python-pptx: pip install python-pptx")
    prs = Presentation(uploaded_file)
    pages = []
    for index, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            _pptx_shape_text(shape, parts)
        # 备注往往才是真正的讲稿，比页面上露出的那几行字更有问答价值
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[备注] {notes}")
        # 整页没文字时连页头一起省掉：纯图片的 PPT 要和既有的「文件内容为空」判定一致，
        # 否则只会索引进去一串 "=== 第 N 页 ===" 这种没有信息量的东西。
        if parts:
            pages.append(f"=== 第 {index} 页 ===\n" + "\n".join(parts))
    return "\n\n".join(pages)


def load_csv(uploaded_file):
    content = _decode_bytes(uploaded_file.read())
    reader = csv.reader(io.StringIO(content))
    rows = [" | ".join(row) for row in reader if any(cell.strip() for cell in row)]
    return "\n".join(rows)


LOADERS = {
    '.pdf': load_pdf,
    '.txt': load_txt,
    '.md': load_txt,
    '.docx': load_docx,
    '.xlsx': load_xlsx,
    '.pptx': load_pptx,
    '.pptm': load_pptx,
    '.csv': load_csv,
}

SUPPORTED_EXTENSIONS = list(LOADERS.keys())


def _unwrap(uploaded_file):
    if hasattr(uploaded_file, 'file') and hasattr(uploaded_file.file, 'seek'):
        f = uploaded_file.file
        f.seek(0)
        return f
    uploaded_file.seek(0)
    return uploaded_file


def load_file(uploaded_file):
    filename = getattr(uploaded_file, 'name', None) or getattr(uploaded_file, 'filename', '')
    ext = os.path.splitext(filename)[1].lower()
    # 旧版 .ppt 是二进制格式，python-pptx 读不了：给一条能照着做的提示，而不是
    # 笼统的「不支持的文件格式」。注意别把它塞进 LOADERS —— 下一句报错里的
    # 「支持: …」是按 LOADERS.keys() 生成的，塞进去等于把不支持的列成支持的。
    if ext == '.ppt':
        raise ValueError("旧版 .ppt 二进制格式不支持，请在 PowerPoint/WPS 里「另存为」.pptx 后重新上传")
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"不支持的文件格式: {ext or '未知'}，支持: {', '.join(SUPPORTED_EXTENSIONS)}")
    logger.info("加载文件: %s (%s)", filename, ext)
    return loader(_unwrap(uploaded_file))
