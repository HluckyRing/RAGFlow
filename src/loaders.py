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
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"不支持的文件格式: {ext or '未知'}，支持: {', '.join(SUPPORTED_EXTENSIONS)}")
    logger.info("加载文件: %s (%s)", filename, ext)
    return loader(_unwrap(uploaded_file))
