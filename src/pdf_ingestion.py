import re
import os
from pypdf import PdfReader
from src.config import logger, CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


def load_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def load_pdf_from_directory(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        return None, f"目录不存在: {data_dir}"
    files = os.listdir(data_dir)
    pdf_files = [f for f in files if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None, "目录中没有 PDF 文件"
    try:
        reader = PdfReader(os.path.join(data_dir, pdf_files[0]))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            return None, "PDF 内容为空或扫描版"
        return text, None
    except Exception as e:
        return None, f"读取 PDF 失败: {e}"


def split_text(text, chunk_size=None, overlap=None):
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk = overlap_text + para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
