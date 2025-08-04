import io
import os
import tempfile
import json
import openai
import requests
import math
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from PIL import Image
import pytesseract
from models import RAGRequest
from utils import retrieve_relevant_docs, build_augmented_prompt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LIMPSAnalysis(BaseModel):
    projection: Optional[str] = None
    structure: Optional[str] = None
    rank: Optional[int] = None
    compressed: Optional[bool] = None

class ChunkMetadata(BaseModel):
    chunk: str
    entropy: float
    token_count: int
    limps_analysis: Optional[LIMPSAnalysis] = None

class ChunkedResponse(BaseModel):
    chunks: List[ChunkMetadata]
    original_text: Optional[str] = None

class TransformRequest(BaseModel):
    text: str
    instruction: str
    use_ai: Optional[bool] = False

def extract_text(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower()
    if ext == ".txt":
        return file.file.read().decode(errors="ignore")
    elif ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp.flush()
            return extract_pdf_text(tmp.name)
    elif ext == ".docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.file.read())
            tmp.flush()
            doc = Document(tmp.name)
            return "\n".join([p.text for p in doc.paragraphs])
    elif ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
        image = Image.open(io.BytesIO(file.file.read()))
        return pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file type")

def byte_tokenize(text: str, chunk_size: int = 512, overlap_percent: float = 0.15) -> List[str]:
    if not text:
        return []
    raw_bytes = text.encode("utf-8")
    overlap = int(chunk_size * overlap_percent)
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(raw_bytes), step):
        chunk = raw_bytes[i:i + chunk_size]
        if not chunk:
            break
        chunks.append(chunk.decode("utf-8", errors="ignore"))
    return chunks

def calculate_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum(p * math.log2(p) for p in prob)

def analyze_with_limps(chunk: str) -> LIMPSAnalysis:
    try:
        resp = requests.post("http://localhost:8001/optimize", json={"text": chunk})
        if resp.status_code == 200:
            data = resp.json()
            return LIMPSAnalysis(**data)
    except:
        pass
    return LIMPSAnalysis()

def fallback_transform(text: str, instruction: str) -> str:
    if "markdown" in instruction.lower():
        return "\n".join([f"- {line}" for line in text.splitlines() if line.strip()])
    elif "uppercase" in instruction.lower():
        return text.upper()
    elif "lowercase" in instruction.lower():
        return text.lower()
    else:
        return f"[Unrecognized instruction: {instruction}]\n\n{text}"

async def ai_transform(text: str, instruction: str) -> str:
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        prompt = f"Transform the following text according to this instruction: '{instruction}'\n\n{text}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a powerful text transformation engine."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI Transform Failed: {str(e)}]\n\n{text}"

@app.post("/upload", response_model=ChunkedResponse)
def upload_and_analyze(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        raw_chunks = byte_tokenize(text)

        enriched_chunks = []
        for chunk in raw_chunks:
            entropy = calculate_entropy(chunk)
            limps = analyze_with_limps(chunk)
            enriched_chunks.append(ChunkMetadata(
                chunk=chunk,
                entropy=entropy,
                token_count=len(chunk.encode("utf-8")),
                limps_analysis=limps
            ))

        return ChunkedResponse(chunks=enriched_chunks, original_text=text)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/transform")
async def transform(request: TransformRequest):
    try:
        if request.use_ai:
            transformed = await ai_transform(request.text, request.instruction)
        else:
            transformed = fallback_transform(request.text, request.instruction)
        return {"result": transformed}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/rag")
async def run_rag(request: RAGRequest):
    """
    RAG endpoint that retrieves relevant polynomial documents and provides augmented answers
    """
    try:
        # Retrieve relevant documents using polynomial analysis
        context = retrieve_relevant_docs(
            request.query, 
            k=request.top_k,
            use_polynomial_analysis=request.use_polynomial_analysis
        )
        
        # Build augmented prompt with context
        prompt = build_augmented_prompt(request.query, context)
        
        # Generate response using OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a mathematical assistant specializing in polynomial analysis. Use the provided polynomial documents to give accurate, detailed answers about polynomial properties, operations, and relationships."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "context": context,
            "query": request.query,
            "top_k": request.top_k
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/export/jsonl")
def export_as_jsonl(chunks: List[str] = Form(...)):
    try:
        jsonl = "\n".join([f'{{"text": "{chunk.replace("\\", "\\\\").replace('"', '\\"')}"}}' for chunk in chunks])
        return JSONResponse(content={"jsonl": jsonl})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/export/txt")
def export_as_txt(chunks: List[str] = Form(...)):
    try:
        txt = "\n\n".join(chunks)
        return JSONResponse(content={"text": txt})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)