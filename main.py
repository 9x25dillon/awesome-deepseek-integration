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
from models import RAGRequest, RAGResponse, PolynomialAnalysisRequest, PolynomialAnalysisResponse, EntropyProcessingRequest, EntropyProcessingResponse, EntropyRAGRequest
from utils import retrieve_relevant_docs, build_augmented_prompt, extract_polynomial_features
from julia_client import julia_client
from core import Token, EntropyNode, EntropyEngine
from entropy_cli import create_polynomial_transformations

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
        if not openai.api_key:
            return JSONResponse(
                status_code=500, 
                content={"error": "OpenAI API key not configured"}
            )
        
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
            "top_k": request.top_k,
            "polynomial_analysis_used": request.use_polynomial_analysis
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze_polynomials", response_model=PolynomialAnalysisResponse)
async def analyze_polynomials(request: PolynomialAnalysisRequest):
    """
    Endpoint for direct polynomial analysis using Julia backend
    """
    try:
        if julia_client.is_available() and request.analysis_type == "advanced":
            # Use Julia for advanced analysis
            analysis_result = julia_client.analyze_polynomials(request.polynomials)
            if analysis_result:
                return PolynomialAnalysisResponse(
                    results=[
                        {
                            "expression": poly["expression"],
                            "degree": poly["degree"],
                            "term_count": poly["term_count"],
                            "variables": poly["variables"],
                            "complexity_score": poly["complexity_score"]
                        }
                        for poly in analysis_result["polynomials"]
                    ],
                    analysis_metadata={"julia_backend": True, "analysis_type": request.analysis_type}
                )
        
        # Fallback to Python-based analysis
        results = []
        for poly_expr in request.polynomials:
            features = extract_polynomial_features(poly_expr)
            results.append({
                "expression": poly_expr,
                "degree": features["degree"],
                "term_count": features["term_count"],
                "variables": features["variables"],
                "complexity_score": features["complexity_score"]
            })
        
        return PolynomialAnalysisResponse(
            results=results,
            analysis_metadata={"julia_backend": False, "analysis_type": "basic"}
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/julia_status")
def julia_status():
    """Check if Julia backend is available"""
    return {
        "julia_available": julia_client.is_available(),
        "status": "Julia backend is ready" if julia_client.is_available() else "Julia backend unavailable, using Python fallback"
    }

@app.post("/entropy_process", response_model=EntropyProcessingResponse)
async def process_with_entropy_engine(request: EntropyProcessingRequest):
    """
    Process input through entropy engine with polynomial transformations
    """
    try:
        # Create token
        token = Token(request.input_value)
        
        # Get available transformations
        transforms = create_polynomial_transformations()
        
        # Build transformation nodes
        root_node = None
        current_node = None
        
        for i, transform_name in enumerate(request.transformations):
            if transform_name not in transforms:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unknown transformation: {transform_name}"}
                )
            
            # Apply entropy limit if provided
            entropy_limit = None
            if request.entropy_limits and i < len(request.entropy_limits):
                entropy_limit = request.entropy_limits[i]
            
            node = EntropyNode(f"node_{i}_{transform_name}", transforms[transform_name], entropy_limit)
            
            if root_node is None:
                root_node = node
                current_node = node
            else:
                current_node.add_child(node)
                current_node = node
        
        if root_node is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid transformations provided"}
            )
        
        # Process through entropy engine
        engine = EntropyEngine(root_node, max_depth=request.max_depth)
        processed_token = engine.run(token)
        
        # Julia analysis if requested
        julia_analysis = None
        if request.use_julia and julia_client.is_available() and token.token_type == "polynomial":
            try:
                julia_analysis = julia_client.analyze_polynomials([processed_token.current_value])
            except:
                pass
        
        return EntropyProcessingResponse(
            input_value=request.input_value,
            final_value=processed_token.current_value,
            input_type=processed_token.token_type,
            entropy_change=processed_token.current_entropy - processed_token.initial_entropy,
            entropy_trend=processed_token.entropy_trend(),
            transformations_applied=len(processed_token.transformations),
            processing_graph=engine.export_graph(),
            polynomial_features=processed_token.polynomial_features,
            julia_analysis=julia_analysis
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/entropy_rag")
async def entropy_enhanced_rag(request: EntropyRAGRequest):
    """
    Enhanced RAG with entropy-based query processing
    """
    try:
        # First, process the query through entropy engine if transformations specified
        processed_query = request.query
        entropy_analysis = None
        
        if request.entropy_transformations:
            # Process query through entropy engine
            entropy_request = EntropyProcessingRequest(
                input_value=request.query,
                transformations=request.entropy_transformations,
                max_depth=3,  # Limit depth for query processing
                use_julia=True
            )
            
            # Get internal entropy processing response
            entropy_result = await process_with_entropy_engine(entropy_request)
            if hasattr(entropy_result, 'final_value'):
                processed_query = entropy_result.final_value
                entropy_analysis = {
                    "original_query": request.query,
                    "processed_query": processed_query,
                    "entropy_change": entropy_result.entropy_change,
                    "transformations": request.entropy_transformations
                }
        
        # Apply entropy threshold filtering if specified
        context_docs = retrieve_relevant_docs(
            processed_query, 
            k=request.top_k,
            use_polynomial_analysis=request.use_polynomial_analysis
        )
        
        # Filter by entropy threshold if specified
        if request.entropy_threshold is not None:
            filtered_docs = []
            for doc in context_docs:
                doc_token = Token(doc)
                if doc_token.current_entropy >= request.entropy_threshold:
                    filtered_docs.append(doc)
            context_docs = filtered_docs[:request.top_k]
        
        # Build augmented prompt
        augmented_prompt = build_augmented_prompt(processed_query, context_docs)
        
        # Generate AI response
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            return JSONResponse(
                status_code=500, 
                content={"error": "OpenAI API key not configured"}
            )
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a mathematical assistant specializing in polynomial analysis with entropy-aware processing. Use the provided polynomial documents and entropy analysis to give accurate, detailed answers."
                },
                {"role": "user", "content": augmented_prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "context": context_docs,
            "original_query": request.query,
            "processed_query": processed_query,
            "entropy_analysis": entropy_analysis,
            "entropy_threshold_applied": request.entropy_threshold,
            "top_k": len(context_docs),
            "polynomial_analysis_used": request.use_polynomial_analysis
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/entropy_transformations")
def list_entropy_transformations():
    """List available entropy transformations"""
    transforms = create_polynomial_transformations()
    
    polynomial_specific = [
        "expand", "factor", "add_variable", "increase_degree", 
        "substitute", "differentiate", "add_constant", "normalize"
    ]
    
    general_transforms = [
        "reverse", "uppercase", "lowercase", "duplicate", 
        "add_random", "add_entropy", "truncate", "multiply"
    ]
    
    return {
        "polynomial_transformations": [name for name in polynomial_specific if name in transforms],
        "general_transformations": [name for name in general_transforms if name in transforms],
        "total_available": len(transforms)
    }

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