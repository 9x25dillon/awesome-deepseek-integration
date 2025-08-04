# Polynomial RAG System

A Retrieval-Augmented Generation (RAG) system specialized for polynomial analysis and mathematical document processing.

## 🚀 Features

- **RAG Endpoint**: Query polynomial documents with AI-powered responses
- **Julia Integration**: Advanced polynomial analysis using Julia backend
- **Document Processing**: Extract text from PDFs, DOCX, images (OCR)
- **Intelligent Chunking**: Byte-level tokenization with overlap
- **Polynomial Analysis**: Extract degree, terms, variables, and complexity
- **Vector Similarity**: Retrieve relevant polynomial documents
- **Fallback Analysis**: Python-based analysis when Julia is unavailable

## 📋 Prerequisites

- Python 3.8+
- Julia (optional, for advanced polynomial analysis)
- OpenAI API key
- Tesseract OCR (for image processing)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd polynomial-rag-system
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Julia (optional)**
   ```bash
   # On Ubuntu/Debian
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
   tar zxvf julia-1.9.3-linux-x86_64.tar.gz
   sudo cp -r julia-1.9.3 /opt/
   sudo ln -s /opt/julia-1.9.3/bin/julia /usr/local/bin/julia
   
   # Verify installation
   julia --version
   ```

4. **Install Tesseract (for OCR)**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   ```

5. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## 🏃 Quick Start

1. **Start the server**
   ```bash
   python main.py
   ```
   The server will start on `http://localhost:8000`

2. **Test the system**
   ```bash
   python test_rag.py
   ```

3. **Access the API documentation**
   Open `http://localhost:8000/docs` in your browser

## 📡 API Endpoints

### `/rag` (POST)
Retrieval-Augmented Generation for polynomial queries

**Request:**
```json
{
  "query": "What is a quadratic polynomial?",
  "top_k": 3,
  "use_polynomial_analysis": true
}
```

**Response:**
```json
{
  "answer": "A quadratic polynomial is a polynomial of degree 2...",
  "context": ["P2: x^2 + y^2 - Quadratic polynomial...", "..."],
  "query": "What is a quadratic polynomial?",
  "top_k": 3,
  "polynomial_analysis_used": true
}
```

### `/analyze_polynomials` (POST)
Direct polynomial analysis

**Request:**
```json
{
  "polynomials": ["x^2 + 3x + 2", "2x^3 - x^2 + 4"],
  "analysis_type": "advanced"
}
```

**Response:**
```json
{
  "results": [
    {
      "expression": "x^2 + 3x + 2",
      "degree": 2,
      "term_count": 3,
      "variables": ["x"],
      "complexity_score": 8.5
    }
  ],
  "analysis_metadata": {
    "julia_backend": true,
    "analysis_type": "advanced"
  }
}
```

### `/julia_status` (GET)
Check Julia backend availability

**Response:**
```json
{
  "julia_available": true,
  "status": "Julia backend is ready"
}
```

### `/upload` (POST)
Upload and analyze documents

**Form Data:**
- `file`: Document file (PDF, DOCX, TXT, images)

**Response:**
```json
{
  "chunks": [
    {
      "chunk": "polynomial content...",
      "entropy": 3.2,
      "token_count": 512,
      "limps_analysis": {...}
    }
  ],
  "original_text": "..."
}
```

### `/transform` (POST)
Transform text with AI or fallback methods

**Request:**
```json
{
  "text": "polynomial expression",
  "instruction": "convert to markdown",
  "use_ai": false
}
```

## 🧮 Polynomial Analysis Features

### Python-based Analysis
- **Degree extraction**: Identifies highest polynomial degree
- **Term counting**: Counts individual polynomial terms  
- **Variable detection**: Extracts unique variables
- **Complexity scoring**: Calculates polynomial complexity
- **Pattern matching**: Uses regex for polynomial parsing

### Julia-based Analysis (Advanced)
- **Symbolic computation**: Advanced mathematical analysis
- **Performance optimization**: Faster processing for complex polynomials
- **Extended features**: More sophisticated polynomial operations
- **Fallback support**: Graceful degradation to Python analysis

## 📚 Polynomial Document Corpus

The system includes a curated corpus of polynomial examples:

- **P1**: `3x + 2y` - Linear polynomial with two variables
- **P2**: `x^2 + y^2` - Quadratic polynomial, sum of squares
- **P3**: `5xy - z` - Bilinear term with linear term
- **P4**: `x^3 + 2x^2 + x + 1` - Cubic polynomial in one variable
- **P5**: `2x^2 + 3xy + y^2` - Quadratic form
- **P6**: `x^4 - 1` - Fourth degree polynomial
- **P7**: `sin(x) + cos(y)` - Non-polynomial functions
- **P8**: `a*x^2 + b*x + c` - General quadratic form
- **P9**: `(x+1)(x-1)` - Factored polynomial
- **P10**: `x^2 + 2xy + y^2` - Perfect square trinomial
- And more...

## 🔍 Retrieval Methods

### 1. Polynomial Similarity Analysis
- Compares degree, variables, term count
- Calculates similarity scores
- Ranks documents by relevance

### 2. Julia-backed Analysis
- Uses Julia for advanced polynomial analysis
- Compares mathematical properties
- Enhanced accuracy for complex queries

### 3. Keyword-based Retrieval
- Fallback for non-polynomial queries
- Uses term frequency analysis
- Keyword matching with polynomial vocabulary

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_rag.py
```

The test suite includes:
- Health checks
- Julia backend status
- RAG query tests
- Polynomial analysis tests
- Performance benchmarks

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for AI-powered responses
- `JULIA_EXECUTABLE`: Path to Julia binary (default: "julia")

### Customization
- Modify `PolynomialCorpus.POLYNOMIAL_DOCS` in `utils.py` to add more documents
- Adjust similarity scoring in `analyze_polynomial_similarity()`
- Configure chunking parameters in `byte_tokenize()`

## 🏗️ Architecture

```
main.py                 # FastAPI application with endpoints
├── models.py          # Pydantic models for requests/responses
├── utils.py           # Retrieval and analysis utilities
├── julia_client.py    # Julia integration for advanced analysis
└── test_rag.py       # Comprehensive test suite
```

### Key Components

1. **RAG Pipeline**: Query → Retrieval → Augmentation → Generation
2. **Polynomial Analysis**: Feature extraction and similarity computation
3. **Julia Integration**: Advanced mathematical computation backend
4. **Document Processing**: Multi-format text extraction
5. **Chunking System**: Intelligent text segmentation

## 🔮 Future Enhancements

- **Vector Database**: Replace static corpus with FAISS/Chroma
- **Symbolic Math**: Enhanced symbolic polynomial manipulation
- **LaTeX Support**: Mathematical notation processing
- **Graph Analysis**: Polynomial relationship visualization
- **Caching**: Redis-based response caching
- **Batch Processing**: Bulk polynomial analysis
- **Web Interface**: React-based frontend

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For questions or issues:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Run the test suite for debugging

---

Built with ❤️ for mathematical document analysis and polynomial research.