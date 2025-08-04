import re
import math
import json
from typing import List, Dict, Tuple, Optional
from models import PolynomialDocument
from julia_client import JuliaClient

class PolynomialCorpus:
    """Static corpus of polynomial documents for retrieval"""
    
    POLYNOMIAL_DOCS = [
        "P1: 3x + 2y - Linear polynomial with two variables",
        "P2: x^2 + y^2 - Quadratic polynomial, sum of squares",
        "P3: 5xy - z - Bilinear term with linear term",
        "P4: x^3 + 2x^2 + x + 1 - Cubic polynomial in one variable",
        "P5: 2x^2 + 3xy + y^2 - Quadratic form",
        "P6: x^4 - 1 - Fourth degree polynomial, difference of squares",
        "P7: sin(x) + cos(y) - Trigonometric functions (non-polynomial)",
        "P8: a*x^2 + b*x + c - General quadratic form with parameters",
        "P9: (x+1)(x-1) - Factored quadratic polynomial",
        "P10: x^2 + 2xy + y^2 - Perfect square trinomial",
        "P11: 3x^3 - 2x^2 + x - 5 - General cubic polynomial",
        "P12: x^2*y + xy^2 - Homogeneous polynomial of degree 3",
        "P13: 7x - Monomial linear polynomial",
        "P14: x^5 + x^4 + x^3 + x^2 + x + 1 - Fifth degree polynomial",
        "P15: 2x^2 - 8 - Quadratic with constant term"
    ]

def extract_polynomial_features(expression: str) -> Dict:
    """Extract mathematical features from polynomial expressions"""
    features = {
        "degree": 0,
        "term_count": 0,
        "variables": set(),
        "coefficients": [],
        "has_constant": False,
        "complexity_score": 0.0
    }
    
    # Remove spaces and normalize
    expr = expression.replace(" ", "").replace("-", "+-")
    
    # Find all terms (split by + but handle leading -)
    terms = [term for term in expr.split("+") if term]
    features["term_count"] = len(terms)
    
    max_degree = 0
    for term in terms:
        # Extract variables and their powers
        var_matches = re.findall(r'([a-zA-Z])\^?(\d*)', term)
        term_degree = 0
        
        for var, power in var_matches:
            features["variables"].add(var)
            power_val = int(power) if power else 1
            term_degree += power_val
            
        max_degree = max(max_degree, term_degree)
        
        # Extract coefficient
        coeff_match = re.match(r'^[+-]?\d*\.?\d*', term)
        if coeff_match:
            coeff_str = coeff_match.group()
            if coeff_str and coeff_str not in ['+', '-']:
                try:
                    features["coefficients"].append(float(coeff_str))
                except ValueError:
                    pass
    
    features["degree"] = max_degree
    features["variables"] = list(features["variables"])
    features["has_constant"] = any("x" not in term and "y" not in term and "z" not in term for term in terms)
    
    # Calculate complexity score
    features["complexity_score"] = (
        features["degree"] * 2 + 
        features["term_count"] + 
        len(features["variables"]) * 0.5
    )
    
    return features

def analyze_polynomial_similarity(query: str, documents: List[str]) -> List[Tuple[str, float]]:
    """Analyze similarity between query and polynomial documents"""
    query_features = extract_polynomial_features(query)
    similarities = []
    
    for doc in documents:
        # Extract polynomial expression from document
        poly_expr = doc.split(":")[1].split("-")[0].strip() if ":" in doc else doc
        doc_features = extract_polynomial_features(poly_expr)
        
        # Calculate similarity score
        score = 0.0
        
        # Degree similarity (higher weight)
        if query_features["degree"] == doc_features["degree"]:
            score += 3.0
        elif abs(query_features["degree"] - doc_features["degree"]) <= 1:
            score += 1.5
            
        # Variable overlap
        query_vars = set(query_features["variables"])
        doc_vars = set(doc_features["variables"])
        if query_vars and doc_vars:
            overlap = len(query_vars.intersection(doc_vars)) / len(query_vars.union(doc_vars))
            score += overlap * 2.0
            
        # Term count similarity
        term_diff = abs(query_features["term_count"] - doc_features["term_count"])
        score += max(0, 1.0 - term_diff * 0.2)
        
        # Complexity similarity
        complexity_diff = abs(query_features["complexity_score"] - doc_features["complexity_score"])
        score += max(0, 1.0 - complexity_diff * 0.1)
        
        similarities.append((doc, score))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def retrieve_relevant_docs(query: str, k: int = 3, use_polynomial_analysis: bool = True) -> List[str]:
    """
    Retrieve relevant polynomial documents based on query analysis
    
    Args:
        query: User's query string
        k: Number of top documents to retrieve
        use_polynomial_analysis: Whether to use Julia-backed polynomial analysis
    
    Returns:
        List of relevant document strings
    """
    
    if use_polynomial_analysis:
        try:
            # Try to use Julia client for advanced polynomial analysis
            julia_client = JuliaClient()
            analysis_result = julia_client.analyze_polynomials([query])
            
            if analysis_result and "polynomials" in analysis_result:
                # Use Julia analysis to enhance retrieval
                julia_features = analysis_result["polynomials"][0] if analysis_result["polynomials"] else {}
                
                # Filter documents based on Julia analysis
                filtered_docs = []
                for doc in PolynomialCorpus.POLYNOMIAL_DOCS:
                    doc_analysis = julia_client.analyze_polynomials([doc.split(":")[1].split("-")[0].strip()])
                    if doc_analysis and "polynomials" in doc_analysis:
                        doc_features = doc_analysis["polynomials"][0] if doc_analysis["polynomials"] else {}
                        
                        # Compare Julia-computed features
                        similarity_score = 0.0
                        if julia_features.get("degree") == doc_features.get("degree"):
                            similarity_score += 2.0
                        if julia_features.get("term_count") == doc_features.get("term_count"):
                            similarity_score += 1.0
                            
                        filtered_docs.append((doc, similarity_score))
                
                # Sort by similarity and return top k
                filtered_docs.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in filtered_docs[:k]]
                
        except Exception as e:
            print(f"Julia analysis failed, falling back to heuristic analysis: {e}")
    
    # Fallback to heuristic polynomial analysis
    # Check if query contains polynomial-like patterns
    has_polynomial_pattern = bool(re.search(r'[a-zA-Z]\^?\d*|[+-]?\d*[a-zA-Z]', query))
    
    if has_polynomial_pattern:
        # Use polynomial similarity analysis
        similarities = analyze_polynomial_similarity(query, PolynomialCorpus.POLYNOMIAL_DOCS)
        return [doc for doc, _ in similarities[:k]]
    else:
        # Use keyword-based retrieval for non-polynomial queries
        query_lower = query.lower()
        scored_docs = []
        
        for doc in PolynomialCorpus.POLYNOMIAL_DOCS:
            score = 0.0
            doc_lower = doc.lower()
            
            # Keyword matching
            keywords = ["linear", "quadratic", "cubic", "polynomial", "degree", "variable", "coefficient"]
            for keyword in keywords:
                if keyword in query_lower and keyword in doc_lower:
                    score += 1.0
            
            # Term overlap
            query_terms = set(query_lower.split())
            doc_terms = set(doc_lower.split())
            overlap = len(query_terms.intersection(doc_terms))
            score += overlap * 0.5
            
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

def build_augmented_prompt(query: str, context_docs: List[str]) -> str:
    """
    Build an augmented prompt combining user query with relevant context documents
    
    Args:
        query: User's original query
        context_docs: List of relevant document strings
    
    Returns:
        Formatted prompt string for LLM
    """
    
    context_section = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)])
    
    prompt = f"""Use the following polynomial documents to answer the user's question:

{context_section}

User Question: {query}

Please provide a detailed mathematical answer based on the provided polynomial documents. Include:
1. Direct analysis of relevant polynomials from the documents
2. Mathematical properties and relationships
3. Step-by-step explanations where applicable
4. Any additional insights about polynomial behavior

Answer:"""
    
    return prompt

def extract_polynomial_from_text(text: str) -> Optional[str]:
    """Extract polynomial expressions from free text"""
    # Pattern to match polynomial-like expressions
    poly_pattern = r'[+-]?\d*[a-zA-Z]\^?\d*(?:[+-]\d*[a-zA-Z]\^?\d*)*(?:[+-]\d+)?'
    matches = re.findall(poly_pattern, text)
    return matches[0] if matches else None

def calculate_polynomial_complexity(expression: str) -> float:
    """Calculate a complexity score for a polynomial expression"""
    features = extract_polynomial_features(expression)
    
    complexity = (
        features["degree"] ** 2 +  # Degree has quadratic impact
        features["term_count"] * 1.5 +  # More terms = more complex
        len(features["variables"]) * 2 +  # More variables = more complex
        len(features["coefficients"]) * 0.5  # More coefficients = slightly more complex
    )
    
    return complexity