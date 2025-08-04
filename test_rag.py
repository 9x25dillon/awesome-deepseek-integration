#!/usr/bin/env python3
"""
Test script for the Polynomial RAG System

This script demonstrates how to use the RAG endpoints to query polynomial documents
and get AI-powered responses with relevant context.
"""

import requests
import json
import time
from typing import Dict, Any

class RAGTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def test_health(self) -> bool:
        """Test if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def test_julia_status(self) -> Dict[str, Any]:
        """Check Julia backend availability"""
        try:
            response = requests.get(f"{self.base_url}/julia_status")
            if response.status_code == 200:
                return response.json()
            return {"error": "Failed to get Julia status"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def test_rag_query(self, query: str, top_k: int = 3, use_polynomial_analysis: bool = True) -> Dict[str, Any]:
        """Test the RAG endpoint with a query"""
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "use_polynomial_analysis": use_polynomial_analysis
            }
            
            response = requests.post(f"{self.base_url}/rag", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def test_polynomial_analysis(self, polynomials: list, analysis_type: str = "basic") -> Dict[str, Any]:
        """Test direct polynomial analysis"""
        try:
            payload = {
                "polynomials": polynomials,
                "analysis_type": analysis_type
            }
            
            response = requests.post(f"{self.base_url}/analyze_polynomials", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_result(result: Dict[str, Any], title: str = "Result"):
    """Print formatted result"""
    print(f"\n{title}:")
    print("-" * 40)
    print(json.dumps(result, indent=2, ensure_ascii=False))

def main():
    """Run the RAG system tests"""
    print("🧮 Polynomial RAG System Test Suite")
    print("=" * 60)
    
    tester = RAGTester()
    
    # Test 1: Health Check
    print_section("1. Health Check")
    if tester.test_health():
        print("✅ Server is running and healthy")
    else:
        print("❌ Server is not responding")
        return
    
    # Test 2: Julia Status
    print_section("2. Julia Backend Status")
    julia_status = tester.test_julia_status()
    print_result(julia_status)
    
    # Test 3: RAG Queries
    print_section("3. RAG Query Tests")
    
    test_queries = [
        "What is a quadratic polynomial?",
        "Explain the difference between linear and cubic polynomials",
        "How do you factor x^2 + 2xy + y^2?",
        "What are the properties of x^4 - 1?",
        "Analyze the polynomial 3x^3 - 2x^2 + x - 5"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        result = tester.test_rag_query(query)
        
        if "error" not in result:
            print(f"📝 Answer: {result.get('answer', 'No answer')[:200]}...")
            print(f"📚 Context Documents ({len(result.get('context', []))}):")
            for j, doc in enumerate(result.get('context', []), 1):
                print(f"  {j}. {doc[:60]}...")
            print(f"🔧 Polynomial Analysis Used: {result.get('polynomial_analysis_used', 'Unknown')}")
        else:
            print(f"❌ Error: {result['error']}")
        
        time.sleep(1)  # Brief pause between queries
    
    # Test 4: Direct Polynomial Analysis
    print_section("4. Direct Polynomial Analysis")
    
    test_polynomials = ["x^2 + 3x + 2", "2x^3 - x^2 + 4", "5xy - 3z + 1"]
    
    print("--- Basic Analysis ---")
    basic_result = tester.test_polynomial_analysis(test_polynomials, "basic")
    print_result(basic_result)
    
    print("\n--- Advanced Analysis (Julia) ---")
    advanced_result = tester.test_polynomial_analysis(test_polynomials, "advanced")
    print_result(advanced_result)
    
    # Test 5: Performance Test
    print_section("5. Performance Test")
    
    start_time = time.time()
    performance_query = "Compare quadratic and cubic polynomials"
    result = tester.test_rag_query(performance_query)
    end_time = time.time()
    
    if "error" not in result:
        print(f"✅ Query processed successfully in {end_time - start_time:.2f} seconds")
        print(f"📊 Retrieved {len(result.get('context', []))} relevant documents")
    else:
        print(f"❌ Performance test failed: {result['error']}")
    
    print_section("Test Summary")
    print("🎉 RAG System test suite completed!")
    print("\nTo use the RAG system:")
    print("1. Start the server: python main.py")
    print("2. Send POST requests to /rag with your polynomial questions")
    print("3. Use /analyze_polynomials for direct polynomial analysis")
    print("4. Check /julia_status to see if Julia backend is available")

if __name__ == "__main__":
    main()