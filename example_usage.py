#!/usr/bin/env python3
"""
Simple example demonstrating the Polynomial RAG System usage

This script shows basic usage patterns for the RAG endpoints.
Make sure the server is running on localhost:8000 before running this script.
"""

import requests
import json

def query_rag_system(query: str, base_url: str = "http://localhost:8000"):
    """Send a query to the RAG system and print the results"""
    
    print(f"🔍 Querying: {query}")
    print("-" * 50)
    
    # Prepare the request
    payload = {
        "query": query,
        "top_k": 3,
        "use_polynomial_analysis": True
    }
    
    try:
        # Send the request
        response = requests.post(f"{base_url}/rag", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📝 Answer:")
            print(result['answer'])
            print(f"\n📚 Retrieved Documents:")
            for i, doc in enumerate(result['context'], 1):
                print(f"  {i}. {doc}")
            print(f"\n🔧 Used Polynomial Analysis: {result['polynomial_analysis_used']}")
            
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
    
    print("\n" + "="*60 + "\n")

def analyze_polynomials(polynomials: list, base_url: str = "http://localhost:8000"):
    """Analyze polynomials directly"""
    
    print(f"🧮 Analyzing polynomials: {polynomials}")
    print("-" * 50)
    
    payload = {
        "polynomials": polynomials,
        "analysis_type": "basic"
    }
    
    try:
        response = requests.post(f"{base_url}/analyze_polynomials", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            for poly_result in result['results']:
                print(f"Expression: {poly_result['expression']}")
                print(f"  Degree: {poly_result['degree']}")
                print(f"  Terms: {poly_result['term_count']}")
                print(f"  Variables: {poly_result['variables']}")
                print(f"  Complexity: {poly_result['complexity_score']}")
                print()
            
            print(f"Analysis Backend: {'Julia' if result['analysis_metadata']['julia_backend'] else 'Python'}")
            
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
    
    print("\n" + "="*60 + "\n")

def main():
    """Run example queries"""
    
    print("🧮 Polynomial RAG System - Example Usage")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ Server is not running. Please start with: python main.py")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Please start with: python main.py")
        return
    
    print("✅ Server is running!")
    print()
    
    # Example 1: Basic polynomial question
    query_rag_system("What is a quadratic polynomial?")
    
    # Example 2: Specific polynomial analysis
    query_rag_system("How do you factor x^2 + 2xy + y^2?")
    
    # Example 3: Comparison question
    query_rag_system("What's the difference between linear and cubic polynomials?")
    
    # Example 4: Complex polynomial query
    query_rag_system("Analyze the polynomial x^4 - 1 and explain its properties")
    
    # Example 5: Direct polynomial analysis
    test_polynomials = ["x^2 + 3x + 2", "2x^3 - x^2 + 4x", "5xy - 3z + 1"]
    analyze_polynomials(test_polynomials)
    
    print("🎉 Example completed! Try your own queries by modifying this script.")

if __name__ == "__main__":
    main()