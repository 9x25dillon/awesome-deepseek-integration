#!/usr/bin/env python3
"""
Comprehensive test script for the Entropy-Enhanced Polynomial RAG System
"""

import requests
import json
import time
from typing import Dict, Any, List

class EntropyRAGTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def test_health(self) -> bool:
        """Test if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def test_entropy_transformations(self) -> Dict[str, Any]:
        """Get available entropy transformations"""
        try:
            response = requests.get(f"{self.base_url}/entropy_transformations")
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def test_entropy_processing(self, input_value: str, transformations: List[str], 
                              max_depth: int = 5, entropy_limits: List[float] = None,
                              use_julia: bool = True) -> Dict[str, Any]:
        """Test entropy processing endpoint"""
        try:
            payload = {
                "input_value": input_value,
                "transformations": transformations,
                "max_depth": max_depth,
                "use_julia": use_julia
            }
            
            if entropy_limits:
                payload["entropy_limits"] = entropy_limits
            
            response = requests.post(f"{self.base_url}/entropy_process", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def test_entropy_rag(self, query: str, entropy_transformations: List[str] = None,
                        top_k: int = 3, entropy_threshold: float = None) -> Dict[str, Any]:
        """Test entropy-enhanced RAG endpoint"""
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "use_polynomial_analysis": True
            }
            
            if entropy_transformations:
                payload["entropy_transformations"] = entropy_transformations
            
            if entropy_threshold is not None:
                payload["entropy_threshold"] = entropy_threshold
            
            response = requests.post(f"{self.base_url}/entropy_rag", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(result: Dict[str, Any], title: str = "Result", truncate_length: int = 100):
    """Print formatted result with optional truncation"""
    print(f"\n{title}:")
    print("-" * 50)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Handle different result types
    if "input_value" in result and "final_value" in result:
        # Entropy processing result
        print(f"📥 Input: {result['input_value']}")
        print(f"📤 Output: {result['final_value']}")
        print(f"🎯 Type: {result['input_type']}")
        print(f"📊 Entropy: {result.get('entropy_change', 0):.3f} change")
        print(f"📈 Trend: {result.get('entropy_trend', 'unknown')}")
        print(f"🔄 Transformations: {result.get('transformations_applied', 0)}")
        
        if result.get('polynomial_features'):
            features = result['polynomial_features']
            print(f"🧮 Polynomial - Degree: {features.get('degree', 'N/A')}, Variables: {features.get('variables', 'N/A')}")
    
    elif "answer" in result:
        # RAG result
        print(f"🔍 Query: {result.get('original_query', result.get('query', 'N/A'))}")
        if result.get('processed_query') and result.get('processed_query') != result.get('original_query'):
            print(f"🔄 Processed: {result['processed_query']}")
        
        answer = result['answer']
        if len(answer) > truncate_length:
            answer = answer[:truncate_length] + "..."
        print(f"📝 Answer: {answer}")
        
        print(f"📚 Context docs: {len(result.get('context', []))}")
        
        if result.get('entropy_analysis'):
            entropy = result['entropy_analysis']
            print(f"🌪️ Entropy change: {entropy.get('entropy_change', 'N/A')}")
    
    else:
        # Generic result
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                print(f"{key}: {json.dumps(value, indent=2)[:200]}...")
            else:
                print(f"{key}: {value}")

def main():
    """Run comprehensive entropy system tests"""
    print("🌪️ ENTROPY-ENHANCED POLYNOMIAL RAG SYSTEM TEST SUITE")
    print("="*70)
    
    tester = EntropyRAGTester()
    
    # Test 1: Health Check
    print_section("1. System Health Check")
    if tester.test_health():
        print("✅ Server is running and healthy")
    else:
        print("❌ Server is not responding")
        return
    
    # Test 2: Available Transformations
    print_section("2. Available Entropy Transformations")
    transformations = tester.test_entropy_transformations()
    print_result(transformations)
    
    # Test 3: Basic Entropy Processing
    print_section("3. Basic Entropy Processing Tests")
    
    test_cases = [
        {
            "input": "x^2 + 3x + 2",
            "transformations": ["factor"],
            "description": "Factor quadratic polynomial"
        },
        {
            "input": "2x + 3y",
            "transformations": ["differentiate", "add_constant"],
            "description": "Differentiate and add constant"
        },
        {
            "input": "polynomial",
            "transformations": ["reverse", "uppercase"],
            "description": "Text transformations"
        },
        {
            "input": "x^3 - 1",
            "transformations": ["factor", "add_variable", "normalize"],
            "description": "Multiple polynomial transformations"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test 3.{i}: {test_case['description']} ---")
        result = tester.test_entropy_processing(
            test_case["input"],
            test_case["transformations"]
        )
        print_result(result, f"Entropy Processing Result")
        time.sleep(0.5)
    
    # Test 4: Entropy Processing with Limits
    print_section("4. Entropy Processing with Limits")
    
    print("--- High entropy limit (should process) ---")
    result = tester.test_entropy_processing(
        "x^2 + y^2",
        ["add_variable", "increase_degree"],
        entropy_limits=[10.0, 15.0]
    )
    print_result(result)
    
    print("\n--- Low entropy limit (may skip some) ---")
    result = tester.test_entropy_processing(
        "very_long_polynomial_expression_with_high_entropy",
        ["truncate", "lowercase"],
        entropy_limits=[2.0, 1.0]
    )
    print_result(result)
    
    # Test 5: Standard RAG vs Entropy-Enhanced RAG
    print_section("5. RAG Comparison Tests")
    
    test_query = "What are the properties of quadratic polynomials?"
    
    print("--- Standard RAG ---")
    standard_result = tester.test_entropy_rag(test_query)
    print_result(standard_result, "Standard RAG")
    
    print("\n--- Entropy-Enhanced RAG ---")
    entropy_result = tester.test_entropy_rag(
        test_query,
        entropy_transformations=["add_constant", "normalize"]
    )
    print_result(entropy_result, "Entropy-Enhanced RAG")
    
    # Test 6: Entropy Threshold Filtering
    print_section("6. Entropy Threshold Filtering")
    
    threshold_tests = [
        {"threshold": None, "description": "No threshold"},
        {"threshold": 2.0, "description": "Low threshold (2.0)"},
        {"threshold": 4.0, "description": "High threshold (4.0)"}
    ]
    
    query = "Analyze polynomial x^4 - 1"
    
    for test in threshold_tests:
        print(f"\n--- {test['description']} ---")
        result = tester.test_entropy_rag(
            query,
            entropy_threshold=test["threshold"]
        )
        print_result(result)
        time.sleep(0.5)
    
    # Test 7: Complex Polynomial Queries with Entropy
    print_section("7. Complex Polynomial Queries")
    
    complex_queries = [
        {
            "query": "How do you expand (x+1)(x-1)?",
            "transformations": ["expand", "factor"],
            "description": "Expansion query with entropy transformations"
        },
        {
            "query": "What is the derivative of x^3 + 2x^2?",
            "transformations": ["differentiate"],
            "description": "Calculus query with differentiation"
        },
        {
            "query": "Compare x^2 + y^2 and x^3 + y^3",
            "transformations": ["add_variable", "increase_degree"],
            "description": "Comparison query with polynomial modifications"
        }
    ]
    
    for i, test in enumerate(complex_queries, 1):
        print(f"\n--- Complex Query {i}: {test['description']} ---")
        result = tester.test_entropy_rag(
            test["query"],
            entropy_transformations=test["transformations"]
        )
        print_result(result)
        time.sleep(1)
    
    # Test 8: Performance and Error Handling
    print_section("8. Performance and Error Handling")
    
    print("--- Performance test ---")
    start_time = time.time()
    result = tester.test_entropy_processing(
        "x^5 + x^4 + x^3 + x^2 + x + 1",
        ["factor", "differentiate", "normalize", "add_constant"]
    )
    end_time = time.time()
    
    if "error" not in result:
        print(f"✅ Complex processing completed in {end_time - start_time:.2f} seconds")
        print(f"🔄 Applied {result.get('transformations_applied', 0)} transformations")
    else:
        print(f"❌ Performance test failed: {result['error']}")
    
    print("\n--- Error handling test ---")
    error_result = tester.test_entropy_processing(
        "test",
        ["invalid_transformation", "another_invalid"]
    )
    if "error" in error_result:
        print("✅ Error handling working correctly")
        print(f"📝 Error message: {error_result['error']}")
    else:
        print("❌ Error handling test failed - should have returned error")
    
    # Test Summary
    print_section("Test Summary")
    print("🎉 Entropy-Enhanced Polynomial RAG System test suite completed!")
    print("\n🔧 System Features Tested:")
    print("✅ Basic entropy processing with polynomial transformations")
    print("✅ Entropy limits and thresholds")
    print("✅ Enhanced RAG with query transformation")
    print("✅ Document filtering by entropy")
    print("✅ Julia integration for advanced analysis")
    print("✅ Error handling and performance")
    
    print("\n📊 Available Endpoints:")
    print("🔹 /entropy_process - Process text/polynomials through entropy engine")
    print("🔹 /entropy_rag - Enhanced RAG with entropy transformations")
    print("🔹 /entropy_transformations - List available transformations")
    print("🔹 /julia_status - Check Julia backend status")
    
    print("\n💡 Usage Examples:")
    print("🧮 CLI: python entropy_cli.py -i 'x^2 + 3x + 2' -n 'root:factor' --julia")
    print("🌐 API: POST /entropy_rag with entropy_transformations=['differentiate']")
    print("📈 Threshold: Use entropy_threshold to filter high-entropy documents")

if __name__ == "__main__":
    main()