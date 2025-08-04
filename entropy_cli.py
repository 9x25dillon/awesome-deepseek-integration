#!/usr/bin/env python3
"""
Command-line interface for the Polynomial Entropy Engine
Integrated with polynomial analysis and RAG system
"""

import argparse
import json
import sys
import random
import re
from core import Token, EntropyNode, EntropyEngine
from utils import extract_polynomial_features, calculate_polynomial_complexity
from julia_client import julia_client

def create_polynomial_transformations():
    """Create polynomial-specific transformation functions"""
    
    def expand_polynomial(value, entropy):
        """Expand polynomial expressions"""
        # Simple expansion: (x+1)(x-1) -> x^2-1
        if "(" in value and ")" in value:
            # Basic pattern matching for simple expansions
            pattern = r'\(([^)]+)\)\(([^)]+)\)'
            match = re.search(pattern, value)
            if match:
                return f"{value} -> expanded form"
        return value
    
    def factor_polynomial(value, entropy):
        """Attempt to factor polynomial expressions"""
        # Look for common patterns
        if "x^2" in value and "+" in value:
            return f"factored({value})"
        return value
    
    def add_variable(value, entropy):
        """Add a new variable to the polynomial"""
        variables = "abcdefghijklmnpqrstuvwxyz"
        existing_vars = set(re.findall(r'[a-zA-Z]', value))
        new_var = next((v for v in variables if v not in existing_vars), 'z')
        return f"{value} + {new_var}"
    
    def increase_degree(value, entropy):
        """Increase the polynomial degree"""
        # Find the main variable
        var_match = re.search(r'([a-zA-Z])', value)
        if var_match:
            var = var_match.group(1)
            return f"{value} + {var}^3"
        return f"{value} + x^3"
    
    def substitute_value(value, entropy):
        """Substitute a numerical value for a variable"""
        var_match = re.search(r'([a-zA-Z])', value)
        if var_match:
            var = var_match.group(1)
            sub_value = int(entropy) % 5 + 1
            return value.replace(var, str(sub_value))
        return value
    
    def differentiate(value, entropy):
        """Simple symbolic differentiation"""
        # Basic differentiation patterns
        if "x^2" in value:
            result = value.replace("x^2", "2x")
        elif "x^3" in value:
            result = value.replace("x^3", "3x^2")
        elif "x" in value and "^" not in value:
            result = value.replace("x", "1")
        else:
            result = f"d/dx({value})"
        return result
    
    def add_constant(value, entropy):
        """Add a constant term"""
        constant = int(entropy * 10) % 20 - 10
        if constant >= 0:
            return f"{value} + {constant}"
        else:
            return f"{value} - {abs(constant)}"
    
    def normalize_coefficients(value, entropy):
        """Normalize polynomial coefficients"""
        # Extract coefficients and normalize
        coeffs = re.findall(r'(\d+)', value)
        if coeffs:
            max_coeff = max(int(c) for c in coeffs)
            if max_coeff > 1:
                normalized = value
                for coeff in coeffs:
                    if int(coeff) > 1:
                        new_coeff = int(coeff) // max_coeff
                        normalized = normalized.replace(coeff, str(new_coeff), 1)
                return normalized
        return value
    
    transformations = {
        # Original transformations
        "reverse": lambda value, entropy: str(value)[::-1],
        "uppercase": lambda value, entropy: str(value).upper(),
        "lowercase": lambda value, entropy: str(value).lower(),
        "duplicate": lambda value, entropy: str(value) * 2,
        "add_random": lambda value, entropy: str(value) + random.choice("abcdefghijklmnopqrstuvwxyz0123456789"),
        "add_entropy": lambda value, entropy: str(value) + f"*{entropy:.2f}",
        "truncate": lambda value, entropy: str(value)[:max(1, len(str(value)) // 2)],
        "multiply": lambda value, entropy: str(value) * int(entropy + 1),
        
        # Polynomial-specific transformations
        "expand": expand_polynomial,
        "factor": factor_polynomial,
        "add_variable": add_variable,
        "increase_degree": increase_degree,
        "substitute": substitute_value,
        "differentiate": differentiate,
        "add_constant": add_constant,
        "normalize": normalize_coefficients,
    }
    return transformations

def parse_node_config(config_str):
    """Parse node configuration from string format: name:transform:limit"""
    parts = config_str.split(":")
    name = parts[0]
    transform_name = parts[1] if len(parts) > 1 else "reverse"
    limit = float(parts[2]) if len(parts) > 2 else None
    return name, transform_name, limit

def build_node_hierarchy(node_configs, transforms):
    """Build a hierarchy of nodes from configuration"""
    nodes = {}
    root_node = None
    
    for config in node_configs:
        name, transform_name, limit = parse_node_config(config)
        
        if transform_name not in transforms:
            raise ValueError(f"Unknown transformation '{transform_name}'")
        
        node = EntropyNode(name, transforms[transform_name], entropy_limit=limit)
        nodes[name] = node
        
        if root_node is None:
            root_node = node
        else:
            # For now, create a linear chain. In future, support parent:child syntax
            root_node.add_child(node)
    
    return root_node, nodes

def analyze_polynomial_with_julia(token):
    """Analyze polynomial using Julia if available"""
    if not julia_client.is_available() or token.token_type != "polynomial":
        return None
    
    try:
        analysis = julia_client.analyze_polynomials([token.current_value])
        return analysis
    except Exception as e:
        print(f"Julia analysis failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Polynomial Entropy Engine - Process polynomial expressions through entropy-based transformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic polynomial transformation
  python entropy_cli.py -i "x^2 + 3x + 2" -n "root:factor"

  # Multiple polynomial transformations with entropy limits
  python entropy_cli.py -i "x^2 + y^2" -n "root:expand:8.0" "child:add_variable:7.0"

  # Text processing with polynomial-aware transformations
  python entropy_cli.py -i "2x + 3y" -n "root:differentiate" "child:add_constant"

  # Save results and include Julia analysis
  python entropy_cli.py -i "x^3 - 1" -n "root:factor" --output results.json --julia

  # Show available transformations
  python entropy_cli.py --list-transforms
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input polynomial expression or text"
    )
    
    parser.add_argument(
        "--nodes", "-n",
        nargs="+",
        help="Node configurations in format 'name:transform:limit' (limit is optional)"
    )
    
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=5,
        help="Maximum processing depth (default: 5)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--list-transforms", "-l",
        action="store_true",
        help="List available transformation functions"
    )
    
    parser.add_argument(
        "--julia", "-j",
        action="store_true",
        help="Enable Julia analysis for polynomials"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--polynomial-only", "-p",
        action="store_true",
        help="Only process if input is detected as polynomial"
    )
    
    args = parser.parse_args()
    
    # List available transformations
    if args.list_transforms:
        transforms = create_polynomial_transformations()
        print("Available transformation functions:")
        print("\n📐 Polynomial-specific:")
        poly_transforms = ["expand", "factor", "add_variable", "increase_degree", 
                          "substitute", "differentiate", "add_constant", "normalize"]
        for name in poly_transforms:
            if name in transforms:
                print(f"  {name}")
        
        print("\n📝 General text:")
        general_transforms = ["reverse", "uppercase", "lowercase", "duplicate", 
                            "add_random", "add_entropy", "truncate", "multiply"]
        for name in general_transforms:
            if name in transforms:
                print(f"  {name}")
        return
    
    # Validate required arguments
    if not args.input:
        parser.error("--input is required")
    
    if not args.nodes:
        parser.error("--nodes is required")
    
    # Create transformation functions
    transforms = create_polynomial_transformations()
    
    # Create token and check if it's polynomial
    token = Token(args.input)
    
    if args.polynomial_only and token.token_type != "polynomial":
        print(f"Input '{args.input}' is not detected as a polynomial expression.")
        print("Use --polynomial-only=false to process anyway.")
        sys.exit(1)
    
    if args.verbose:
        print(f"Input type detected: {token.token_type}")
        if token.token_type == "polynomial":
            print(f"Polynomial features: {token.polynomial_features}")
    
    # Build node hierarchy
    try:
        root_node, nodes = build_node_hierarchy(args.nodes, transforms)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Use --list-transforms to see available transformations", file=sys.stderr)
        sys.exit(1)
    
    # Create engine and process token
    engine = EntropyEngine(root_node, max_depth=args.max_depth)
    
    if args.verbose:
        print(f"Initial token: {token}")
        print(f"Julia backend available: {julia_client.is_available()}")
    
    # Process token
    processed_token = engine.run(token)
    
    if args.verbose:
        print(f"Final token: {processed_token}")
    
    # Julia analysis if requested and available
    julia_analysis = None
    if args.julia:
        julia_analysis = analyze_polynomial_with_julia(processed_token)
        if julia_analysis and args.verbose:
            print(f"Julia analysis: {julia_analysis}")
    
    # Prepare results
    results = {
        "input": args.input,
        "input_type": token.token_type,
        "token_summary": processed_token.summary(),
        "entropy_stats": engine.entropy_stats(),
        "processing_graph": engine.export_graph(),
        "polynomial_analysis": engine.polynomial_transformation_summary(),
        "julia_analysis": julia_analysis,
        "configuration": {
            "max_depth": args.max_depth,
            "nodes": args.nodes,
            "julia_enabled": args.julia,
            "polynomial_only": args.polynomial_only
        }
    }
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        # Pretty print key results
        print("\n🧮 POLYNOMIAL ENTROPY ENGINE RESULTS")
        print("=" * 50)
        print(f"Input: {args.input}")
        print(f"Type: {token.token_type}")
        print(f"Final: {processed_token.current_value}")
        print(f"Entropy: {processed_token.initial_entropy:.3f} → {processed_token.current_entropy:.3f}")
        print(f"Trend: {processed_token.entropy_trend()}")
        print(f"Transformations: {len(processed_token.transformations)}")
        
        if token.token_type == "polynomial" and processed_token.polynomial_features:
            print(f"Polynomial degree: {processed_token.polynomial_features['degree']}")
            print(f"Variables: {processed_token.polynomial_features['variables']}")
        
        if args.verbose:
            print(f"\nFull results:")
            print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()