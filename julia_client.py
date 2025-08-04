import subprocess
import json
import tempfile
import os
from typing import List, Dict, Optional, Any

class JuliaClient:
    """
    Client for interfacing with Julia for advanced polynomial analysis
    """
    
    def __init__(self, julia_executable: str = "julia"):
        """
        Initialize Julia client
        
        Args:
            julia_executable: Path to Julia executable
        """
        self.julia_executable = julia_executable
        self.julia_available = self._check_julia_availability()
        
    def _check_julia_availability(self) -> bool:
        """Check if Julia is available on the system"""
        try:
            result = subprocess.run(
                [self.julia_executable, "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _create_julia_script(self, polynomials: List[str]) -> str:
        """
        Create Julia script for polynomial analysis
        
        Args:
            polynomials: List of polynomial expressions as strings
            
        Returns:
            Julia script content as string
        """
        
        julia_script = '''
using JSON

# Simple polynomial analysis functions
function analyze_polynomial_degree(poly_str)
    # Extract highest power from polynomial string
    degree = 0
    for match in eachmatch(r"[a-zA-Z]\\^?(\\d+)", poly_str)
        power = match.captures[1] === nothing ? 1 : parse(Int, match.captures[1])
        degree = max(degree, power)
    end
    return degree
end

function count_terms(poly_str)
    # Count terms by splitting on + and - (excluding leading signs)
    cleaned = replace(poly_str, r"^[+-]" => "")
    terms = split(cleaned, r"[+-]")
    return length(filter(x -> !isempty(strip(x)), terms))
end

function extract_variables(poly_str)
    # Extract unique variables from polynomial
    variables = Set{String}()
    for match in eachmatch(r"([a-zA-Z])", poly_str)
        push!(variables, match.captures[1])
    end
    return collect(variables)
end

function calculate_complexity(degree, term_count, var_count)
    # Calculate polynomial complexity score
    return degree^2 + term_count * 1.5 + var_count * 2
end

function analyze_polynomial(poly_str)
    degree = analyze_polynomial_degree(poly_str)
    term_count = count_terms(poly_str)
    variables = extract_variables(poly_str)
    complexity = calculate_complexity(degree, term_count, length(variables))
    
    return Dict(
        "expression" => poly_str,
        "degree" => degree,
        "term_count" => term_count,
        "variables" => variables,
        "complexity_score" => complexity
    )
end

# Analyze polynomials from input
polynomials = ''' + json.dumps(polynomials) + '''
results = [analyze_polynomial(poly) for poly in polynomials]

# Output results as JSON
println(JSON.json(Dict("polynomials" => results)))
'''
        return julia_script
    
    def analyze_polynomials(self, polynomials: List[str]) -> Optional[Dict[str, Any]]:
        """
        Analyze polynomials using Julia
        
        Args:
            polynomials: List of polynomial expressions as strings
            
        Returns:
            Dictionary containing analysis results, or None if Julia is unavailable
        """
        
        if not self.julia_available:
            return None
            
        try:
            # Create Julia script
            julia_script = self._create_julia_script(polynomials)
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
                f.write(julia_script)
                script_path = f.name
            
            try:
                # Execute Julia script
                result = subprocess.run(
                    [self.julia_executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Parse JSON output
                    output_lines = result.stdout.strip().split('\n')
                    json_line = output_lines[-1]  # Last line should be JSON
                    return json.loads(json_line)
                else:
                    print(f"Julia execution failed: {result.stderr}")
                    return None
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(script_path)
                except OSError:
                    pass
                    
        except (subprocess.SubprocessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            print(f"Error in Julia analysis: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Julia client is available for use"""
        return self.julia_available
    
    def compare_polynomials(self, poly1: str, poly2: str) -> Optional[Dict[str, Any]]:
        """
        Compare two polynomials using Julia analysis
        
        Args:
            poly1: First polynomial expression
            poly2: Second polynomial expression
            
        Returns:
            Dictionary containing comparison results
        """
        
        analysis = self.analyze_polynomials([poly1, poly2])
        if not analysis or len(analysis.get("polynomials", [])) != 2:
            return None
            
        p1_data = analysis["polynomials"][0]
        p2_data = analysis["polynomials"][1]
        
        comparison = {
            "polynomial_1": p1_data,
            "polynomial_2": p2_data,
            "same_degree": p1_data["degree"] == p2_data["degree"],
            "same_term_count": p1_data["term_count"] == p2_data["term_count"],
            "shared_variables": list(set(p1_data["variables"]).intersection(set(p2_data["variables"]))),
            "complexity_difference": abs(p1_data["complexity_score"] - p2_data["complexity_score"])
        }
        
        return comparison

# Global Julia client instance
julia_client = JuliaClient()