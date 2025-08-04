#!/usr/bin/env python3
"""
Core entropy engine modules for polynomial analysis and processing
"""

import math
import json
import re
from typing import Dict, List, Any, Optional, Callable
from utils import extract_polynomial_features, calculate_polynomial_complexity
from julia_client import julia_client

class Token:
    """
    Represents a token (polynomial expression or text) with entropy tracking
    """
    
    def __init__(self, value: str, token_type: str = "auto"):
        self.original_value = value
        self.current_value = value
        self.token_type = self._detect_type(value) if token_type == "auto" else token_type
        self.transformations = []
        self.entropy_history = []
        self.polynomial_features = None
        
        # Calculate initial entropy
        self.initial_entropy = self._calculate_entropy(value)
        self.current_entropy = self.initial_entropy
        self.entropy_history.append({"step": 0, "entropy": self.initial_entropy, "value": value})
        
        # Extract polynomial features if applicable
        if self.token_type == "polynomial":
            self.polynomial_features = extract_polynomial_features(value)
    
    def _detect_type(self, value: str) -> str:
        """Detect if the value is a polynomial expression or general text"""
        # Check for polynomial patterns
        polynomial_pattern = r'[a-zA-Z]\^?\d*|[+-]?\d*[a-zA-Z]'
        if re.search(polynomial_pattern, value):
            return "polynomial"
        return "text"
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def apply_transformation(self, transformation_name: str, transform_func: Callable, step: int):
        """Apply a transformation to the token"""
        old_value = self.current_value
        old_entropy = self.current_entropy
        
        # Apply transformation
        self.current_value = transform_func(self.current_value, self.current_entropy)
        self.current_entropy = self._calculate_entropy(self.current_value)
        
        # Record transformation
        transformation_record = {
            "step": step,
            "name": transformation_name,
            "old_value": old_value,
            "new_value": self.current_value,
            "old_entropy": old_entropy,
            "new_entropy": self.current_entropy,
            "entropy_change": self.current_entropy - old_entropy
        }
        
        self.transformations.append(transformation_record)
        self.entropy_history.append({
            "step": step,
            "entropy": self.current_entropy,
            "value": self.current_value
        })
        
        # Update polynomial features if applicable
        if self.token_type == "polynomial":
            try:
                self.polynomial_features = extract_polynomial_features(self.current_value)
            except:
                # If transformation makes it non-polynomial, update type
                self.token_type = "text"
                self.polynomial_features = None
    
    def entropy_trend(self) -> str:
        """Determine the overall entropy trend"""
        if len(self.entropy_history) < 2:
            return "stable"
        
        start_entropy = self.entropy_history[0]["entropy"]
        end_entropy = self.entropy_history[-1]["entropy"]
        
        if end_entropy > start_entropy * 1.1:
            return "increasing"
        elif end_entropy < start_entropy * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the token's state and transformations"""
        return {
            "original_value": self.original_value,
            "current_value": self.current_value,
            "token_type": self.token_type,
            "initial_entropy": self.initial_entropy,
            "current_entropy": self.current_entropy,
            "entropy_change": self.current_entropy - self.initial_entropy,
            "entropy_trend": self.entropy_trend(),
            "transformations_count": len(self.transformations),
            "polynomial_features": self.polynomial_features,
            "transformations": self.transformations
        }
    
    def __str__(self):
        return f"Token('{self.current_value}', entropy={self.current_entropy:.2f}, type={self.token_type})"


class EntropyNode:
    """
    Represents a node in the entropy processing graph that can transform tokens
    """
    
    def __init__(self, name: str, transform_func: Callable, entropy_limit: Optional[float] = None):
        self.name = name
        self.transform_func = transform_func
        self.entropy_limit = entropy_limit
        self.children = []
        self.parent = None
        self.processed_tokens = []
        self.activation_count = 0
    
    def add_child(self, child_node: 'EntropyNode'):
        """Add a child node"""
        child_node.parent = self
        self.children.append(child_node)
    
    def can_process(self, token: Token) -> bool:
        """Check if this node can process the given token based on entropy limit"""
        if self.entropy_limit is None:
            return True
        return token.current_entropy <= self.entropy_limit
    
    def process(self, token: Token, step: int) -> bool:
        """Process a token through this node"""
        if not self.can_process(token):
            return False
        
        # Apply transformation
        token.apply_transformation(self.name, self.transform_func, step)
        
        # Record processing
        self.processed_tokens.append({
            "step": step,
            "original_value": token.original_value,
            "processed_value": token.current_value,
            "entropy_before": token.entropy_history[-2]["entropy"] if len(token.entropy_history) > 1 else token.initial_entropy,
            "entropy_after": token.current_entropy
        })
        
        self.activation_count += 1
        return True
    
    def get_path_to_root(self) -> List[str]:
        """Get the path from this node to the root"""
        path = [self.name]
        current = self.parent
        while current:
            path.append(current.name)
            current = current.parent
        return list(reversed(path))
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of this node's activity"""
        return {
            "name": self.name,
            "entropy_limit": self.entropy_limit,
            "activation_count": self.activation_count,
            "children_count": len(self.children),
            "children_names": [child.name for child in self.children],
            "processed_tokens": len(self.processed_tokens)
        }


class EntropyEngine:
    """
    Main engine for processing tokens through entropy-based transformation networks
    """
    
    def __init__(self, root_node: EntropyNode, max_depth: int = 10):
        self.root_node = root_node
        self.max_depth = max_depth
        self.processing_history = []
        self.step_counter = 0
        self.julia_analysis_enabled = julia_client.is_available()
    
    def run(self, token: Token) -> Token:
        """Run the token through the entropy processing network"""
        self.processing_history.clear()
        self.step_counter = 0
        
        # Record initial state
        self.processing_history.append({
            "step": 0,
            "node": "input",
            "token_state": token.summary(),
            "action": "initial_state"
        })
        
        # Process through network
        self._process_node(token, self.root_node, 0)
        
        return token
    
    def _process_node(self, token: Token, node: EntropyNode, depth: int):
        """Recursively process token through nodes"""
        if depth >= self.max_depth:
            return
        
        self.step_counter += 1
        
        # Try to process with current node
        if node.can_process(token):
            processed = node.process(token, self.step_counter)
            
            self.processing_history.append({
                "step": self.step_counter,
                "node": node.name,
                "token_state": token.summary(),
                "action": "processed" if processed else "skipped",
                "depth": depth
            })
            
            # If we have polynomial features and Julia is available, get analysis
            if token.token_type == "polynomial" and self.julia_analysis_enabled:
                try:
                    julia_analysis = julia_client.analyze_polynomials([token.current_value])
                    if julia_analysis:
                        self.processing_history[-1]["julia_analysis"] = julia_analysis
                except:
                    pass
            
            # Process children if transformation was successful
            if processed:
                for child in node.children:
                    self._process_node(token, child, depth + 1)
    
    def entropy_stats(self) -> Dict[str, Any]:
        """Generate entropy statistics for the processing session"""
        if not self.processing_history:
            return {}
        
        entropies = []
        for record in self.processing_history:
            if "token_state" in record:
                entropies.append(record["token_state"]["current_entropy"])
        
        if not entropies:
            return {}
        
        return {
            "min_entropy": min(entropies),
            "max_entropy": max(entropies),
            "avg_entropy": sum(entropies) / len(entropies),
            "entropy_variance": sum((e - sum(entropies)/len(entropies))**2 for e in entropies) / len(entropies),
            "total_steps": self.step_counter,
            "julia_enabled": self.julia_analysis_enabled
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the processing graph structure"""
        def export_node(node: EntropyNode) -> Dict[str, Any]:
            return {
                "name": node.name,
                "entropy_limit": node.entropy_limit,
                "activation_count": node.activation_count,
                "children": [export_node(child) for child in node.children]
            }
        
        return {
            "root": export_node(self.root_node),
            "processing_history": self.processing_history,
            "total_nodes": self._count_nodes(self.root_node)
        }
    
    def _count_nodes(self, node: EntropyNode) -> int:
        """Count total nodes in the graph"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def polynomial_transformation_summary(self) -> Dict[str, Any]:
        """Generate summary focused on polynomial transformations"""
        polynomial_steps = []
        
        for record in self.processing_history:
            if "token_state" in record and record["token_state"]["token_type"] == "polynomial":
                polynomial_steps.append({
                    "step": record["step"],
                    "node": record["node"],
                    "polynomial_features": record["token_state"]["polynomial_features"],
                    "entropy": record["token_state"]["current_entropy"],
                    "julia_analysis": record.get("julia_analysis")
                })
        
        return {
            "polynomial_processing_steps": len(polynomial_steps),
            "steps": polynomial_steps,
            "julia_analysis_available": self.julia_analysis_enabled
        }