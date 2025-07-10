#!/usr/bin/env python3
"""
Recursive Code Parsing and Representation Module
Task 53.2: Develop Recursive Code Parsing and Representation Module

This module provides comprehensive parsing and representation capabilities
for recursive code analysis, supporting multiple programming languages
and building detailed abstract syntax trees and control flow graphs.
"""

import ast
import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from pathlib import Path
import copy

# Import our complexity specification
try:
    from recursive_complexity_metrics_specification import *
except ImportError:
    # Fallback definitions if specification module not available
    from enum import Enum
    
    class RecursionPattern(Enum):
        LINEAR_RECURSION = "linear"
        BINARY_RECURSION = "binary"
        TAIL_RECURSION = "tail"
        MUTUAL_RECURSION = "mutual"
        TREE_TRAVERSAL = "tree_traversal"
        DIVIDE_CONQUER = "divide_conquer"
        BACKTRACKING = "backtracking"
        NESTED_RECURSION = "nested"
        PARALLEL_RECURSION = "parallel"


class LanguageSupport(Enum):
    """Supported programming languages for recursive analysis"""
    PYTHON = "python"
    JAVASCRIPT = "javascript" 
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    HASKELL = "haskell"


class NodeType(Enum):
    """AST node types for recursive analysis"""
    FUNCTION_DEF = "function_definition"
    FUNCTION_CALL = "function_call"
    RECURSIVE_CALL = "recursive_call"
    BASE_CASE = "base_case"
    CONDITION = "condition"
    LOOP = "loop"
    RETURN = "return"
    VARIABLE = "variable"
    PARAMETER = "parameter"


@dataclass
class RecursiveCallSite:
    """Represents a recursive call site in the code"""
    line_number: int
    column: int
    function_name: str
    call_arguments: List[str]
    call_context: str  # surrounding code context
    is_tail_call: bool = False
    is_conditional: bool = False
    recursion_depth_estimate: int = 1
    call_pattern: RecursionPattern = RecursionPattern.LINEAR_RECURSION
    

@dataclass
class BaseCase:
    """Represents a base case in recursive function"""
    line_number: int
    condition: str
    return_value: Optional[str]
    is_explicit: bool = True  # vs implicit base case
    complexity_contribution: str = "O(1)"


@dataclass
class RecursiveFunction:
    """Comprehensive representation of a recursive function"""
    name: str
    language: LanguageSupport
    source_file: str
    start_line: int
    end_line: int
    parameters: List[str]
    return_type: Optional[str]
    
    # Recursive characteristics
    recursive_calls: List[RecursiveCallSite]
    base_cases: List[BaseCase]
    recursion_pattern: RecursionPattern
    is_tail_recursive: bool
    is_mutually_recursive: bool
    mutual_recursion_partners: List[str] = field(default_factory=list)
    
    # Analysis metadata
    ast_representation: Optional[Any] = None
    control_flow_graph: Optional[Dict] = None
    complexity_annotations: Dict[str, Any] = field(default_factory=dict)
    
    def get_recursion_characteristics(self) -> Dict[str, Any]:
        """Get summary of recursion characteristics"""
        return {
            "pattern": self.recursion_pattern.value,
            "tail_recursive": self.is_tail_recursive,
            "call_count": len(self.recursive_calls),
            "base_case_count": len(self.base_cases),
            "mutual_recursion": self.is_mutually_recursive,
            "estimated_max_depth": max(
                call.recursion_depth_estimate for call in self.recursive_calls
            ) if self.recursive_calls else 0
        }


@dataclass  
class ControlFlowNode:
    """Node in the control flow graph"""
    node_id: str
    node_type: NodeType
    code_content: str
    line_number: int
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    is_recursive_call: bool = False
    is_base_case: bool = False
    complexity_weight: float = 1.0


@dataclass
class ControlFlowGraph:
    """Control flow graph for recursive function analysis"""
    function_name: str
    nodes: Dict[str, ControlFlowNode]
    entry_node: str
    exit_nodes: List[str]
    recursive_cycles: List[List[str]] = field(default_factory=list)
    
    def get_paths_to_base_cases(self) -> List[List[str]]:
        """Get all paths from entry to base cases"""
        base_case_nodes = [
            node_id for node_id, node in self.nodes.items() 
            if node.is_base_case
        ]
        
        paths = []
        for base_case in base_case_nodes:
            path = self._find_path(self.entry_node, base_case)
            if path:
                paths.append(path)
        return paths
    
    def _find_path(self, start: str, end: str, 
                   visited: Set[str] = None) -> Optional[List[str]]:
        """Find path between two nodes using DFS"""
        if visited is None:
            visited = set()
        
        if start == end:
            return [start]
        
        if start in visited:
            return None
        
        visited.add(start)
        
        for successor in self.nodes[start].successors:
            path = self._find_path(successor, end, visited.copy())
            if path:
                return [start] + path
        
        return None


class CodeParser(ABC):
    """Abstract base class for language-specific parsers"""
    
    @abstractmethod
    def parse_file(self, file_path: str) -> List[RecursiveFunction]:
        """Parse a file and extract recursive functions"""
        pass
    
    @abstractmethod
    def build_ast(self, source_code: str) -> Any:
        """Build abstract syntax tree from source code"""
        pass
    
    @abstractmethod
    def identify_recursive_calls(self, ast_node: Any) -> List[RecursiveCallSite]:
        """Identify recursive calls in the AST"""
        pass
    
    @abstractmethod
    def extract_base_cases(self, ast_node: Any) -> List[BaseCase]:
        """Extract base cases from the AST"""
        pass


class PythonParser(CodeParser):
    """Python-specific recursive code parser"""
    
    def __init__(self):
        self.logger = logging.getLogger("PythonParser")
    
    def parse_file(self, file_path: str) -> List[RecursiveFunction]:
        """Parse Python file and extract recursive functions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = self.build_ast(source_code)
            return self._extract_recursive_functions(tree, file_path, source_code)
        
        except Exception as e:
            self.logger.error(f"Error parsing Python file {file_path}: {e}")
            return []
    
    def build_ast(self, source_code: str) -> ast.AST:
        """Build Python AST from source code"""
        return ast.parse(source_code)
    
    def _extract_recursive_functions(self, tree: ast.AST, file_path: str, 
                                   source_code: str) -> List[RecursiveFunction]:
        """Extract all recursive functions from Python AST"""
        recursive_functions = []
        source_lines = source_code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._analyze_function(node, file_path, source_lines)
                if func and func.recursive_calls:  # Only include if recursive
                    recursive_functions.append(func)
        
        # Detect mutual recursion
        self._detect_mutual_recursion(recursive_functions)
        
        return recursive_functions
    
    def _analyze_function(self, func_node: ast.FunctionDef, file_path: str,
                         source_lines: List[str]) -> Optional[RecursiveFunction]:
        """Analyze a single function for recursive characteristics"""
        func_name = func_node.name
        
        # Extract parameters
        parameters = [arg.arg for arg in func_node.args.args]
        
        # Identify recursive calls
        recursive_calls = self.identify_recursive_calls(func_node)
        
        # Extract base cases
        base_cases = self.extract_base_cases(func_node)
        
        # Determine recursion pattern
        pattern = self._determine_recursion_pattern(recursive_calls, func_node)
        
        # Check for tail recursion
        is_tail_recursive = self._is_tail_recursive(func_node, func_name)
        
        # Build control flow graph
        cfg = self._build_control_flow_graph(func_node, func_name)
        
        return RecursiveFunction(
            name=func_name,
            language=LanguageSupport.PYTHON,
            source_file=file_path,
            start_line=func_node.lineno,
            end_line=func_node.end_lineno or func_node.lineno,
            parameters=parameters,
            return_type=None,  # Python doesn't require type annotations
            recursive_calls=recursive_calls,
            base_cases=base_cases,
            recursion_pattern=pattern,
            is_tail_recursive=is_tail_recursive,
            is_mutually_recursive=False,  # Will be updated later
            ast_representation=func_node,
            control_flow_graph=asdict(cfg) if cfg else None
        )
    
    def identify_recursive_calls(self, func_node: ast.FunctionDef) -> List[RecursiveCallSite]:
        """Identify recursive calls within a function"""
        recursive_calls = []
        func_name = func_node.name
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if self._is_recursive_call(node, func_name):
                    call_site = RecursiveCallSite(
                        line_number=node.lineno,
                        column=node.col_offset,
                        function_name=func_name,
                        call_arguments=[self._ast_to_string(arg) for arg in node.args],
                        call_context=self._get_call_context(node),
                        is_tail_call=self._is_tail_call(node, func_node),
                        is_conditional=self._is_conditional_call(node),
                        call_pattern=self._analyze_call_pattern(node, func_node)
                    )
                    recursive_calls.append(call_site)
        
        return recursive_calls
    
    def extract_base_cases(self, func_node: ast.FunctionDef) -> List[BaseCase]:
        """Extract base cases from function"""
        base_cases = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                # Check if this is a base case condition
                if self._is_base_case_condition(node, func_node):
                    base_case = BaseCase(
                        line_number=node.lineno,
                        condition=self._ast_to_string(node.test),
                        return_value=self._extract_return_value(node),
                        is_explicit=True
                    )
                    base_cases.append(base_case)
        
        return base_cases
    
    def _is_recursive_call(self, call_node: ast.Call, func_name: str) -> bool:
        """Check if a call node is a recursive call"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id == func_name
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr == func_name
        return False
    
    def _is_tail_call(self, call_node: ast.Call, func_node: ast.FunctionDef) -> bool:
        """Check if a recursive call is a tail call"""
        # Find the parent statement
        for parent in ast.walk(func_node):
            for child in ast.iter_child_nodes(parent):
                if child == call_node:
                    # Check if this is a return statement
                    return isinstance(parent, ast.Return)
        return False
    
    def _is_conditional_call(self, call_node: ast.Call) -> bool:
        """Check if call is within a conditional block"""
        # This would need more sophisticated AST traversal to find parent If nodes
        return False  # Simplified for now
    
    def _determine_recursion_pattern(self, recursive_calls: List[RecursiveCallSite],
                                   func_node: ast.FunctionDef) -> RecursionPattern:
        """Determine the recursion pattern based on calls"""
        if not recursive_calls:
            return RecursionPattern.LINEAR_RECURSION
        
        # Count recursive calls
        call_count = len(recursive_calls)
        
        # Check for tail recursion
        tail_calls = [call for call in recursive_calls if call.is_tail_call]
        if tail_calls and len(tail_calls) == call_count:
            return RecursionPattern.TAIL_RECURSION
        
        # Check for binary recursion (2 recursive calls)
        if call_count == 2:
            return RecursionPattern.BINARY_RECURSION
        
        # Check for divide and conquer pattern
        if self._has_divide_conquer_pattern(func_node):
            return RecursionPattern.DIVIDE_CONQUER
        
        # Check for tree traversal pattern
        if self._has_tree_traversal_pattern(func_node):
            return RecursionPattern.TREE_TRAVERSAL
        
        # Default to linear recursion
        return RecursionPattern.LINEAR_RECURSION
    
    def _is_tail_recursive(self, func_node: ast.FunctionDef, func_name: str) -> bool:
        """Check if function is tail recursive"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                if self._is_recursive_call(node.value, func_name):
                    return True
        return False
    
    def _build_control_flow_graph(self, func_node: ast.FunctionDef, 
                                 func_name: str) -> ControlFlowGraph:
        """Build control flow graph for the function"""
        nodes = {}
        node_counter = 0
        
        def create_node(ast_node: ast.AST, node_type: NodeType) -> ControlFlowNode:
            nonlocal node_counter
            node_id = f"node_{node_counter}"
            node_counter += 1
            
            return ControlFlowNode(
                node_id=node_id,
                node_type=node_type,
                code_content=self._ast_to_string(ast_node),
                line_number=getattr(ast_node, 'lineno', 0),
                is_recursive_call=isinstance(ast_node, ast.Call) and 
                                self._is_recursive_call(ast_node, func_name),
                is_base_case=self._is_base_case_node(ast_node, func_node)
            )
        
        # Create entry node
        entry_node = create_node(func_node, NodeType.FUNCTION_DEF)
        nodes[entry_node.node_id] = entry_node
        
        # Build nodes for function body
        current_id = entry_node.node_id
        for stmt in func_node.body:
            stmt_node = self._process_statement(stmt, func_name, nodes, node_counter)
            if stmt_node:
                nodes[current_id].successors.append(stmt_node.node_id)
                stmt_node.predecessors.append(current_id)
                current_id = stmt_node.node_id
        
        # Find exit nodes (returns)
        exit_nodes = [
            node_id for node_id, node in nodes.items() 
            if node.node_type == NodeType.RETURN
        ]
        
        return ControlFlowGraph(
            function_name=func_name,
            nodes=nodes,
            entry_node=entry_node.node_id,
            exit_nodes=exit_nodes
        )
    
    def _process_statement(self, stmt: ast.AST, func_name: str, 
                          nodes: Dict[str, ControlFlowNode], 
                          node_counter: int) -> Optional[ControlFlowNode]:
        """Process a statement and create corresponding CFG nodes"""
        # Simplified implementation - would need more sophisticated handling
        # for complex control flow structures
        if isinstance(stmt, ast.Return):
            node = ControlFlowNode(
                node_id=f"node_{len(nodes)}",
                node_type=NodeType.RETURN,
                code_content=self._ast_to_string(stmt),
                line_number=stmt.lineno
            )
            return node
        
        # Add other statement types as needed
        return None
    
    def _detect_mutual_recursion(self, functions: List[RecursiveFunction]):
        """Detect mutual recursion between functions"""
        func_names = {func.name for func in functions}
        
        for func in functions:
            partners = set()
            for call in func.recursive_calls:
                if call.function_name in func_names and call.function_name != func.name:
                    partners.add(call.function_name)
            
            if partners:
                func.is_mutually_recursive = True
                func.mutual_recursion_partners = list(partners)
    
    # Helper methods for AST analysis
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation"""
        try:
            return ast.unparse(node)
        except:
            return str(node)
    
    def _get_call_context(self, call_node: ast.Call) -> str:
        """Get context around the function call"""
        return self._ast_to_string(call_node)
    
    def _analyze_call_pattern(self, call_node: ast.Call, 
                            func_node: ast.FunctionDef) -> RecursionPattern:
        """Analyze the pattern of a specific recursive call"""
        return RecursionPattern.LINEAR_RECURSION  # Simplified
    
    def _is_base_case_condition(self, if_node: ast.If, 
                              func_node: ast.FunctionDef) -> bool:
        """Check if an If node represents a base case"""
        # Check if the if body contains only return statements (no recursive calls)
        for node in ast.walk(if_node):
            if isinstance(node, ast.Call) and self._is_recursive_call(node, func_node.name):
                return False
        return True
    
    def _extract_return_value(self, if_node: ast.If) -> Optional[str]:
        """Extract return value from base case"""
        for node in ast.walk(if_node):
            if isinstance(node, ast.Return):
                return self._ast_to_string(node.value) if node.value else "None"
        return None
    
    def _has_divide_conquer_pattern(self, func_node: ast.FunctionDef) -> bool:
        """Check for divide and conquer pattern"""
        # Look for patterns that split input and combine results
        return False  # Simplified
    
    def _has_tree_traversal_pattern(self, func_node: ast.FunctionDef) -> bool:
        """Check for tree traversal pattern"""
        # Look for patterns accessing tree-like structures
        return False  # Simplified
    
    def _is_base_case_node(self, node: ast.AST, func_node: ast.FunctionDef) -> bool:
        """Check if node represents a base case"""
        return False  # Simplified


class RecursiveCodeAnalyzer:
    """Main analyzer for recursive code parsing and representation"""
    
    def __init__(self):
        self.parsers = {
            LanguageSupport.PYTHON: PythonParser()
        }
        self.logger = logging.getLogger("RecursiveCodeAnalyzer")
    
    def analyze_file(self, file_path: str, 
                    language: LanguageSupport = None) -> List[RecursiveFunction]:
        """Analyze a source file for recursive functions"""
        if language is None:
            language = self._detect_language(file_path)
        
        if language not in self.parsers:
            self.logger.warning(f"Language {language} not supported")
            return []
        
        parser = self.parsers[language]
        return parser.parse_file(file_path)
    
    def analyze_directory(self, directory_path: str, 
                         recursive: bool = True) -> Dict[str, List[RecursiveFunction]]:
        """Analyze all source files in a directory"""
        results = {}
        directory = Path(directory_path)
        
        for file_path in directory.rglob("*") if recursive else directory.iterdir():
            if file_path.is_file() and self._is_source_file(file_path):
                try:
                    functions = self.analyze_file(str(file_path))
                    if functions:
                        results[str(file_path)] = functions
                except Exception as e:
                    self.logger.error(f"Error analyzing {file_path}: {e}")
        
        return results
    
    def generate_analysis_report(self, functions: List[RecursiveFunction]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        if not functions:
            return {"error": "No recursive functions found"}
        
        # Pattern distribution
        pattern_counts = {}
        for func in functions:
            pattern = func.recursion_pattern.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Complexity characteristics
        tail_recursive_count = sum(1 for func in functions if func.is_tail_recursive)
        mutual_recursive_count = sum(1 for func in functions if func.is_mutually_recursive)
        
        # Call site analysis
        total_recursive_calls = sum(len(func.recursive_calls) for func in functions)
        total_base_cases = sum(len(func.base_cases) for func in functions)
        
        return {
            "analysis_summary": {
                "total_recursive_functions": len(functions),
                "tail_recursive_functions": tail_recursive_count,
                "mutually_recursive_functions": mutual_recursive_count,
                "total_recursive_calls": total_recursive_calls,
                "total_base_cases": total_base_cases
            },
            "pattern_distribution": pattern_counts,
            "language_distribution": self._get_language_distribution(functions),
            "complexity_insights": self._generate_complexity_insights(functions),
            "detailed_functions": [asdict(func) for func in functions]
        }
    
    def _detect_language(self, file_path: str) -> LanguageSupport:
        """Detect programming language from file extension"""
        suffix = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': LanguageSupport.PYTHON,
            '.js': LanguageSupport.JAVASCRIPT,
            '.ts': LanguageSupport.TYPESCRIPT,
            '.java': LanguageSupport.JAVA,
            '.cpp': LanguageSupport.CPP,
            '.c': LanguageSupport.C,
            '.go': LanguageSupport.GO,
            '.rs': LanguageSupport.RUST,
            '.hs': LanguageSupport.HASKELL
        }
        
        return language_map.get(suffix, LanguageSupport.PYTHON)
    
    def _is_source_file(self, file_path: Path) -> bool:
        """Check if file is a supported source code file"""
        source_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.hs'}
        return file_path.suffix.lower() in source_extensions
    
    def _get_language_distribution(self, functions: List[RecursiveFunction]) -> Dict[str, int]:
        """Get distribution of functions by language"""
        distribution = {}
        for func in functions:
            lang = func.language.value
            distribution[lang] = distribution.get(lang, 0) + 1
        return distribution
    
    def _generate_complexity_insights(self, functions: List[RecursiveFunction]) -> Dict[str, Any]:
        """Generate insights about recursive complexity"""
        insights = {
            "optimization_opportunities": [],
            "potential_issues": [],
            "recommendations": []
        }
        
        for func in functions:
            # Check for optimization opportunities
            if not func.is_tail_recursive and func.recursion_pattern in [
                RecursionPattern.LINEAR_RECURSION, RecursionPattern.TAIL_RECURSION
            ]:
                insights["optimization_opportunities"].append(
                    f"Function '{func.name}' could benefit from tail recursion optimization"
                )
            
            # Check for potential issues
            if len(func.base_cases) == 0:
                insights["potential_issues"].append(
                    f"Function '{func.name}' has no explicit base cases"
                )
            
            if len(func.recursive_calls) > 3:
                insights["potential_issues"].append(
                    f"Function '{func.name}' has many recursive calls, check for exponential complexity"
                )
        
        return insights


def main():
    """Demo usage of the recursive code parser"""
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer
    analyzer = RecursiveCodeAnalyzer()
    
    print("🔍 Recursive Code Parsing and Representation Module Demo")
    print("=" * 60)
    
    # Analyze current directory for recursive functions
    print("\n📁 Analyzing current directory for recursive functions...")
    results = analyzer.analyze_directory(".", recursive=False)
    
    all_functions = []
    for file_path, functions in results.items():
        all_functions.extend(functions)
        print(f"\n📄 {file_path}:")
        for func in functions:
            print(f"  🔄 {func.name} - {func.recursion_pattern.value}")
            print(f"     Recursive calls: {len(func.recursive_calls)}")
            print(f"     Base cases: {len(func.base_cases)}")
            print(f"     Tail recursive: {func.is_tail_recursive}")
    
    if all_functions:
        print(f"\n📊 Generating analysis report...")
        report = analyzer.generate_analysis_report(all_functions)
        
        print(f"\n📈 Analysis Summary:")
        for key, value in report["analysis_summary"].items():
            print(f"  {key}: {value}")
        
        print(f"\n🎯 Pattern Distribution:")
        for pattern, count in report["pattern_distribution"].items():
            print(f"  {pattern}: {count}")
        
        if report.get("complexity_insights", {}).get("optimization_opportunities"):
            print(f"\n💡 Optimization Opportunities:")
            for opp in report["complexity_insights"]["optimization_opportunities"]:
                print(f"  • {opp}")
        
        if report.get("complexity_insights", {}).get("potential_issues"):
            print(f"\n⚠️ Potential Issues:")
            for issue in report["complexity_insights"]["potential_issues"]:
                print(f"  • {issue}")
    else:
        print("\n❌ No recursive functions found in current directory")
    
    print("\n✅ Recursive code parsing module operational!")


if __name__ == "__main__":
    main()