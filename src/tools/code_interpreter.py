"""
Code Interpreter Tool
Allows the AI agent to write and execute custom Python code for tasks that don't have predefined tools.
This is what makes it a TRUE AI Agent, not just a function-calling bot.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import polars as pl


def execute_python_code(
    code: str,
    working_directory: str = "./outputs/code",
    timeout: int = 60,
    allow_file_operations: bool = True,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute custom Python code written by the AI agent.
    
    This is the KEY tool that transforms the agent from a function-calling bot
    into a true AI agent capable of solving ANY data science problem.
    
    Use cases:
    - Custom visualizations not covered by existing tools
    - Data transformations too specific for generic tools
    - Domain-specific calculations
    - Interactive dashboards
    - Custom export formats
    
    Args:
        code: Python code to execute
        working_directory: Where to run the code (default: ./outputs/code)
        timeout: Maximum execution time in seconds
        allow_file_operations: Whether code can read/write files
        output_file: Optional file path to save output (e.g., HTML plot)
        
    Returns:
        Dict with execution results, stdout, stderr, and any generated files
        
    Example:
        # Agent can write custom Plotly code for specific visualizations
        code = '''
        import plotly.express as px
        import pandas as pd
        
        df = pd.read_csv('./temp/sales_data.csv')
        fig = px.line(df, x='month', y='sales', color='bike_model',
                     title='Extended Sales by Month for Each Bike Model')
        
        # Add dropdown filter
        fig.update_layout(
            updatemenus=[{
                'buttons': [{'label': model, 'method': 'update',
                           'args': [{'visible': [model == m for m in df['bike_model'].unique()]}]}
                          for model in df['bike_model'].unique()],
                'direction': 'down',
                'showactive': True
            }]
        )
        
        fig.write_html('./outputs/code/bike_sales_interactive.html')
        print("Chart saved to: ./outputs/code/bike_sales_interactive.html")
        '''
        
        result = execute_python_code(code)
    """
    try:
        # ⚠️ CRITICAL: Basic syntax validation BEFORE execution
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Syntax error in generated code: {str(e)}",
                "error_type": "SyntaxError",
                "line": e.lineno,
                "suggestion": "Fix syntax errors in the code. Common issues: missing quotes, parentheses, indentation"
            }
        
        # Create working directory with proper permissions
        try:
            os.makedirs(working_directory, exist_ok=True)
            # Ensure directory is writable
            test_file = os.path.join(working_directory, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except PermissionError:
            return {
                "success": False,
                "error": f"No write permission for directory: {working_directory}",
                "error_type": "PermissionError",
                "suggestion": f"Check folder permissions or use a different directory"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create working directory: {str(e)}",
                "error_type": type(e).__name__
            }
        
        # Security: Validate code doesn't contain dangerous operations
        dangerous_patterns = {
            'subprocess': 'Use specialized tools instead of shell commands',
            '__import__': 'Dynamic imports not allowed for security',
            'eval(': 'eval() is dangerous - rewrite without it',
            'exec(': 'exec() is dangerous - rewrite without it',
            'compile(': 'compile() not needed - write code directly',
            'os.system': 'Shell commands not allowed - use Python libraries',
            'os.popen': 'Shell commands not allowed - use Python libraries'
        }
        
        for pattern, reason in dangerous_patterns.items():
            if pattern in code:
                return {
                    "success": False,
                    "error": f"Code contains restricted operation: {pattern}",
                    "error_type": "SecurityError",
                    "reason": reason,
                    "suggestion": "Rewrite code using safe Python operations"
                }
        
        # Create temporary Python file with better error handling
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, 
                                            dir=working_directory, encoding='utf-8') as f:
                temp_file = f.name
                
                # Add helper imports at the top + error handling wrapper
                enhanced_code = """
# Auto-imported libraries for convenience
import pandas as pd
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import traceback

# Ensure output directory exists
import os
os.makedirs('./outputs/code', exist_ok=True)
os.makedirs('./outputs/data', exist_ok=True)

try:
    # User's code starts here
""" + "\n".join("    " + line for line in code.split("\n")) + """

except Exception as e:
    print(f"❌ Error in code execution: {str(e)}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
"""
                
                f.write(enhanced_code)
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write temporary file: {str(e)}",
                "error_type": type(e).__name__,
                "suggestion": "Check file write permissions"
            }
        
        # Track existing files BEFORE execution to detect new files
        existing_files = set()
        if allow_file_operations:
            for output_dir in ['./outputs/code', './outputs/data', './outputs/plots']:
                if os.path.exists(output_dir):
                    for file_path in Path(output_dir).resolve().glob('**/*'):
                        if file_path.is_file():
                            existing_files.add(file_path.resolve())
        
        try:
            # Execute the code with better error capture
            # Use absolute path and normalize it for Windows
            abs_temp_file = os.path.abspath(temp_file)
            abs_cwd = os.path.abspath(Path.cwd())
            
            result = subprocess.run(
                [sys.executable, abs_temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=abs_cwd  # Use absolute path to avoid permission issues
            )
            
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            returncode = result.returncode
            
            # Check for errors with detailed diagnostics
            if returncode != 0:
                # Parse error message for common issues
                error_hints = []
                if "PermissionError" in stderr:
                    error_hints.append("💡 File permission issue - check if file is open in another program")
                if "FileNotFoundError" in stderr:
                    error_hints.append("💡 File not found - check if path is correct (use relative paths like './outputs/data/file.csv')")
                if "KeyError" in stderr:
                    error_hints.append("💡 Column not found - check column names in the CSV")
                if "ModuleNotFoundError" in stderr:
                    error_hints.append("💡 Missing library - may need to install additional packages")
                if "ValueError" in stderr:
                    error_hints.append("💡 Data type mismatch - check data types and conversions")
                
                return {
                    "success": False,
                    "error": f"Code execution failed",
                    "stderr": stderr,
                    "stdout": stdout if stdout else None,
                    "error_type": "ExecutionError",
                    "exit_code": returncode,
                    "hints": error_hints if error_hints else ["Check the error message above for details"]
                }
            
            # Success! Find NEWLY generated files (not existing before execution)
            generated_files = []
            if allow_file_operations:
                cwd = Path.cwd()
                for output_dir in ['./outputs/code', './outputs/data', './outputs/plots']:
                    if os.path.exists(output_dir):
                        abs_output_dir = Path(output_dir).resolve()
                        for file_path in abs_output_dir.glob('**/*'):
                            if file_path.is_file():
                                abs_file = file_path.resolve()
                                
                                # Only include if it's NEW (didn't exist before) or MODIFIED
                                is_new = abs_file not in existing_files
                                
                                # Check if file was modified in last 5 seconds (just created/updated)
                                import time
                                file_age = time.time() - file_path.stat().st_mtime
                                is_recent = file_age < 5
                                
                                if (is_new or is_recent):
                                    # Get relative path safely (handle Windows paths)
                                    try:
                                        rel_path = file_path.relative_to(cwd)
                                    except ValueError:
                                        # Fallback: just use the file name with output dir
                                        rel_path = Path(output_dir) / file_path.name
                                    
                                    # Only include if not temp file and has content
                                    abs_temp = Path(temp_file).resolve() if temp_file else None
                                    if file_path != abs_temp and file_path.stat().st_size > 0:
                                        generated_files.append(str(rel_path).replace('\\', '/'))
            
            # Sort by modification time (newest first)
            if generated_files:
                generated_files = sorted(
                    generated_files,
                    key=lambda x: Path(x).stat().st_mtime,
                    reverse=True
                )[:10]  # Limit to 10 most recent files
            
            return {
                "success": True,
                "stdout": stdout if stdout else "✅ Code executed successfully (no output)",
                "stderr": stderr if stderr else None,
                "message": "✅ Code executed successfully",
                "generated_files": generated_files,
                "working_directory": working_directory,
                "execution_summary": {
                    "lines_of_code": len(code.split('\n')),
                    "files_generated": len(generated_files)
                }
            }
            
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass  # Ignore cleanup errors
                
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Code execution timed out after {timeout} seconds",
            "error_type": "TimeoutError",
            "suggestion": "Code is taking too long. Optimize it or increase timeout. Avoid large loops or heavy computations."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "suggestion": "This is an unexpected error. Try simplifying the code."
        }


def execute_code_from_file(
    file_path: str,
    working_directory: str = "./outputs/code",
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute Python code from a file.
    
    Useful when code is too long to pass as a string, or when the agent
    wants to run an existing script.
    
    Args:
        file_path: Path to Python file to execute
        working_directory: Where to run the code
        timeout: Maximum execution time in seconds
        
    Returns:
        Dict with execution results
    """
    try:
        # Read code from file
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        return execute_python_code(
            code=code,
            working_directory=working_directory,
            timeout=timeout
        )
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "error_type": "FileNotFoundError"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read file: {str(e)}",
            "error_type": type(e).__name__
        }


def generate_custom_visualization(
    data_file: str,
    visualization_description: str,
    output_path: str = "./outputs/code/custom_plot.html",
    timeout: int = 60
) -> Dict[str, Any]:
    """
    HIGH-LEVEL helper: Generate custom visualization from natural language description.
    
    The agent describes what it wants, and this function attempts to generate the code.
    This is a convenience wrapper that could use an LLM to generate the plotting code.
    
    Args:
        data_file: Path to dataset
        visualization_description: Natural language description of desired plot
        output_path: Where to save the visualization
        timeout: Execution timeout
        
    Returns:
        Dict with execution results
        
    Example:
        result = generate_custom_visualization(
            data_file="./temp/sales.csv",
            visualization_description="Line plot of sales by month for each bike model, with dropdown filter",
            output_path="./outputs/code/sales_plot.html"
        )
    """
    # This is a placeholder - in a full implementation, this would use an LLM
    # to generate the Plotly code from the description
    
    return {
        "success": False,
        "error": "Not yet implemented - use execute_python_code with explicit code instead",
        "error_type": "NotImplementedError",
        "suggestion": "Write the Plotly code explicitly and use execute_python_code()"
    }
