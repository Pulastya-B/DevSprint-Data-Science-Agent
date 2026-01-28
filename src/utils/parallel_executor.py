"""
Parallel Tool Execution with Dependency Detection

Enables concurrent execution of independent tools while respecting
dependencies and avoiding overwhelming system resources.
"""

import asyncio
from typing import Dict, List, Any, Set, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import time


class ToolWeight(Enum):
    """Tool execution weight (resource intensity)."""
    LIGHT = 1      # Fast operations (< 1s): profiling, validation
    MEDIUM = 2     # Moderate operations (1-10s): cleaning, encoding
    HEAVY = 3      # Expensive operations (> 10s): ML training, large viz


# Tool weight classification
TOOL_WEIGHTS = {
    # Light tools (can run many in parallel)
    "profile_dataset": ToolWeight.LIGHT,
    "detect_data_quality_issues": ToolWeight.LIGHT,
    "analyze_correlations": ToolWeight.LIGHT,
    "get_smart_summary": ToolWeight.LIGHT,
    "smart_type_inference": ToolWeight.LIGHT,
    
    # Medium tools (limit 2-3 concurrent)
    "clean_missing_values": ToolWeight.MEDIUM,
    "handle_outliers": ToolWeight.MEDIUM,
    "encode_categorical": ToolWeight.MEDIUM,
    "create_time_features": ToolWeight.MEDIUM,
    "create_interaction_features": ToolWeight.MEDIUM,
    "create_ratio_features": ToolWeight.MEDIUM,
    "create_statistical_features": ToolWeight.MEDIUM,
    "generate_interactive_scatter": ToolWeight.MEDIUM,
    "generate_interactive_histogram": ToolWeight.MEDIUM,
    "generate_interactive_box_plots": ToolWeight.MEDIUM,
    "generate_interactive_correlation_heatmap": ToolWeight.MEDIUM,
    
    # Heavy tools (limit 1 concurrent) - NEVER RUN MULTIPLE HEAVY TOOLS IN PARALLEL
    "train_baseline_models": ToolWeight.HEAVY,
    "hyperparameter_tuning": ToolWeight.HEAVY,
    "perform_cross_validation": ToolWeight.HEAVY,
    "train_ensemble_models": ToolWeight.HEAVY,
    "auto_ml_pipeline": ToolWeight.HEAVY,
    "generate_ydata_profiling_report": ToolWeight.HEAVY,
    "generate_combined_eda_report": ToolWeight.HEAVY,
    "generate_plotly_dashboard": ToolWeight.HEAVY,
    "execute_python_code": ToolWeight.HEAVY,  # Unknown code complexity
    "auto_feature_engineering": ToolWeight.HEAVY,  # ML-based feature generation
}


@dataclass
class ToolExecution:
    """Represents a tool execution task."""
    tool_name: str
    arguments: Dict[str, Any]
    weight: ToolWeight
    dependencies: Set[str]  # Other tool names that must complete first
    execution_id: str
    
    def __hash__(self):
        return hash(self.execution_id)


class ToolDependencyGraph:
    """
    Analyzes tool dependencies based on input/output files.
    
    Detects dependencies like:
    - clean_missing_values → encode_categorical (same file transformation)
    - profile_dataset → train_baseline_models (uses profiling results)
    - Multiple visualizations (can run in parallel)
    """
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = {}
    
    def detect_dependencies(self, executions: List[ToolExecution]) -> Dict[str, Set[str]]:
        """
        Detect dependencies between tool executions.
        
        Rules:
        1. If tool B reads output of tool A → B depends on A
        2. If tools read/write same file → sequential execution
        3. If tools are independent (different files/ops) → parallel
        
        Args:
            executions: List of tool executions
        
        Returns:
            Dict mapping execution_id → set of execution_ids it depends on
        """
        dependencies: Dict[str, Set[str]] = {ex.execution_id: set() for ex in executions}
        
        # Build file I/O map
        file_producers: Dict[str, str] = {}  # file_path → execution_id
        file_consumers: Dict[str, List[str]] = {}  # file_path → [execution_ids]
        
        for ex in executions:
            # Check input files
            input_file = ex.arguments.get("file_path")
            if input_file:
                if input_file not in file_consumers:
                    file_consumers[input_file] = []
                file_consumers[input_file].append(ex.execution_id)
            
            # Check output files
            output_file = ex.arguments.get("output_path") or ex.arguments.get("output_file")
            if output_file:
                file_producers[output_file] = ex.execution_id
        
        # Detect dependencies: consumers depend on producers
        for output_file, producer_id in file_producers.items():
            if output_file in file_consumers:
                for consumer_id in file_consumers[output_file]:
                    if consumer_id != producer_id:
                        dependencies[consumer_id].add(producer_id)
        
        # Special rule: training tools depend on profiling/cleaning if they exist
        training_tools = ["train_baseline_models", "hyperparameter_tuning", "train_ensemble_models"]
        prep_tools = ["profile_dataset", "clean_missing_values", "encode_categorical"]
        
        training_execs = [ex for ex in executions if ex.tool_name in training_tools]
        prep_execs = [ex for ex in executions if ex.tool_name in prep_tools]
        
        for train_ex in training_execs:
            for prep_ex in prep_execs:
                # Same file? Training depends on prep
                if train_ex.arguments.get("file_path") == prep_ex.arguments.get("file_path"):
                    dependencies[train_ex.execution_id].add(prep_ex.execution_id)
        
        return dependencies
    
    def get_execution_batches(self, executions: List[ToolExecution]) -> List[List[ToolExecution]]:
        """
        Group executions into batches that can run in parallel.
        
        Returns:
            List of batches, where each batch contains independent tools
        """
        dependencies = self.detect_dependencies(executions)
        
        # Topological sort to get execution order
        batches: List[List[ToolExecution]] = []
        completed: Set[str] = set()
        remaining = {ex.execution_id: ex for ex in executions}
        
        while remaining:
            # Find all tools with satisfied dependencies
            ready = []
            for exec_id, ex in remaining.items():
                deps = dependencies[exec_id]
                if deps.issubset(completed):
                    ready.append(ex)
            
            if not ready:
                # Circular dependency or error - add remaining as single batch
                print("⚠️ Warning: Possible circular dependency detected")
                batches.append(list(remaining.values()))
                break
            
            # Add ready tools as a batch
            batches.append(ready)
            
            # Mark as completed
            for ex in ready:
                completed.add(ex.execution_id)
                del remaining[ex.execution_id]
        
        return batches


class ParallelToolExecutor:
    """
    Executes tools in parallel while respecting dependencies and resource limits.
    
    Features:
    - Automatic dependency detection
    - Weight-based resource management (limit heavy tools)
    - Progress reporting for parallel executions
    - Error isolation (one tool failure doesn't crash others)
    """
    
    def __init__(self, max_heavy_concurrent: int = 1, max_medium_concurrent: int = 2,
                 max_light_concurrent: int = 5):
        """
        Initialize parallel executor.
        
        Args:
            max_heavy_concurrent: Max heavy tools running simultaneously
            max_medium_concurrent: Max medium tools running simultaneously  
            max_light_concurrent: Max light tools running simultaneously
        """
        self.max_heavy = max_heavy_concurrent
        self.max_medium = max_medium_concurrent
        self.max_light = max_light_concurrent
        
        # Semaphores for resource control
        self.heavy_semaphore = asyncio.Semaphore(max_heavy_concurrent)
        self.medium_semaphore = asyncio.Semaphore(max_medium_concurrent)
        self.light_semaphore = asyncio.Semaphore(max_light_concurrent)
        
        self.dependency_graph = ToolDependencyGraph()
        
        print(f"⚡ Parallel Executor initialized:")
        print(f"   Heavy tools: {max_heavy_concurrent} concurrent")
        print(f"   Medium tools: {max_medium_concurrent} concurrent")
        print(f"   Light tools: {max_light_concurrent} concurrent")
    
    def _get_semaphore(self, weight: ToolWeight) -> asyncio.Semaphore:
        """Get appropriate semaphore for tool weight."""
        if weight == ToolWeight.HEAVY:
            return self.heavy_semaphore
        elif weight == ToolWeight.MEDIUM:
            return self.medium_semaphore
        else:
            return self.light_semaphore
    
    async def _execute_single(self, execution: ToolExecution,
                             execute_func: Callable, 
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute a single tool with resource management.
        
        Args:
            execution: Tool execution details
            execute_func: Function to execute tool (sync)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Execution result
        """
        semaphore = self._get_semaphore(execution.weight)
        
        async with semaphore:
            if progress_callback:
                await progress_callback(f"⚡ Executing {execution.tool_name}", "start")
            
            start_time = time.time()
            
            try:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    execute_func,
                    execution.tool_name,
                    execution.arguments
                )
                
                duration = time.time() - start_time
                
                if progress_callback:
                    await progress_callback(
                        f"✅ {execution.tool_name} completed ({duration:.1f}s)",
                        "complete"
                    )
                
                return {
                    "execution_id": execution.execution_id,
                    "tool_name": execution.tool_name,
                    "success": True,
                    "result": result,
                    "duration": duration
                }
                
            except Exception as e:
                duration = time.time() - start_time
                
                if progress_callback:
                    await progress_callback(
                        f"❌ {execution.tool_name} failed: {str(e)[:100]}",
                        "error"
                    )
                
                return {
                    "execution_id": execution.execution_id,
                    "tool_name": execution.tool_name,
                    "success": False,
                    "error": str(e),
                    "duration": duration
                }
    
    async def execute_batch(self, batch: List[ToolExecution],
                          execute_func: Callable,
                          progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Execute a batch of independent tools in parallel.
        
        Args:
            batch: List of tool executions (no dependencies between them)
            execute_func: Sync function to execute tools
            progress_callback: Optional progress callback
        
        Returns:
            List of execution results
        """
        print(f"⚡ Parallel batch: {len(batch)} tools")
        for ex in batch:
            print(f"   - {ex.tool_name} ({ex.weight.name})")
        
        # Execute all in parallel
        tasks = [
            self._execute_single(ex, execute_func, progress_callback)
            for ex in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "execution_id": batch[i].execution_id,
                    "tool_name": batch[i].tool_name,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_all(self, executions: List[ToolExecution],
                         execute_func: Callable,
                         progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Execute all tools with automatic dependency resolution and parallelization.
        
        Args:
            executions: List of all tool executions
            execute_func: Sync function to execute tools
            progress_callback: Optional progress callback
        
        Returns:
            List of all execution results in order
        """
        if not executions:
            return []
        
        # Get execution batches (respecting dependencies)
        batches = self.dependency_graph.get_execution_batches(executions)
        
        print(f"⚡ Execution plan: {len(batches)} batches for {len(executions)} tools")
        
        all_results = []
        
        for i, batch in enumerate(batches):
            print(f"\n📦 Batch {i+1}/{len(batches)}")
            batch_results = await self.execute_batch(batch, execute_func, progress_callback)
            all_results.extend(batch_results)
        
        return all_results
    
    def classify_tools(self, tool_calls: List[Dict[str, Any]]) -> List[ToolExecution]:
        """
        Convert tool calls to ToolExecution objects with weights.
        
        Args:
            tool_calls: List of tool calls from LLM
        
        Returns:
            List of ToolExecution objects
        """
        executions = []
        
        for i, call in enumerate(tool_calls):
            tool_name = call.get("name") or call.get("tool_name")
            arguments = call.get("arguments", {})
            
            # Get weight
            weight = TOOL_WEIGHTS.get(tool_name, ToolWeight.MEDIUM)
            
            execution = ToolExecution(
                tool_name=tool_name,
                arguments=arguments,
                weight=weight,
                dependencies=set(),  # Will be computed by dependency graph
                execution_id=f"{tool_name}_{i}"
            )
            
            executions.append(execution)
        
        return executions


# Global parallel executor
_parallel_executor = None

def get_parallel_executor() -> ParallelToolExecutor:
    """Get or create global parallel executor."""
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = ParallelToolExecutor()
    return _parallel_executor
