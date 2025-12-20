"""
Titanic Example - Demonstrating the complete Data Science Copilot workflow
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator import DataScienceCopilot
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    """
    Complete example using the Titanic dataset.
    
    This demonstrates the full workflow:
    1. Dataset profiling
    2. Quality issue detection
    3. Data cleaning
    4. Feature engineering
    5. Model training
    6. Report generation
    """
    
    console.print(Panel.fit(
        "🚢 Titanic Survival Prediction - Complete Workflow Example",
        style="bold blue"
    ))
    
    # Setup
    titanic_path = "./data/titanic.csv"
    
    # Check if dataset exists
    if not Path(titanic_path).exists():
        console.print("\n[yellow]⚠ Titanic dataset not found at ./data/titanic.csv[/yellow]")
        console.print("[yellow]Please download it from: https://www.kaggle.com/c/titanic/data[/yellow]")
        console.print("[yellow]Or place your own CSV file in the data directory[/yellow]\n")
        
        # Use a sample path instead
        console.print("[blue]Using sample dataset path for demonstration...[/blue]\n")
        titanic_path = "your_dataset.csv"  # User should replace this
    
    # Initialize copilot
    console.print("\n[bold]Step 1: Initialize Data Science Copilot[/bold]")
    try:
        copilot = DataScienceCopilot(reasoning_effort="medium")
        console.print("[green]✓ Copilot initialized successfully[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        console.print("[yellow]Make sure to set GROQ_API_KEY in .env file[/yellow]")
        return
    
    # Define the task
    task_description = """
    Analyze the Titanic dataset and build a model to predict passenger survival.
    
    Key objectives:
    1. Understand the data structure and identify quality issues
    2. Handle missing values appropriately
    3. Engineer relevant features from available data (e.g., family size, titles from names)
    4. Train and compare multiple baseline models
    5. Identify the most important features for prediction
    6. Provide recommendations for improvement
    
    Target: Achieve competitive performance (aim for 50-70th percentile on Kaggle leaderboard)
    """
    
    target_column = "Survived"
    
    console.print("\n[bold]Step 2: Run Complete Analysis Workflow[/bold]")
    console.print(f"Dataset: {titanic_path}")
    console.print(f"Target: {target_column}")
    console.print(f"Task: Predict passenger survival\n")
    
    # Run analysis
    try:
        result = copilot.analyze(
            file_path=titanic_path,
            task_description=task_description,
            target_col=target_column,
            use_cache=True,
            max_iterations=15  # Allow more iterations for complex workflow
        )
        
        # Display results
        if result["status"] == "success":
            console.print("\n[green]✓ Analysis Complete![/green]\n")
            
            # Display summary
            console.print(Panel(
                result["summary"],
                title="📋 Final Analysis Summary",
                border_style="green"
            ))
            
            # Display workflow steps
            console.print("\n[bold]🔧 Workflow Steps Executed:[/bold]")
            for i, step in enumerate(result["workflow_history"], 1):
                tool = step["tool"]
                success = step["result"].get("success", False)
                icon = "✓" if success else "✗"
                color = "green" if success else "red"
                console.print(f"{i}. [{color}]{icon}[/{color}] {tool}")
            
            # Display statistics
            console.print(f"\n[bold]📊 Execution Statistics:[/bold]")
            console.print(f"  Total Iterations: {result['iterations']}")
            console.print(f"  API Calls Made: {result['api_calls']}")
            console.print(f"  Execution Time: {result['execution_time']}s")
            
            # Check for trained models
            console.print("\n[bold]🤖 Model Training Results:[/bold]")
            for step in result["workflow_history"]:
                if step["tool"] == "train_baseline_models":
                    if step["result"].get("success"):
                        models_result = step["result"]["result"]
                        best_model = models_result.get("best_model", {})
                        console.print(f"  Best Model: {best_model.get('name')}")
                        console.print(f"  Score: {best_model.get('score'):.4f}")
                        console.print(f"  Model Path: {best_model.get('model_path')}")
            
            # Save results
            output_file = "./outputs/reports/titanic_analysis.json"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            console.print(f"\n[cyan]💾 Full results saved to: {output_file}[/cyan]")
            
            # Next steps
            console.print("\n[bold]🎯 Next Steps:[/bold]")
            console.print("  1. Review the generated models in ./outputs/models/")
            console.print("  2. Check data quality reports in ./outputs/reports/")
            console.print("  3. Examine cleaned datasets in ./outputs/data/")
            console.print("  4. Use the best model for predictions on new data")
            
        elif result["status"] == "error":
            console.print(f"\n[red]✗ Analysis failed: {result['error']}[/red]")
            console.print(f"Error type: {result['error_type']}")
            
        else:
            console.print(f"\n[yellow]⚠ Analysis incomplete: {result.get('message')}[/yellow]")
    
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
    
    # Cache statistics
    console.print("\n[bold]📦 Cache Statistics:[/bold]")
    cache_stats = copilot.get_cache_stats()
    console.print(f"  Valid Entries: {cache_stats['valid_entries']}")
    console.print(f"  Cache Size: {cache_stats['size_mb']} MB")


if __name__ == "__main__":
    main()
