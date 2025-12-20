"""
AI Agent Data Scientist - Interactive Chat UI
==============================================

A simple web interface to interact with your AI Agent.
Upload datasets, ask questions, and get AI-powered insights!
"""

import gradio as gr
import sys
import os
import shutil
from pathlib import Path
import traceback

# Add src to path
sys.path.append('src')

from tools.data_profiling import profile_dataset, detect_data_quality_issues
from tools.model_training import train_baseline_models

# Try to import AI agent (optional)
try:
    from orchestrator import DataScienceCopilot
    agent = DataScienceCopilot()
    AI_ENABLED = True
    print("✅ AI Agent loaded successfully!")
    print(f"📊 Model: {agent.model}")
    print(f"🔧 Tools available: {len(agent.tool_functions)}")
except Exception as e:
    print(f"ℹ️  Running in manual mode (AI agent not available)")
    print(f"   Error: {str(e)}")
    print("💡 You can still use all the quick actions and tools!")
    AI_ENABLED = False
    agent = None

# Store uploaded file path
current_file = None
current_profile = None
last_agent_response = None  # Store last agent response for visualization extraction


# Helper functions for Gradio 6.x message format
def add_message(history, role, content):
    """Add a message to history in Gradio 6.x format."""
    if history is None:
        history = []
    history.append({"role": role, "content": content})
    return history


def add_user_message(history, content):
    """Add a user message to history."""
    return add_message(history, "user", content)


def add_assistant_message(history, content):
    """Add an assistant message to history."""
    return add_message(history, "assistant", content)


def update_last_assistant_message(history, content):
    """Update the last assistant message in history."""
    if history and len(history) > 0 and history[-1].get("role") == "assistant":
        history[-1]["content"] = content
    return history


def get_last_user_content(history):
    """Get the content of the last user message."""
    if history:
        for msg in reversed(history):
            if msg.get("role") == "user":
                return msg.get("content", "")
    return ""


def analyze_dataset(file, user_message, history):
    """Process uploaded dataset(s) and user message. Supports single or multiple file uploads."""
    global current_file, current_profile, last_agent_response
    
    # Initialize with empty plot list (will collect PNG file paths)
    plots_paths = []
    html_reports = []  # Initialize HTML reports list
    
    # Initialize history if None
    if history is None:
        history = []
    
    # Debug: Log the call
    print(f"[DEBUG] analyze_dataset called - file: {file is not None}, message: '{user_message}', current_file: {current_file}")
    
    try:
        # Handle file uploads (single or multiple)
        if file is not None:
            # file can be a single filepath or a list of filepaths
            files_to_process = file if isinstance(file, list) else [file]
            
            # Filter out None values
            files_to_process = [f for f in files_to_process if f is not None]
            
            if len(files_to_process) > 0:
                print(f"[DEBUG] Processing {len(files_to_process)} file(s) upload")
                
                # Copy all files to simpler paths
                os.makedirs("./temp", exist_ok=True)
                processed_files = []
                seen_files = {}  # Track files by content hash to detect duplicates
                duplicate_count = 0
                
                for uploaded_file in files_to_process:
                    simple_filename = Path(uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file).name
                    file_source = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
                    
                    # Calculate file hash to detect duplicates (even with different names)
                    import hashlib
                    hasher = hashlib.md5()
                    with open(file_source, 'rb') as f:
                        # Read file in chunks to handle large files efficiently
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)
                    file_hash = hasher.hexdigest()
                    
                    # Check if this exact file was already uploaded
                    if file_hash in seen_files:
                        print(f"[DEBUG] Duplicate file detected: {simple_filename} (same as {seen_files[file_hash]})")
                        duplicate_count += 1
                        continue  # Skip duplicate
                    
                    # Not a duplicate - process it
                    simple_path = f"./temp/{simple_filename}"
                    
                    # Handle filename collision (different files with same name)
                    if os.path.exists(simple_path):
                        # Check if existing file is the same (by comparing with already processed files)
                        existing_in_processed = simple_path in processed_files
                        if not existing_in_processed:
                            # Different file with same name - add suffix
                            base_name = Path(simple_filename).stem
                            extension = Path(simple_filename).suffix
                            counter = 1
                            while os.path.exists(f"./temp/{base_name}_{counter}{extension}"):
                                counter += 1
                            simple_filename = f"{base_name}_{counter}{extension}"
                            simple_path = f"./temp/{simple_filename}"
                            print(f"[DEBUG] Filename collision - renamed to: {simple_filename}")
                    
                    shutil.copy2(file_source, simple_path)
                    processed_files.append(simple_path)
                    seen_files[file_hash] = simple_filename
                    print(f"[DEBUG] Copied file to: {simple_path}")
                
                # Set current_file to the first file (for single-file operations)
                # For multi-file operations, the agent will use all files from ./temp/
                current_file = processed_files[0] if processed_files else None
                
                # Only show file upload response if there's no user message
                if not (user_message and user_message.strip()):
                    if len(processed_files) == 0:
                        # All files were duplicates
                        response = f"⚠️ **No New Files Uploaded**\n\n"
                        response += f"All {len(files_to_process)} file(s) were duplicates of already uploaded files.\n\n"
                        response += "Your previously uploaded dataset is still active."
                    elif len(processed_files) == 1:
                        # Single file upload - show detailed profile
                        response = f"📊 **Dataset Uploaded Successfully!**\n\n"
                        if duplicate_count > 0:
                            response += f"ℹ️ *({duplicate_count} duplicate file(s) were skipped)*\n\n"
                        response += f"**File:** {Path(current_file).name}\n\n"
                        
                        # Get basic profile
                        profile = profile_dataset(current_file)
                        current_profile = profile
                        
                        response += f"**Dataset Overview:**\n"
                        response += f"- Rows: {profile['shape']['rows']:,}\n"
                        response += f"- Columns: {profile['shape']['columns']}\n"
                        
                        # Handle memory_usage (can be float or dict)
                        memory = profile.get('memory_usage', 0)
                        if isinstance(memory, dict):
                            memory = memory.get('total_mb', 0)
                        response += f"- Memory: {memory:.2f} MB\n\n"
                        
                        response += f"**Column Types:**\n"
                        response += f"- Numeric: {len(profile['column_types']['numeric'])} columns\n"
                        response += f"- Categorical: {len(profile['column_types']['categorical'])} columns\n"
                        response += f"- Datetime: {len(profile['column_types']['datetime'])} columns\n\n"
                        
                        # Check data quality
                        quality = detect_data_quality_issues(current_file)
                        if quality['critical']:
                            response += f"🔴 **Critical Issues:** {len(quality['critical'])}\n"
                            for issue in quality['critical'][:3]:
                                response += f"  - {issue['message']}\n"
                        if quality['warning']:
                            response += f"🟡 **Warnings:** {len(quality['warning'])}\n"
                            for issue in quality['warning'][:3]:
                                response += f"  - {issue['message']}\n"
                    else:
                        # Multiple files uploaded
                        response = f"📊 **{len(processed_files)} Datasets Uploaded Successfully!**\n\n"
                        if duplicate_count > 0:
                            response += f"ℹ️ *({duplicate_count} duplicate file(s) were skipped)*\n\n"
                        response += f"**Files:**\n"
                        for i, fp in enumerate(processed_files, 1):
                            response += f"{i}. {Path(fp).name}\n"
                        response += f"\n**💡 You can now use multi-dataset operations!**\n\n"
                    
                    response += f"\n\n💬 **What would you like to do with {'this dataset' if len(processed_files) == 1 else 'these datasets'}?**\n\n"
                    response += "You can ask me to:\n"
                    if len(processed_files) > 1:
                        response += "- **Merge these datasets** (e.g., 'merge customers and orders on customer_id')\n"
                        response += "- **Combine/concatenate** them (e.g., 'combine all monthly sales files')\n"
                    response += "- Train a classification or regression model\n"
                    response += "- Analyze specific columns\n"
                    response += "- Detect outliers\n"
                    response += "- Engineer features\n"
                    response += "- Generate predictions\n"
                    response += "- And much more!\n"
                    
                    # Add assistant message to history
                    history = add_assistant_message(history, response)
                    yield history, "", [], []
                    return
                # If user uploaded file AND sent a message, don't return - continue to process the message
                elif user_message and user_message.strip():
                    # Continue processing the message below
                    pass
        
        # If user sends a message about the current file
        print(f"[DEBUG] Checking message conditions: user_message={bool(user_message and user_message.strip())}, current_file={bool(current_file)}")
        if user_message and user_message.strip() and current_file:
            print(f"[DEBUG] User message detected. AI_ENABLED={AI_ENABLED}, agent={agent is not None}")
            if AI_ENABLED and agent:
                print(f"[DEBUG] Entering AI Agent block...")
                try:
                    # Show immediate processing message
                    print(f"🤖 AI Agent analyzing: {user_message}")
                    history = add_user_message(history, user_message)
                    history = add_assistant_message(history, "🤖 **AI Agent is thinking...**\n\n⏳ Analyzing your request and planning the workflow...")
                    yield history, "", [], []
                    
                    # Use the AI agent to process the request
                    print(f"📂 File path: {current_file}")
                    print(f"📝 Task: {user_message}")
                    print(f"🚀 Calling agent.analyze()...")
                    
                    agent_response = agent.analyze(
                        file_path=current_file,
                        task_description=user_message,
                        use_cache=False,  # Disable cache to avoid dict hashing issues
                        stream=False
                    )
                    
                    print(f"✅ Agent response received: {agent_response.get('status', 'unknown')}")
                    
                    # Store agent response for visualization extraction
                    last_agent_response = agent_response
                    
                    # Format the response
                    if agent_response.get('status') == 'success':
                        response = f"🤖 **AI Agent Analysis Complete!**\n\n"
                        response += f"{agent_response.get('summary', '')}\n\n"
                        
                        if 'workflow_history' in agent_response and agent_response['workflow_history']:
                            response += f"**Execution Summary:**\n"
                            response += f"- Tools Executed: {len(agent_response['workflow_history'])}\n"
                            response += f"- Iterations: {agent_response.get('iterations', 0)}\n"
                            response += f"- Time: {agent_response.get('execution_time', 0):.1f}s\n\n"
                            
                            # Find and display MODEL TRAINING RESULTS with ALL METRICS
                            model_results = None
                            for step in agent_response['workflow_history']:
                                if step.get('tool') == 'train_baseline_models':
                                    result = step.get('result', {})
                                    if isinstance(result, dict) and 'result' in result:
                                        model_results = result['result']
                                    elif isinstance(result, dict):
                                        model_results = result
                                    break
                            
                            if model_results and 'models' in model_results:
                                response += f"## 🎯 Model Training Results\n\n"
                                task_type = model_results.get('task_type', 'unknown')
                                response += f"**Task Type:** {task_type.title()}\n"
                                response += f"**Features:** {model_results.get('n_features', 0)}\n"
                                response += f"**Training Samples:** {model_results.get('train_size', 0):,}\n"
                                response += f"**Test Samples:** {model_results.get('test_size', 0):,}\n\n"
                                
                                # Show ALL models tested
                                response += "### 📊 All Models Tested:\n\n"
                                models_data = model_results.get('models', {})
                                
                                for model_name, model_info in models_data.items():
                                    if 'test_metrics' in model_info:
                                        metrics = model_info['test_metrics']
                                        response += f"**{model_name}:**\n"
                                        
                                        if task_type == 'classification':
                                            response += f"- Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                                            response += f"- Precision: {metrics.get('precision', 0):.4f}\n"
                                            response += f"- Recall: {metrics.get('recall', 0):.4f}\n"
                                            response += f"- F1 Score: {metrics.get('f1', 0):.4f}\n"
                                        else:
                                            response += f"- R² Score: {metrics.get('r2', 0):.4f}\n"
                                            response += f"- RMSE: {metrics.get('rmse', 0):.2f}\n"
                                            response += f"- MAE: {metrics.get('mae', 0):.2f}\n"
                                            response += f"- MAPE: {metrics.get('mape', 0):.2f}%\n"
                                        response += "\n"
                                
                                # Highlight BEST MODEL
                                best_model = model_results.get('best_model', {})
                                if best_model and best_model.get('name'):
                                    response += f"### 🏆 Best Model: **{best_model['name']}**\n"
                                    response += f"Score: {best_model.get('score', 0):.4f}\n\n"
                            
                            # Show workflow execution summary
                            response += "### 🔧 Workflow Steps:\n"
                            for i, step in enumerate(agent_response['workflow_history'], 1):
                                tool_name = step['tool']
                                success = step['result'].get('success', False)
                                icon = "✅" if success else "❌"
                                response += f"{i}. {icon} {tool_name}\n"
                            response += "\n"
                            
                            # Check for plots AND reports in workflow results
                            html_reports = []  # Separate list for HTML reports
                            
                            for step in agent_response['workflow_history']:
                                result = step.get('result', {})
                                
                                # Deep search for plots and reports in nested results
                                def find_plots_and_reports(obj, plots_list, reports_list):
                                    if isinstance(obj, dict):
                                        # Check direct plot/report keys
                                        for key in ['plot_path', 'plot_file', 'output_path', 'html_path', 'report_path',
                                                   'plots', 'plot_paths', 'performance_plots', 'feature_importance_plot']:
                                            if key in obj and obj[key]:
                                                if isinstance(obj[key], list):
                                                    for path in obj[key]:
                                                        if isinstance(path, str) and os.path.exists(path):
                                                            if path.endswith('.html'):
                                                                # Check if it's a report (in reports folder) or interactive plot
                                                                if '/reports/' in path or 'report' in Path(path).stem.lower():
                                                                    reports_list.append(path)
                                                                else:
                                                                    reports_list.append(path)  # Interactive plots also go to reports
                                                            elif path.endswith(('.png', '.jpg', '.jpeg')):
                                                                plots_list.append(path)
                                                elif isinstance(obj[key], str) and os.path.exists(obj[key]):
                                                    if obj[key].endswith('.html'):
                                                        if '/reports/' in obj[key] or 'report' in Path(obj[key]).stem.lower():
                                                            reports_list.append(obj[key])
                                                        else:
                                                            reports_list.append(obj[key])
                                                    elif obj[key].endswith(('.png', '.jpg', '.jpeg')):
                                                        plots_list.append(obj[key])
                                        # Recursively search nested dicts
                                        for value in obj.values():
                                            find_plots_and_reports(value, plots_list, reports_list)
                                
                                find_plots_and_reports(result, plots_paths, html_reports)
                            
                            # Remove duplicates while preserving order
                            plots_paths = list(dict.fromkeys(plots_paths))
                            html_reports = list(dict.fromkeys(html_reports))
                            
                            # Display visualization and report information in response
                            if plots_paths or html_reports:
                                response += f"## 📊 Generated Outputs\n\n"
                                
                                if plots_paths:
                                    response += f"### 📈 Visualizations ({len(plots_paths)} plots)\n"
                                    response += "✅ Plots are displayed in the **Visualization Gallery** below!\n\n"
                                    
                                    # List plot files
                                    for i, plot_path in enumerate(plots_paths[:10], 1):
                                        try:
                                            plot_name = Path(plot_path).stem.replace('_', ' ').title()
                                            rel_path = os.path.relpath(plot_path, '.')
                                            response += f"{i}. 📊 **{plot_name}**\n"
                                            response += f"   📁 `{rel_path}`\n\n"
                                        except Exception as e:
                                            response += f"{i}. ❌ Error: {str(e)}\n"
                                
                                if html_reports:
                                    response += f"### 📋 Reports & Interactive Plots ({len(html_reports)} files)\n"
                                    response += "✅ Reports are displayed in the **Reports Viewer** below!\n\n"
                                    
                                    # List report files
                                    for i, report_path in enumerate(html_reports[:10], 1):
                                        try:
                                            report_name = Path(report_path).stem.replace('_', ' ').title()
                                            rel_path = os.path.relpath(report_path, '.')
                                            file_size = os.path.getsize(report_path) / 1024  # KB
                                            response += f"{i}. 📄 **{report_name}**\n"
                                            response += f"   📁 `{rel_path}` ({file_size:.1f} KB)\n\n"
                                        except Exception as e:
                                            response += f"{i}. ❌ Error: {str(e)}\n"
                            else:
                                response += "ℹ️ No visualizations or reports were generated in this workflow.\n"
                    else:
                        response = f"⚠️ **AI Agent Status:** {agent_response.get('status', 'unknown')}\n\n"
                        response += f"{agent_response.get('message', agent_response.get('error', 'Unknown error'))}\n"
                    
                    # Update the last assistant message with the response
                    history = update_last_assistant_message(history, response)
                    
                    # Return plot paths for gallery and html_reports for HTML viewer
                    # Store html_reports in a format the HTML component can use
                    yield history, "", plots_paths if plots_paths else [], html_reports if html_reports else []
                    return
                except Exception as e:
                    import sys
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    response = f"⚠️ **AI Agent Error:**\n\n"
                    response += f"**Error Type:** {exc_type.__name__}\n\n"
                    response += f"**Error Message:** {str(e)}\n\n"
                    response += f"**Full Traceback:**\n```python\n{traceback.format_exc()}\n```\n\n"
                    response += "💡 **Fallback Options:**\n"
                    response += "- Use the **Quick Train** feature on the right\n"
                    response += "- Try manual commands: `profile`, `quality`, `columns`\n"
                    # Update the last assistant message with error
                    history = update_last_assistant_message(history, response)
                    yield history, "", plots_paths if plots_paths else []
                    return
            else:
                # Manual mode - Handle commands directly
                user_msg_lower = user_message.lower().strip()
                
                # Handle simple commands manually
                if 'profile' in user_msg_lower:
                    response = "📊 **Dataset Profile:**\n\n"
                    if current_profile:
                        response += f"**Shape:** {current_profile['shape']['rows']:,} rows × {current_profile['shape']['columns']} columns\n\n"
                        response += f"**Column Types:**\n"
                        response += f"- Numeric: {len(current_profile['column_types']['numeric'])} columns\n"
                        response += f"- Categorical: {len(current_profile['column_types']['categorical'])} columns\n"
                        response += f"- Datetime: {len(current_profile['column_types']['datetime'])} columns\n\n"
                        response += f"**Overall Stats:**\n"
                        response += f"- Total cells: {current_profile['overall_stats']['total_cells']:,}\n"
                        response += f"- Null values: {current_profile['overall_stats']['total_nulls']} ({current_profile['overall_stats']['null_percentage']:.1f}%)\n"
                        response += f"- Duplicates: {current_profile['overall_stats']['duplicate_rows']}\n"
                    else:
                        response += "Profile information is available at the top of the chat!"
                        
                elif 'quality' in user_msg_lower or 'issues' in user_msg_lower:
                    quality = detect_data_quality_issues(current_file)
                    response = "🔍 **Data Quality Report:**\n\n"
                    
                    if quality['critical']:
                        response += f"🔴 **Critical Issues:** {len(quality['critical'])}\n"
                        for issue in quality['critical']:
                            response += f"  • {issue['message']}\n"
                        response += "\n"
                    
                    if quality['warning']:
                        response += f"🟡 **Warnings:** {len(quality['warning'])}\n"
                        for issue in quality['warning'][:5]:  # Show first 5
                            response += f"  • {issue['message']}\n"
                        if len(quality['warning']) > 5:
                            response += f"  • ... and {len(quality['warning']) - 5} more\n"
                        response += "\n"
                    
                    if quality['info']:
                        response += f"🔵 **Info:** {len(quality['info'])} observations\n"
                    
                    if not quality['critical'] and not quality['warning'] and not quality['info']:
                        response += "✅ No issues detected! Your data looks good.\n"
                        
                elif 'columns' in user_msg_lower or 'column' in user_msg_lower:
                    if current_profile:
                        response = "📋 **Dataset Columns:**\n\n"
                        for col, info in current_profile['columns'].items():
                            nulls = info.get('null_count', 0)
                            null_pct = (nulls / current_profile['shape']['rows'] * 100) if current_profile['shape']['rows'] > 0 else 0
                            response += f"• **{col}** ({info['type']})\n"
                            response += f"  - Nulls: {nulls} ({null_pct:.1f}%)\n"
                            if 'unique' in info:
                                response += f"  - Unique: {info['unique']}\n"
                    else:
                        response = "📋 **Columns:** Please upload a file first to see column information."
                
                elif 'help' in user_msg_lower:
                    response = "💡 **Available Commands:**\n\n"
                    response += "**Manual Commands:**\n"
                    response += "• `profile` - Show detailed dataset statistics\n"
                    response += "• `quality` - Check data quality issues\n"
                    response += "• `columns` - List all columns with details\n"
                    response += "• `help` - Show this help message\n\n"
                    response += "**Quick Actions:**\n"
                    response += "• Use the **Quick Train** panel on the right to train models\n"
                    response += "• Check **Dataset Info** in the sidebar for quick stats\n"
                
                else:
                    # Default response for unrecognized commands
                    response = f"💬 **You said:** {user_message}\n\n"
                    response += "⚠️ AI agent is not available. I can respond to these commands:\n\n"
                    response += "• `profile` - Show detailed statistics\n"
                    response += "• `quality` - Check data quality\n"
                    response += "• `columns` - List all columns\n"
                    response += "• `help` - Show available commands\n\n"
                    response += "**Or use Quick Train** on the right to train models directly!\n"
                
                # Add user message and assistant response
                history = add_user_message(history, user_message)
                history = add_assistant_message(history, response)
                yield history, "", [], []
                return
        
        # If no file is uploaded yet
        if user_message and user_message.strip() and not current_file:
            response = "⚠️ **Please upload a dataset first!**\n\n"
            response += "Click the 'Upload Dataset' button above and select a CSV or Parquet file."
            # Add user message and assistant response
            history = add_user_message(history, user_message)
            history = add_assistant_message(history, response)
            yield history, "", [], []
            return
            
    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}\n\n"
        error_msg += "**Traceback:**\n```\n" + traceback.format_exc() + "\n```"
        if user_message:
            # Check if we already added the user message
            last_user = get_last_user_content(history)
            if last_user != user_message:
                history = add_user_message(history, user_message)
            history = add_assistant_message(history, error_msg)
        else:
            history = add_assistant_message(history, error_msg)
        yield history, "", [], []
        return
    
    # Default return if nothing matched
    yield history, "", [], []


def quick_profile(file):
    """Quick profile display in the sidebar."""
    if file is None:
        return "No file uploaded yet."
    
    try:
        profile = profile_dataset(file.name)
        
        info = f"**{Path(file.name).name}**\n\n"
        info += f"📊 {profile['shape']['rows']:,} rows × {profile['shape']['columns']} cols\n\n"
        info += f"**Columns:**\n"
        for col, col_info in list(profile['columns'].items())[:10]:
            info += f"- {col} ({col_info['type']})\n"
        
        if len(profile['columns']) > 10:
            info += f"- ... and {len(profile['columns']) - 10} more\n"
        
        return info
    except Exception as e:
        return f"Error: {str(e)}"


def train_model_ui(file, target_col, model_type, test_size, progress=gr.Progress()):
    """Train a model directly from the UI."""
    if file is None:
        return "⚠️ Please upload a dataset first!"
    
    if not target_col:
        return "⚠️ Please specify a target column!"
    
    # Clean up the target column name - remove surrounding quotes if present
    target_col = target_col.strip().strip("'").strip('"')
    
    try:
        # Show progress
        progress(0, desc="🔄 Loading dataset...")
        yield "⏳ **Training in progress...**\n\n📊 Loading dataset..."
        
        import time
        time.sleep(0.5)  # Brief pause for UI feedback
        
        progress(0.2, desc="🔄 Preparing data...")
        yield "⏳ **Training in progress...**\n\n📊 Dataset loaded\n🔄 Preparing data..."
        
        time.sleep(0.3)
        # Determine problem type
        problem_type = "classification" if model_type == "Classification" else "regression"
        
        progress(0.4, desc="🤖 Training models...")
        yield "⏳ **Training in progress...**\n\n📊 Dataset loaded\n✅ Data prepared\n🤖 Training multiple models..."
        
        # Train baseline models
        result = train_baseline_models(
            file.name,
            target_col=target_col,
            task_type=problem_type,
            test_size=test_size
        )
        
        progress(0.9, desc="📊 Evaluating results...")
        
        # Check if training was successful
        if result.get('status') == 'error':
            yield f"❌ **Training Failed**\n\n{result.get('message', 'Unknown error')}"
            return
        
        if 'best_model' not in result:
            yield f"❌ **Training Failed**\n\nNo models were successfully trained. Result: {result}"
            return
        
        # Get the best model
        best_model_name = result['best_model']['name']
        if not best_model_name:
            yield f"❌ **Training Failed**\n\nNo model could be selected as best model."
            return
            
        best_model_info = result['models'][best_model_name]
        best_metrics = best_model_info.get('test_metrics', {})
        
        output = f"✅ **Model Training Complete!**\n\n"
        output += f"## 🏆 Best Model: **{best_model_name}**\n\n"
        
        output += f"**Dataset Info:**\n"
        output += f"- Features: {result.get('n_features', 0)}\n"
        output += f"- Training samples: {result.get('train_size', 0):,}\n"
        output += f"- Test samples: {result.get('test_size', 0):,}\n\n"
        
        if problem_type == "classification":
            output += f"**Test Metrics:**\n"
            output += f"- ✅ Accuracy: {best_metrics.get('accuracy', 0):.4f}\n"
            output += f"- 🎯 Precision: {best_metrics.get('precision', 0):.4f}\n"
            output += f"- 📊 Recall: {best_metrics.get('recall', 0):.4f}\n"
            output += f"- 🔥 F1 Score: {best_metrics.get('f1', 0):.4f}\n\n"
        else:
            output += f"**Test Metrics:**\n"
            output += f"- 📈 R² Score: {best_metrics.get('r2', 0):.4f}\n"
            output += f"- 📉 RMSE: {best_metrics.get('rmse', 0):.2f}\n"
            output += f"- 📊 MAE: {best_metrics.get('mae', 0):.2f}\n"
            output += f"- 💯 MAPE: {best_metrics.get('mape', 0):.2f}%\n\n"
        
        output += f"## 📊 All Models Comparison:\n\n"
        for model_name, model_info in result['models'].items():
            if 'test_metrics' in model_info:
                test_metrics = model_info['test_metrics']
                indicator = "🏆 " if model_name == best_model_name else "   "
                if problem_type == "classification":
                    f1 = test_metrics.get('f1', 0)
                    acc = test_metrics.get('accuracy', 0)
                    output += f"{indicator}**{model_name}:**\n"
                    output += f"   - F1: {f1:.4f} | Accuracy: {acc:.4f}\n"
                else:
                    r2 = test_metrics.get('r2', 0)
                    rmse = test_metrics.get('rmse', 0)
                    output += f"{indicator}**{model_name}:**\n"
                    output += f"   - R²: {r2:.4f} | RMSE: {rmse:.2f}\n"
            elif 'status' in model_info and model_info['status'] == 'error':
                output += f"   ❌ **{model_name}:** {model_info.get('message', 'Error')}\n"
        
        # Display generated plots if available
        plots_to_show = []
        
        # Check for performance plots
        if 'performance_plots' in result and result['performance_plots']:
            if isinstance(result['performance_plots'], list):
                plots_to_show.extend(result['performance_plots'])
            else:
                plots_to_show.append(result['performance_plots'])
        
        # Check for feature importance plot
        if 'feature_importance_plot' in result and result['feature_importance_plot']:
            plots_to_show.append(result['feature_importance_plot'])
        
        # Embed plots
        if plots_to_show:
            output += f"\n\n📊 **Visualizations:**\n\n"
            for plot_path in plots_to_show:
                if isinstance(plot_path, str) and plot_path.endswith('.html') and os.path.exists(plot_path):
                    try:
                        with open(plot_path, 'r', encoding='utf-8') as f:
                            plot_html = f.read()
                        # Add plot title based on filename
                        plot_name = Path(plot_path).stem.replace('_', ' ').title()
                        output += f"**{plot_name}:**\n"
                        output += f'<iframe srcdoc="{plot_html.replace(chr(34), "&quot;")}" width="100%" height="500" frameborder="0"></iframe>\n\n'
                    except Exception as e:
                        # Fallback to file path
                        output += f"📁 {Path(plot_path).name}: `{plot_path}`\n"
        
        progress(1.0, desc="✅ Complete!")
        yield output
            
    except Exception as e:
        yield f"❌ **Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def clear_conversation():
    """Clear the conversation and reset state."""
    global current_file, current_profile
    current_file = None
    current_profile = None
    return [], None, "", [], ""


def format_html_reports(html_paths):
    """Format HTML reports/plots for display in HTML component."""
    if not html_paths or len(html_paths) == 0:
        return "<div style='text-align:center; padding:40px; color:#666;'>No reports generated yet. Try: 'Generate a quality report' or 'Create interactive visualizations'</div>"
    
    html_output = """
    <style>
        .report-container {
            padding: 20px;
            background: #f8f9fa;
        }
        .report-card {
            margin-bottom: 30px;
            border: 2px solid #dee2e6;
            border-radius: 12px;
            overflow: hidden;
            background: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-weight: bold;
            font-size: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .report-meta {
            font-size: 12px;
            opacity: 0.9;
        }
        .report-iframe {
            width: 100%;
            min-height: 600px;
            border: none;
            background: white;
        }
        .report-footer {
            background: #f8f9fa;
            padding: 10px 20px;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #dee2e6;
        }
    </style>
    <div class="report-container">
    """
    
    html_output += f"<h2 style='color: #667eea; margin-bottom: 20px;'>📋 {len(html_paths)} Report(s) Generated</h2>"
    
    for i, html_path in enumerate(html_paths, 1):
        try:
            # Get file metadata
            file_name = Path(html_path).name
            file_size = os.path.getsize(html_path) / 1024  # KB
            report_title = Path(html_path).stem.replace('_', ' ').title()
            
            # Read the HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Escape the content for embedding
            escaped_content = html_content.replace('\\', '\\\\').replace('"', '&quot;').replace("'", "\\'")
            
            html_output += f"""
            <div class="report-card">
                <div class="report-header">
                    <span>📊 {i}. {report_title}</span>
                    <span class="report-meta">{file_size:.1f} KB</span>
                </div>
                <iframe class="report-iframe" srcdoc="{escaped_content}"></iframe>
                <div class="report-footer">
                    📁 {html_path}
                </div>
            </div>
            """
        except Exception as e:
            html_output += f"""
            <div class="report-card">
                <div class="report-header" style="background: linear-gradient(135deg, #f44336 0%, #e91e63 100%);">
                    <span>❌ Error loading: {Path(html_path).name}</span>
                </div>
                <div style="padding: 20px;">
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p><strong>Path:</strong> {html_path}</p>
                </div>
            </div>
            """
    
    html_output += "</div>"
    
    return html_output


def extract_and_display_plots(agent_response):
    """Extract plots from agent response and format them for display."""
    plots_html = ""
    
    if not agent_response or agent_response.get('status') != 'success':
        return gr.update(value="<p style='text-align:center; color:#666;'>No visualizations generated yet. Upload a dataset and run analysis!</p>")
    
    workflow_history = agent_response.get('workflow_history', [])
    if not workflow_history:
        return gr.update(value="<p style='text-align:center; color:#666;'>No visualizations in this workflow.</p>")
    
    # Find all plots
    plots_paths = []
    
    def find_plots(obj, plots_list):
        if isinstance(obj, dict):
            # Check direct plot keys
            for key in ['plot_path', 'plot_file', 'html_path', 'output_path', 
                       'plots', 'plot_paths', 'performance_plots', 'feature_importance_plot']:
                if key in obj and obj[key]:
                    if isinstance(obj[key], list):
                        for plot_path in obj[key]:
                            if isinstance(plot_path, str) and plot_path.endswith('.html') and os.path.exists(plot_path):
                                plots_list.append(plot_path)
                    elif isinstance(obj[key], str) and obj[key].endswith('.html') and os.path.exists(obj[key]):
                        plots_list.append(obj[key])
            # Recursively search nested dicts
            for value in obj.values():
                find_plots(value, plots_list)
    
    for step in workflow_history:
        result = step.get('result', {})
        find_plots(result, plots_paths)
    
    # Remove duplicates while preserving order
    plots_paths = list(dict.fromkeys(plots_paths))
    
    if not plots_paths:
        return gr.update(value="<p style='text-align:center; color:#666;'>No plots were generated in this analysis.</p>")
    
    # Build HTML gallery
    plots_html = f"""
    <div style='padding: 20px;'>
        <h2 style='color: #1f77b4; margin-bottom: 20px;'>📊 Visualization Gallery ({len(plots_paths)} plots)</h2>
    """
    
    for i, plot_path in enumerate(plots_paths, 1):
        try:
            with open(plot_path, 'r', encoding='utf-8') as f:
                plot_content = f.read()
            
            plot_name = Path(plot_path).stem.replace('_', ' ').title()
            
            plots_html += f"""
            <div style='margin-bottom: 30px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;'>
                <div style='background: linear-gradient(90deg, #1f77b4, #2ca02c); color: white; padding: 10px 15px; font-weight: bold;'>
                    {i}. {plot_name}
                </div>
                <div style='padding: 10px; background: white;'>
                    <iframe srcdoc='{plot_content.replace("'", "&apos;").replace('"', "&quot;")}' 
                            width='100%' height='500' frameborder='0' 
                            style='border: none; border-radius: 5px;'></iframe>
                </div>
                <div style='background: #f8f9fa; padding: 8px 15px; font-size: 12px; color: #666;'>
                    📁 {plot_path}
                </div>
            </div>
            """
        except Exception as e:
            plots_html += f"""
            <div style='margin-bottom: 20px; padding: 15px; border: 1px solid #f44336; border-radius: 5px; background: #ffebee;'>
                <strong>❌ Failed to load: {Path(plot_path).name}</strong><br>
                <small>{str(e)}</small>
            </div>
            """
    
    plots_html += "</div>"
    
    return gr.update(value=plots_html)


# Custom CSS for better visual feedback
custom_css = """
.status-box {
    padding: 10px;
    border-radius: 5px;
    background: linear-gradient(90deg, #e8f5e9 0%, #c8e6c9 100%);
    margin-bottom: 10px;
    text-align: center;
    font-weight: bold;
}
"""

    # Create the Gradio interface
with gr.Blocks(title="AI Agent Data Scientist", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 🤖 AI Agent Data Scientist
    
    Upload your dataset and chat with the AI agent to perform data science tasks!
    
    **Features:**
    - 📊 Automatic dataset profiling
    - 🤖 Natural language queries
    - 🎯 Model training (classification & regression)
    - 🔍 Data quality analysis
    - 📈 Feature engineering
    - 🎨 **NEW:** Automatic visualization generation!
    - And 59 tools total!
    """)
    
    # Store agent response for visualization extraction
    agent_response_state = gr.State(None)
    
    with gr.Row():
        # Left column - Main chat interface
        with gr.Column(scale=2):
            # Status indicator
            status_box = gr.Markdown("🟢 **Ready** - Upload a dataset to begin", elem_classes=["status-box"])
            
            chatbot = gr.Chatbot(
                label="Chat with AI Agent",
                height=450,
                show_label=True,
                avatar_images=(None, "🤖"),
                sanitize_html=False  # Allow HTML content including iframes
            )
            
            with gr.Row():
                file_upload = gr.File(
                    label="📁 Upload Dataset(s) (CSV/Parquet) - Single or Multiple Files",
                    file_types=[".csv", ".parquet"],
                    file_count="multiple",  # Allow multiple file uploads
                    type="filepath"
                )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask anything: 'train a model', 'analyze my data', 'generate visualizations'",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("📤 Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")        # Right column - Quick actions and info
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Dataset Info")
            dataset_info = gr.Markdown("Upload a dataset to see information here.")
            
            gr.Markdown("## 🎯 Quick Train")
            with gr.Group():
                target_column = gr.Textbox(
                    label="Target Column",
                    placeholder="e.g., 'price', 'class', 'label'"
                )
                model_type_choice = gr.Radio(
                    ["Classification", "Regression"],
                    label="Model Type",
                    value="Classification"
                )
                test_size_slider = gr.Slider(
                    0.1, 0.5, 0.3,
                    label="Test Size",
                    step=0.05
                )
                train_btn = gr.Button("🚀 Train Model", variant="primary")
            
            training_output = gr.Markdown("Training results will appear here.")
            
            gr.Markdown("""
            ## 💡 Example Queries
            
            - "Train a classification model to predict [target]"
            - "Show me statistics for [column]"
            - "Detect outliers in the dataset"
            - "What are the most important features?"
            - "Generate a quality report"
            - "Create polynomial features"
            - "Balance the dataset using SMOTE"
            """)
    
    # Visualization Gallery Section (Full Width)
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🎨 Visualization Gallery")
            visualization_gallery = gr.Gallery(
                label="Generated Plots (PNG/JPG)",
                show_label=True,
                elem_id="gallery",
                columns=2,
                height=400
            )
    
    # Reports Viewer Section (Full Width)
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📋 Reports & Interactive Visualizations")
            gr.Markdown("*HTML reports and interactive Plotly charts will be displayed here*")
            reports_viewer = gr.HTML(
                value="<div style='text-align:center; padding:40px; color:#666;'>No reports generated yet. Try: 'Generate a quality report' or 'Create interactive visualizations'</div>",
                elem_id="reports_viewer"
            )
    
    # Create state to hold HTML report paths
    html_reports_state = gr.State([])
    
    # Event handlers with streaming support
    submit_result = submit_btn.click(
        fn=analyze_dataset,
        inputs=[file_upload, user_input, chatbot],
        outputs=[chatbot, user_input, visualization_gallery, html_reports_state],
        show_progress="full"  # Show progress bar
    )
    submit_result.then(
        fn=format_html_reports,
        inputs=[html_reports_state],
        outputs=[reports_viewer]
    )
    
    user_input_result = user_input.submit(
        fn=analyze_dataset,
        inputs=[file_upload, user_input, chatbot],
        outputs=[chatbot, user_input, visualization_gallery, html_reports_state],
        show_progress="full"
    )
    user_input_result.then(
        fn=format_html_reports,
        inputs=[html_reports_state],
        outputs=[reports_viewer]
    )
    
    file_result = file_upload.change(
        fn=analyze_dataset,
        inputs=[file_upload, gr.Textbox(value="", visible=False), chatbot],
        outputs=[chatbot, user_input, visualization_gallery, html_reports_state],
        show_progress="full"
    )
    file_result.then(
        fn=quick_profile,
        inputs=[file_upload],
        outputs=[dataset_info]
    )
    file_result.then(
        fn=format_html_reports,
        inputs=[html_reports_state],
        outputs=[reports_viewer]
    )
    
    train_btn.click(
        fn=train_model_ui,
        inputs=[file_upload, target_column, model_type_choice, test_size_slider],
        outputs=[training_output],
        show_progress="full"  # Show progress bar
    )
    
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, file_upload, user_input, visualization_gallery, reports_viewer]
    )

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting AI Agent Data Scientist Chat UI...")
    print("=" * 70)
    print("\n🌐 The UI will open in your browser automatically.")
    print("💡 If it doesn't, copy the URL shown below.\n")
    
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7865,  # Changed port to avoid conflict
        show_error=True,
        inbrowser=True  # Auto-open browser
    )
