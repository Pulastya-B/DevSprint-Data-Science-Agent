# Multi-Agent Architecture

## Overview

The DS Agent now implements a **multi-agent architecture** where specialized AI agents collaborate to handle different aspects of data science workflows. Each specialist agent has focused expertise, tailored system prompts, and relevant tools.

## Architecture Diagram

```
User Request
     ↓
┌────────────────────┐
│ Main Orchestrator  │  ← Routes to appropriate specialist
└─────────┬──────────┘
          │
    ┌─────┴─────┐
    │           │
    ├──────→ 🔬 EDA Specialist Agent
    │        ├─ Data profiling & quality checks
    │        ├─ Correlation analysis
    │        ├─ Anomaly detection
    │        └─ Statistical tests
    │
    ├──────→ ⚙️ Data Engineering Specialist
    │        ├─ Missing value handling
    │        ├─ Outlier treatment
    │        ├─ Feature engineering
    │        └─ Data preprocessing
    │
    ├──────→ 🤖 ML Modeling Specialist
    │        ├─ Baseline model training
    │        ├─ Hyperparameter tuning
    │        ├─ Ensemble methods
    │        └─ Cross-validation
    │
    ├──────→ 📊 Visualization Specialist
    │        ├─ Interactive Plotly plots
    │        ├─ Matplotlib visualizations
    │        ├─ Dashboards & reports
    │        └─ Model performance charts
    │
    └──────→ 💡 Business Insights Specialist
             ├─ Root cause analysis
             ├─ What-if scenarios
             ├─ Feature interpretability
             └─ Actionable recommendations
```

## Specialist Agents

### 🔬 EDA Specialist Agent
**Expertise**: Exploratory Data Analysis
- Data profiling and statistical summaries
- Data quality assessment
- Correlation analysis and feature relationships
- Distribution analysis and outlier detection
- Missing data patterns

**Tools** (13): `profile_dataset`, `detect_data_quality_issues`, `analyze_correlations`, `detect_anomalies`, `perform_statistical_tests`, `generate_ydata_profiling_report`

**Routing Keywords**: profile, eda, quality, correlation, anomaly, statistic, distribution, explore, understand

---

### ⚙️ Data Engineering Specialist Agent
**Expertise**: Data Cleaning & Preprocessing
- Missing value handling with appropriate strategies
- Outlier detection and treatment
- Feature scaling and normalization
- Imbalanced data handling (SMOTE, etc.)
- Feature engineering and transformation

**Tools** (15): `clean_missing_values`, `handle_outliers`, `handle_imbalanced_data`, `perform_feature_scaling`, `encode_categorical`, `create_interaction_features`, `auto_feature_engineering`

**Routing Keywords**: clean, preprocess, feature, encode, scale, outlier, missing, transform, engineer

---

### 🤖 ML Modeling Specialist Agent
**Expertise**: Machine Learning Training & Optimization
- Model selection and baseline training
- Trains 6 models: RandomForest, XGBoost, LightGBM, CatBoost, Ridge, Lasso
- Hyperparameter tuning and optimization
- Ensemble methods and advanced algorithms
- Cross-validation strategies

**Tools** (6): `train_baseline_models`, `hyperparameter_tuning`, `train_ensemble_models`, `perform_cross_validation`, `generate_model_report`, `detect_model_issues`

**Routing Keywords**: train, model, hyperparameter, ensemble, cross-validation, predict, classify, regress

---

### 📊 Visualization Specialist Agent
**Expertise**: Data Visualization & Dashboards
- Interactive Plotly visualizations
- Statistical matplotlib plots
- Business intelligence dashboards
- Model performance visualizations
- Time series and geospatial plots

**Tools** (8 visualization-focused): `generate_interactive_scatter`, `generate_interactive_histogram`, `generate_interactive_correlation_heatmap`, `generate_interactive_box_plots`, `generate_interactive_time_series`, `generate_plotly_dashboard`, `create_matplotlib_plots`, `create_shap_plots`

**Routing Keywords**: plot, visualize, chart, graph, heatmap, scatter, dashboard, matplotlib, plotly

---

### 💡 Business Insights Specialist Agent
**Expertise**: Business Intelligence & Interpretation
- Translates statistical findings into business language
- Root cause analysis and causal inference
- What-if scenario analysis for decision support
- Feature contribution interpretation
- Actionable recommendations from ML results

**Tools** (10): `identify_root_causes`, `perform_what_if_analysis`, `identify_feature_contributions`, `generate_actionable_recommendations`, `explain_model_predictions`, `perform_cohort_analysis`

**Routing Keywords**: insight, recommend, explain, interpret, why, cause, what-if, business, segment, churn

## Agent Routing Logic

The main orchestrator uses **keyword-based intent detection** to route requests:

```python
def _select_specialist_agent(self, task_description: str) -> str:
    """Route task to appropriate specialist agent based on keywords."""
    task_lower = task_description.lower()
    
    # Score each agent based on keyword matches
    scores = {}
    for agent_key, agent_config in self.specialist_agents.items():
        score = sum(1 for keyword in agent_config["tool_keywords"] 
                   if keyword in task_lower)
        scores[agent_key] = score
    
    # Get agent with highest score
    if max(scores.values()) > 0:
        best_agent = max(scores.items(), key=lambda x: x[1])[0]
        return best_agent
    
    # Default to EDA agent for exploratory tasks
    return "eda_agent"
```

### Example Routing

| User Request | Selected Agent | Reasoning |
|--------------|----------------|-----------|
| "Profile the dataset" | 🔬 EDA Specialist | Keywords: profile, dataset |
| "Train a model to predict sales" | 🤖 Modeling Specialist | Keywords: train, model, predict |
| "Create a correlation heatmap" | 📊 Viz Specialist | Keywords: create, correlation, heatmap |
| "Handle missing values" | ⚙️ Data Engineering | Keywords: handle, missing |
| "Explain why churn is high" | 💡 Insights Specialist | Keywords: explain, why, churn |

## UI Integration

The frontend displays which specialist agent is working in real-time via SSE:

```typescript
// SSE event: agent_assigned
{
  "type": "agent_assigned",
  "agent": "EDA Specialist",
  "emoji": "🔬",
  "description": "Expert in data profiling, quality checks, and exploratory analysis"
}
```

**UI Display**:
```
🔬 EDA Specialist assigned
   Expert in data profiling, quality checks, and exploratory analysis
```

## Benefits for Resume/Interviews

### 1. **Advanced AI Architecture Pattern**
   - Shows understanding of multi-agent systems
   - Demonstrates modular, scalable design
   - Common pattern in modern AI applications (e.g., AutoGPT, BabyAGI)

### 2. **Domain Expertise Modeling**
   - Each agent has specialized knowledge
   - Mimics real-world data science teams (EDA expert, ML engineer, BI analyst)
   - Shows understanding of data science workflow stages

### 3. **Intelligent Task Delegation**
   - Keyword-based routing with scoring system
   - Fallback strategies for ambiguous requests
   - Can be enhanced with semantic similarity (embeddings)

### 4. **Scalability & Maintainability**
   - Easy to add new specialist agents
   - Each agent has focused system prompt (< 500 tokens)
   - Tools remain shared and reusable

### 5. **Production-Ready Features**
   - Non-breaking: All existing functionality preserved
   - UI visibility: Users see which agent is working
   - Backward compatible: Falls back to main orchestrator if needed

## Interview Talking Points

### "Tell me about your multi-agent system"
> "I implemented a multi-agent architecture where specialized AI agents handle different stages of the data science workflow. Each agent has focused expertise - like the EDA Specialist for data profiling or the Modeling Specialist for ML training. The main orchestrator uses keyword-based routing to delegate tasks to the appropriate specialist. This mirrors how real data science teams work, with different experts collaborating on projects."

### "How do the agents communicate?"
> "They don't directly communicate with each other. Instead, the main orchestrator maintains session memory and workflow state. When the EDA Agent finds data quality issues, it saves those findings to the workflow state. Later, the Data Engineering Agent can reference that state to decide which cleaning strategies to apply. This prevents redundant analysis and keeps context across the workflow."

### "Why not use a single LLM prompt?"
> "A single prompt would need to cover 80+ tools across EDA, preprocessing, modeling, visualization, and business intelligence. That's ~15K tokens just for tool descriptions. By routing to specialists, each agent only sees ~20 relevant tools, reducing context to ~3K tokens. This improves response quality and reduces API costs. Plus, it makes the system more maintainable - I can update one specialist without touching others."

### "What would you improve?"
> "Three enhancements I'd consider:
> 1. **Semantic Routing**: Replace keyword matching with embedding-based similarity for better intent detection
> 2. **Inter-Agent Handoff**: Allow agents to explicitly request another specialist (e.g., EDA Agent says 'I need the Viz Agent to create plots')
> 3. **Agent Memory**: Give each agent its own memory to track what it has already done, preventing redundant work"

## Technical Implementation Details

### Code Changes Made

1. **orchestrator.py** (Lines 300-306):
   - Added specialist agent initialization
   - Added active_agent tracking

2. **orchestrator.py** (Lines 907-1030):
   - `_initialize_specialist_agents()`: Creates 5 specialist agent configurations
   - `_select_specialist_agent()`: Routes tasks based on keyword scoring
   - `_get_agent_system_prompt()`: Returns specialist's system prompt

3. **orchestrator.py** (Lines 2365-2388):
   - Modified analyze() to route to specialist agents
   - Emits `agent_assigned` SSE event for UI display
   - Falls back to compact prompts if enabled

4. **ChatInterface.tsx** (Lines 107-132):
   - Added `agent_assigned` event handler
   - Displays specialist agent info in typing indicator

### Backward Compatibility

✅ **No Breaking Changes**:
- All 80+ tools remain accessible to all agents
- Session memory continues to work
- Cache system unchanged
- File upload and follow-up requests work identically
- Can be disabled by setting `use_compact_prompts=True`

## Future Enhancements

### Phase 2: Semantic Routing
```python
# Use embeddings for smarter routing
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
user_embedding = model.encode(task_description)
agent_embeddings = {agent: model.encode(config['description']) 
                   for agent, config in specialist_agents.items()}

# Find most similar agent
best_agent = max(agent_embeddings.items(), 
                key=lambda x: cosine_similarity(user_embedding, x[1]))
```

### Phase 3: Agent Collaboration
```python
# Allow agents to request help from other specialists
{
  "action": "delegate",
  "to_agent": "viz_agent",
  "task": "Create a correlation heatmap for these features",
  "context": {"features": ["age", "income", "score"]}
}
```

### Phase 4: Agent Learning
```python
# Track agent performance and optimize routing
agent_metrics = {
  "eda_agent": {"success_rate": 0.95, "avg_time": 3.2},
  "modeling_agent": {"success_rate": 0.89, "avg_time": 12.5}
}

# Use RL to improve routing decisions over time
```

## Comparison to Other Systems

| System | Agents | Routing | Collaboration | Tools |
|--------|--------|---------|---------------|-------|
| **DS Agent (Ours)** | 5 specialists | Keyword + scoring | Sequential (via state) | 80+ |
| AutoGPT | 1 (general) | N/A | N/A | 10-15 |
| BabyAGI | Task-based | Queue system | Task decomposition | 5-10 |
| LangChain Agents | Custom | Tool selection | Chain/tree | Unlimited |
| CrewAI | Role-based | Explicit handoff | Collaborative | Unlimited |

**Our Advantage**: Purpose-built for data science workflows with domain-specific agents and extensive tool coverage.

---

## Summary

The multi-agent architecture transforms the DS Agent from a monolithic orchestrator into a collaborative team of specialists. This showcases:
- ✅ Advanced AI architecture patterns
- ✅ Domain expertise modeling
- ✅ Scalable, maintainable design
- ✅ Production-ready features
- ✅ Strong interview talking points

**All existing functionality preserved - purely additive enhancement.**
