# Test Scenarios for Parameter Remapping Fixes

## Test Case 1: train_baseline_models with invalid 'models' parameter

### Input (from LLM):
```json
{
  "tool": "train_baseline_models",
  "arguments": {
    "file_path": "/tmp/data.csv",
    "target_column": "price",
    "models": ["linear_regression", "random_forest", "xgboost"],
    "test_size": 0.2,
    "random_state": 42
  }
}
```

### Expected Output (after remapping):
```
   ✓ Parameter remapped: target_column → target_col
   ✓ Stripped invalid parameter 'models': ['linear_regression', 'random_forest', 'xgboost']
   ℹ️ train_baseline_models trains all baseline models automatically
   📋 Final parameters: ['file_path', 'target_col', 'test_size', 'random_state']
🔧 Executing tool: train_baseline_models
✅ Tool executed successfully
```

### What Gets Called:
```python
train_baseline_models(
    file_path="/tmp/data.csv",
    target_col="price",  # Remapped from target_column
    test_size=0.2,
    random_state=42
    # models parameter stripped
)
```

---

## Test Case 2: generate_model_report with wrong parameter name

### Input (from LLM):
```json
{
  "tool": "generate_model_report",
  "arguments": {
    "model_path": "/tmp/model.pkl",
    "file_path": "/tmp/test.csv",
    "target_column": "price",
    "output_path": "/tmp/report.json"
  }
}
```

### Expected Output (after remapping):
```
   ✓ Parameter remapped: target_column → target_col
   ✓ Parameter remapped: file_path → test_data_path
   📋 Final parameters: ['model_path', 'test_data_path', 'target_col', 'output_path']
🔧 Executing tool: generate_model_report
✅ Tool executed successfully
```

### What Gets Called:
```python
generate_model_report(
    model_path="/tmp/model.pkl",
    test_data_path="/tmp/test.csv",  # Remapped from file_path
    target_col="price",  # Remapped from target_column
    output_path="/tmp/report.json"
)
```

---

## Test Case 3: detect_model_issues with invalid split parameters

### Input (from LLM):
```json
{
  "tool": "detect_model_issues",
  "arguments": {
    "model_path": "/tmp/model.pkl",
    "train_data_path": "/tmp/train.csv",
    "test_data_path": "/tmp/test.csv",
    "target_column": "price",
    "train_target_path": "/tmp/y_train.csv",
    "test_target_path": "/tmp/y_test.csv"
  }
}
```

### Expected Output (after remapping):
```
   ✓ Parameter remapped: target_column → target_col
   ✓ Stripped invalid parameter 'train_target_path': /tmp/y_train.csv
   ✓ Stripped invalid parameter 'test_target_path': /tmp/y_test.csv
   📋 Final parameters: ['model_path', 'train_data_path', 'test_data_path', 'target_col']
🔧 Executing tool: detect_model_issues
✅ Tool executed successfully
```

### What Gets Called:
```python
detect_model_issues(
    model_path="/tmp/model.pkl",
    train_data_path="/tmp/train.csv",
    test_data_path="/tmp/test.csv",
    target_col="price"  # Remapped from target_column
    # train_target_path and test_target_path stripped
)
```

---

## Test Case 4: detect_model_issues missing required parameter

### Input (from LLM):
```json
{
  "tool": "detect_model_issues",
  "arguments": {
    "model_path": "/tmp/model.pkl",
    "test_data_path": "/tmp/test.csv",
    "target_column": "price"
  }
}
```

### Expected Output (after remapping):
```
   ✓ Parameter remapped: target_column → target_col
   ⚠️ WARNING: detect_model_issues requires 'train_data_path' parameter
   📋 Final parameters: ['model_path', 'test_data_path', 'target_col']
🔧 Executing tool: detect_model_issues
❌ Error: detect_model_issues() missing 1 required positional argument: 'train_data_path'
```

### Result:
Tool will still fail (as expected) but with clear warning that train_data_path is required. LLM can retry with correct parameters.

---

## Test Case 5: Combined parameter issues

### Input (from LLM):
```json
{
  "tool": "train_baseline_models",
  "arguments": {
    "file_path": "/tmp/data.csv",
    "target_column": "price",
    "models": ["xgboost"],
    "test_size": "0.3",
    "random_state": "None"
  }
}
```

### Expected Output (after remapping):
```
   ✓ Parameter remapped: target_column → target_col
   ✓ Stripped invalid parameter 'models': ['xgboost']
   ℹ️ train_baseline_models trains all baseline models automatically
   📋 Final parameters: ['file_path', 'target_col', 'test_size', 'random_state']
🔧 Executing tool: train_baseline_models
✅ Tool executed successfully
```

### What Gets Called:
```python
train_baseline_models(
    file_path="/tmp/data.csv",
    target_col="price",  # Remapped
    test_size="0.3",  # String (may cause type error - should be float)
    random_state=None  # "None" string converted to None
)
```

**Note**: Type conversion from string "None" to None works. String "0.3" to float conversion needs testing.

---

## Test Case 6: No remapping needed (correct parameters)

### Input (from LLM):
```json
{
  "tool": "train_baseline_models",
  "arguments": {
    "file_path": "/tmp/data.csv",
    "target_col": "price",
    "test_size": 0.2,
    "random_state": 42
  }
}
```

### Expected Output:
```
   📋 Final parameters: ['file_path', 'target_col', 'test_size', 'random_state']
🔧 Executing tool: train_baseline_models
✅ Tool executed successfully
```

**No remapping messages** - parameters already correct!

---

## Validation Commands

### Check logs for parameter remapping:
```bash
grep "✓ Parameter remapped" logs.txt
grep "✓ Stripped invalid parameter" logs.txt
```

### Check for remaining errors:
```bash
grep "unexpected keyword argument" logs.txt
grep "missing.*required.*argument" logs.txt
```

### Count successful modeling tool executions:
```bash
grep -A5 "train_baseline_models" logs.txt | grep "✅ Tool executed successfully" | wc -l
grep -A5 "generate_model_report" logs.txt | grep "✅ Tool executed successfully" | wc -l
grep -A5 "detect_model_issues" logs.txt | grep "✅ Tool executed successfully" | wc -l
```

---

## Integration Test Flow

**Complete ML Pipeline Test**:

1. Load earthquake dataset
2. Profile data (`profile_dataset`)
3. Create time features (`create_time_features`)
4. Create interaction features (`create_interaction_features`)
5. Encode categorical (`encode_categorical`)
6. **Train baseline models** (`train_baseline_models` - WITH REMAPPING)
7. Hyperparameter tuning (`hyperparameter_tuning`)
8. Cross-validation (`perform_cross_validation`)
9. **Generate report** (`generate_model_report` - WITH REMAPPING)
10. **Detect issues** (`detect_model_issues` - WITH REMAPPING)

**Expected**: All steps succeed without parameter errors.

---

## Edge Cases to Consider

### 1. Both old and new parameter provided:
```json
{
  "target_column": "price",
  "target_col": "sales"
}
```
**Behavior**: Keep `target_col`, ignore `target_column` (remapping checks `target_col not in arguments`)

### 2. Parameter is None:
```json
{
  "models": null
}
```
**Behavior**: Still stripped (check is `if "models" in arguments`)

### 3. Empty list parameter:
```json
{
  "models": []
}
```
**Behavior**: Stripped with log showing empty list

### 4. Multiple invalid parameters:
```json
{
  "train_target_path": "/tmp/y_train.csv",
  "test_target_path": "/tmp/y_test.csv",
  "validation_target_path": "/tmp/y_val.csv"
}
```
**Behavior**: Only `train_target_path` and `test_target_path` stripped (not in remapping list)

---

## Success Metrics

After deployment, measure:
- ✅ Number of parameter remapping logs (should increase)
- ✅ Successful modeling tool executions (should increase)
- ✅ Parameter error count (should decrease to near zero)
- ✅ execute_python_code fallbacks for modeling (should decrease)
- ✅ Complete workflow success rate (should increase)
