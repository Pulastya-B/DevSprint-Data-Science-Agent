# BigQuery Output Schemas for Looker Compatibility

**Purpose**: Define stable BigQuery table schemas that BI tools (Looker, Data Studio) can query reliably.

**Design Principles**:
- ✅ **Stable Schema**: No breaking changes without versioning
- ✅ **Consistent Naming**: snake_case columns, clear dimension/metric separation
- ✅ **BI-Friendly Types**: Standard SQL types, no complex nested structures
- ✅ **Documented Grain**: Clear primary keys and update patterns
- ✅ **Dashboard-Ready**: Metrics aligned with common visualizations

---

## 📊 Table 1: `model_metrics`

**Description**: Model performance metrics tracked over time for monitoring and comparison.

**Use Cases**:
- Performance dashboards
- Model comparison reports
- Drift detection alerts
- A/B test analysis

**Update Frequency**: On every model training run

**Grain**: One row per model training execution

### Schema

| Column Name | Type | Description | Dimension/Metric | Example |
|------------|------|-------------|------------------|---------|
| `project_id` | STRING | Google Cloud project ID | Dimension | `my-ml-project` |
| `dataset_id` | STRING | BigQuery dataset name | Dimension | `ml_models` |
| `model_id` | STRING | Unique model identifier | Dimension (Primary Key) | `xgboost_churn_20251223_153045` |
| `model_name` | STRING | Human-readable model name | Dimension | `Customer Churn Predictor` |
| `model_type` | STRING | Algorithm used | Dimension | `XGBoost`, `RandomForest`, `LightGBM` |
| `task_type` | STRING | ML task category | Dimension | `classification`, `regression` |
| `training_dataset` | STRING | Source table/file reference | Dimension | `project.dataset.train_data` |
| `target_column` | STRING | Prediction target name | Dimension | `churn`, `price`, `survived` |
| `created_at` | TIMESTAMP | Model training timestamp | Dimension (Time) | `2025-12-23 15:30:45 UTC` |
| `created_date` | DATE | Training date (for partitioning) | Dimension (Time) | `2025-12-23` |
| `feature_count` | INTEGER | Number of features used | Metric | `42` |
| `training_rows` | INTEGER | Training set size | Metric | `10000` |
| `test_rows` | INTEGER | Test set size | Metric | `2500` |
| `training_duration_seconds` | FLOAT | Time to train model | Metric | `123.45` |
| `accuracy` | FLOAT | Overall accuracy (0-1) | Metric | `0.95` |
| `precision` | FLOAT | Precision score (0-1) | Metric | `0.92` |
| `recall` | FLOAT | Recall score (0-1) | Metric | `0.88` |
| `f1_score` | FLOAT | F1 score (0-1) | Metric | `0.90` |
| `roc_auc` | FLOAT | ROC AUC score (0-1) | Metric | `0.94` |
| `pr_auc` | FLOAT | Precision-Recall AUC (0-1) | Metric | `0.91` |
| `mae` | FLOAT | Mean Absolute Error (regression) | Metric | `1234.56` |
| `mse` | FLOAT | Mean Squared Error (regression) | Metric | `567890.12` |
| `rmse` | FLOAT | Root Mean Squared Error (regression) | Metric | `753.59` |
| `r2_score` | FLOAT | R² coefficient (regression) | Metric | `0.85` |
| `cross_val_mean` | FLOAT | Mean CV score | Metric | `0.93` |
| `cross_val_std` | FLOAT | CV score std deviation | Metric | `0.02` |
| `hyperparameters` | STRING (JSON) | Model hyperparameters | Metadata | `{"max_depth": 6, "n_estimators": 100}` |
| `version` | STRING | Model version tag | Dimension | `v1.2.3` |
| `environment` | STRING | Training environment | Dimension | `production`, `staging`, `development` |
| `user_email` | STRING | User who trained model | Dimension | `data-scientist@company.com` |

### Partitioning & Clustering

```sql
-- Recommended table setup
CREATE TABLE `project.dataset.model_metrics`
(
  -- columns as above
)
PARTITION BY created_date
CLUSTER BY model_type, task_type, environment
OPTIONS(
  description="Model performance metrics for BI dashboards",
  require_partition_filter=true
);
```

### Primary Dimensions for Looker

- **Time**: `created_at`, `created_date`
- **Model**: `model_type`, `model_name`, `task_type`
- **Performance Tier**: CASE expression on `accuracy`/`f1_score`
  - `Excellent` (>0.90)
  - `Good` (0.80-0.90)
  - `Fair` (0.70-0.80)
  - `Poor` (<0.70)

### Sample Looker View

```lookml
view: model_metrics {
  sql_table_name: `project.dataset.model_metrics` ;;

  dimension: model_id {
    primary_key: yes
    type: string
    sql: ${TABLE}.model_id ;;
  }

  dimension_group: created {
    type: time
    timeframes: [date, week, month, quarter, year]
    sql: ${TABLE}.created_at ;;
  }

  dimension: model_type {
    type: string
    sql: ${TABLE}.model_type ;;
  }

  dimension: performance_tier {
    type: string
    sql: CASE
      WHEN ${TABLE}.accuracy >= 0.90 THEN 'Excellent'
      WHEN ${TABLE}.accuracy >= 0.80 THEN 'Good'
      WHEN ${TABLE}.accuracy >= 0.70 THEN 'Fair'
      ELSE 'Poor'
    END ;;
  }

  measure: count {
    type: count
  }

  measure: avg_accuracy {
    type: average
    sql: ${TABLE}.accuracy ;;
    value_format_name: percent_2
  }

  measure: avg_f1_score {
    type: average
    sql: ${TABLE}.f1_score ;;
    value_format_name: percent_2
  }
}
```

---

## 🎯 Table 2: `feature_importance`

**Description**: Feature importance scores for model interpretability.

**Use Cases**:
- Feature impact analysis
- Feature selection dashboards
- Model explainability reports

**Update Frequency**: On every model training run

**Grain**: One row per feature per model

### Schema

| Column Name | Type | Description | Dimension/Metric | Example |
|------------|------|-------------|------------------|---------|
| `model_id` | STRING | Foreign key to model_metrics | Dimension (Foreign Key) | `xgboost_churn_20251223_153045` |
| `feature_name` | STRING | Name of the feature | Dimension (Primary Key) | `age`, `total_purchases`, `days_since_last_login` |
| `importance_score` | FLOAT | Importance value (0-1) | Metric | `0.35` |
| `importance_rank` | INTEGER | Rank by importance (1=most important) | Metric | `1`, `2`, `3` |
| `importance_type` | STRING | Calculation method | Dimension | `gain`, `weight`, `cover`, `shap` |
| `feature_type` | STRING | Data type category | Dimension | `numeric`, `categorical`, `datetime`, `text` |
| `is_engineered` | BOOLEAN | Created by feature engineering? | Dimension | `true`, `false` |
| `created_at` | TIMESTAMP | When importance was calculated | Dimension (Time) | `2025-12-23 15:30:45 UTC` |
| `created_date` | DATE | Calculation date | Dimension (Time) | `2025-12-23` |

### Partitioning & Clustering

```sql
CREATE TABLE `project.dataset.feature_importance`
(
  -- columns as above
)
PARTITION BY created_date
CLUSTER BY model_id, importance_rank
OPTIONS(
  description="Feature importance scores for model explainability",
  require_partition_filter=false  -- Allow cross-model queries
);
```

### Primary Dimensions for Looker

- **Feature**: `feature_name`, `feature_type`, `is_engineered`
- **Model**: `model_id` (join to model_metrics)
- **Importance**: `importance_rank`, `importance_type`

### Sample Looker View

```lookml
view: feature_importance {
  sql_table_name: `project.dataset.feature_importance` ;;

  dimension: compound_key {
    primary_key: yes
    hidden: yes
    sql: CONCAT(${TABLE}.model_id, '|', ${TABLE}.feature_name) ;;
  }

  dimension: feature_name {
    type: string
    sql: ${TABLE}.feature_name ;;
  }

  dimension: is_top_10 {
    type: yesno
    sql: ${TABLE}.importance_rank <= 10 ;;
  }

  measure: avg_importance {
    type: average
    sql: ${TABLE}.importance_score ;;
    value_format_name: percent_2
  }

  measure: count_features {
    type: count_distinct
    sql: ${TABLE}.feature_name ;;
  }
}
```

---

## 🔮 Table 3: `predictions`

**Description**: Model predictions with actuals for monitoring and evaluation.

**Use Cases**:
- Prediction monitoring
- Accuracy tracking over time
- Segment performance analysis
- Business impact measurement

**Update Frequency**: Real-time or batch (daily/hourly)

**Grain**: One row per prediction

### Schema

| Column Name | Type | Description | Dimension/Metric | Example |
|------------|------|-------------|------------------|---------|
| `prediction_id` | STRING | Unique prediction identifier | Dimension (Primary Key) | `pred_abc123xyz` |
| `model_id` | STRING | Model used for prediction | Dimension (Foreign Key) | `xgboost_churn_20251223_153045` |
| `entity_id` | STRING | Entity being predicted (customer_id, product_id, etc.) | Dimension | `customer_12345` |
| `predicted_at` | TIMESTAMP | When prediction was made | Dimension (Time) | `2025-12-23 15:30:45 UTC` |
| `predicted_date` | DATE | Prediction date (for partitioning) | Dimension (Time) | `2025-12-23` |
| `prediction_value` | FLOAT | Predicted value | Metric | `0.85` (probability), `49.99` (price) |
| `prediction_class` | STRING | Predicted class (classification) | Dimension | `churn`, `not_churn` |
| `prediction_confidence` | FLOAT | Model confidence (0-1) | Metric | `0.92` |
| `actual_value` | FLOAT | True value (when available) | Metric | `1.0` (churned), `52.50` (actual price) |
| `actual_class` | STRING | True class (when available) | Dimension | `churn`, `not_churn` |
| `actual_recorded_at` | TIMESTAMP | When actual became known | Dimension (Time) | `2025-12-30 10:00:00 UTC` |
| `is_correct` | BOOLEAN | Prediction was correct? | Dimension | `true`, `false` |
| `absolute_error` | FLOAT | \|predicted - actual\| | Metric | `2.51` |
| `squared_error` | FLOAT | (predicted - actual)² | Metric | `6.30` |
| `feature_values` | STRING (JSON) | Input features used | Metadata | `{"age": 35, "tenure": 24}` |
| `segment` | STRING | Business segment | Dimension | `enterprise`, `smb`, `consumer` |
| `region` | STRING | Geographic region | Dimension | `us-west`, `eu-central` |
| `model_version` | STRING | Model version | Dimension | `v1.2.3` |
| `prediction_latency_ms` | FLOAT | Inference time | Metric | `23.4` |

### Partitioning & Clustering

```sql
CREATE TABLE `project.dataset.predictions`
(
  -- columns as above
)
PARTITION BY predicted_date
CLUSTER BY model_id, segment, is_correct
OPTIONS(
  description="Model predictions with actuals for monitoring",
  require_partition_filter=true,
  partition_expiration_days=730  -- 2 years retention
);
```

### Primary Dimensions for Looker

- **Time**: `predicted_date`, days since prediction
- **Model**: `model_id`, `model_version`
- **Segment**: `segment`, `region`
- **Accuracy**: `is_correct`, error buckets

### Sample Looker View

```lookml
view: predictions {
  sql_table_name: `project.dataset.predictions` ;;

  dimension: prediction_id {
    primary_key: yes
    type: string
    sql: ${TABLE}.prediction_id ;;
  }

  dimension_group: predicted {
    type: time
    timeframes: [date, week, month]
    sql: ${TABLE}.predicted_at ;;
  }

  dimension: segment {
    type: string
    sql: ${TABLE}.segment ;;
  }

  dimension: error_bucket {
    type: string
    sql: CASE
      WHEN ${TABLE}.absolute_error IS NULL THEN 'No Actual Yet'
      WHEN ${TABLE}.absolute_error <= 0.1 THEN '0-10%'
      WHEN ${TABLE}.absolute_error <= 0.2 THEN '10-20%'
      ELSE '>20%'
    END ;;
  }

  measure: count {
    type: count
  }

  measure: accuracy_rate {
    type: average
    sql: CAST(${TABLE}.is_correct AS FLOAT64) ;;
    value_format_name: percent_1
  }

  measure: avg_confidence {
    type: average
    sql: ${TABLE}.prediction_confidence ;;
    value_format_name: percent_2
  }

  measure: mae {
    type: average
    sql: ${TABLE}.absolute_error ;;
    value_format_name: decimal_2
  }
}
```

---

## 📋 Table 4: `data_profile_summary`

**Description**: Dataset profiling statistics for data quality monitoring.

**Use Cases**:
- Data quality dashboards
- Schema drift detection
- Data validation reports
- Column-level monitoring

**Update Frequency**: Daily or on-demand

**Grain**: One row per column per dataset per run

### Schema

| Column Name | Type | Description | Dimension/Metric | Example |
|------------|------|-------------|------------------|---------|
| `profile_id` | STRING | Unique profile run identifier | Dimension (Primary Key) | `profile_abc123xyz` |
| `dataset_name` | STRING | Source table/file name | Dimension | `project.dataset.customers` |
| `column_name` | STRING | Column being profiled | Dimension | `age`, `email`, `signup_date` |
| `profiled_at` | TIMESTAMP | When profiling ran | Dimension (Time) | `2025-12-23 15:30:45 UTC` |
| `profiled_date` | DATE | Profiling date | Dimension (Time) | `2025-12-23` |
| `data_type` | STRING | Column data type | Dimension | `INTEGER`, `STRING`, `FLOAT`, `TIMESTAMP` |
| `inferred_type` | STRING | Smart type inference | Dimension | `numeric`, `categorical`, `datetime`, `text`, `email` |
| `row_count` | INTEGER | Total rows in dataset | Metric | `10000` |
| `non_null_count` | INTEGER | Non-null values | Metric | `9850` |
| `null_count` | INTEGER | Null values | Metric | `150` |
| `null_percentage` | FLOAT | % null (0-100) | Metric | `1.5` |
| `unique_count` | INTEGER | Distinct values | Metric | `450` |
| `uniqueness_percentage` | FLOAT | % unique (0-100) | Metric | `4.5` |
| `min_value` | STRING | Minimum value (as string) | Metadata | `18`, `2020-01-01` |
| `max_value` | STRING | Maximum value (as string) | Metadata | `95`, `2025-12-23` |
| `mean_value` | FLOAT | Mean (numeric only) | Metric | `42.5` |
| `median_value` | FLOAT | Median (numeric only) | Metric | `38.0` |
| `std_dev` | FLOAT | Standard deviation (numeric only) | Metric | `15.2` |
| `skewness` | FLOAT | Distribution skewness | Metric | `0.85` |
| `kurtosis` | FLOAT | Distribution kurtosis | Metric | `2.1` |
| `top_value` | STRING | Most common value | Metadata | `male`, `active` |
| `top_value_frequency` | INTEGER | Count of most common value | Metric | `6500` |
| `top_value_percentage` | FLOAT | % of most common value | Metric | `65.0` |
| `has_outliers` | BOOLEAN | Outliers detected? | Dimension | `true`, `false` |
| `outlier_count` | INTEGER | Number of outliers | Metric | `23` |
| `outlier_percentage` | FLOAT | % outliers | Metric | `0.23` |
| `quality_score` | FLOAT | Overall quality score (0-100) | Metric | `92.5` |
| `quality_issues` | STRING (JSON) | Detected issues | Metadata | `["high_nulls", "duplicate_values"]` |
| `validation_status` | STRING | Quality check result | Dimension | `pass`, `warn`, `fail` |

### Partitioning & Clustering

```sql
CREATE TABLE `project.dataset.data_profile_summary`
(
  -- columns as above
)
PARTITION BY profiled_date
CLUSTER BY dataset_name, validation_status
OPTIONS(
  description="Dataset profiling for data quality monitoring",
  require_partition_filter=true,
  partition_expiration_days=90  -- 3 months retention
);
```

### Primary Dimensions for Looker

- **Dataset**: `dataset_name`
- **Column**: `column_name`, `data_type`, `inferred_type`
- **Quality**: `validation_status`, `quality_score` buckets
- **Time**: `profiled_date`

### Sample Looker View

```lookml
view: data_profile_summary {
  sql_table_name: `project.dataset.data_profile_summary` ;;

  dimension: compound_key {
    primary_key: yes
    hidden: yes
    sql: CONCAT(${TABLE}.profile_id, '|', ${TABLE}.column_name) ;;
  }

  dimension: column_name {
    type: string
    sql: ${TABLE}.column_name ;;
  }

  dimension: quality_tier {
    type: string
    sql: CASE
      WHEN ${TABLE}.quality_score >= 90 THEN 'Excellent'
      WHEN ${TABLE}.quality_score >= 75 THEN 'Good'
      WHEN ${TABLE}.quality_score >= 60 THEN 'Fair'
      ELSE 'Poor'
    END ;;
  }

  dimension: has_quality_issues {
    type: yesno
    sql: ${TABLE}.validation_status IN ('warn', 'fail') ;;
  }

  measure: count_columns {
    type: count_distinct
    sql: ${TABLE}.column_name ;;
  }

  measure: avg_quality_score {
    type: average
    sql: ${TABLE}.quality_score ;;
    value_format_name: decimal_1
  }

  measure: avg_null_percentage {
    type: average
    sql: ${TABLE}.null_percentage ;;
    value_format_name: percent_1
  }

  measure: columns_with_issues {
    type: count_distinct
    sql: ${TABLE}.column_name ;;
    filters: [has_quality_issues: "yes"]
  }
}
```

---

## 🔄 Schema Evolution Guidelines

### ✅ **SAFE Changes** (Non-Breaking)

1. **Add new columns** (always nullable or with defaults)
   ```sql
   ALTER TABLE `project.dataset.model_metrics`
   ADD COLUMN IF NOT EXISTS new_metric FLOAT64;
   ```

2. **Add new tables** (doesn't affect existing dashboards)

3. **Lengthen STRING columns** (VARCHAR(50) → VARCHAR(100))

4. **Add indexes/clustering** (performance only)

5. **Add column descriptions**
   ```sql
   ALTER TABLE `project.dataset.model_metrics`
   ALTER COLUMN accuracy SET OPTIONS (description='Model accuracy (0-1)');
   ```

### ❌ **BREAKING Changes** (Require Dashboard Updates)

1. **Rename columns** → Use views for backward compatibility:
   ```sql
   CREATE OR REPLACE VIEW `project.dataset.model_metrics_v2` AS
   SELECT
     model_id,
     accuracy AS acc,  -- renamed column
     ...
   FROM `project.dataset.model_metrics`;
   ```

2. **Change data types** → Create new column, migrate, deprecate old:
   ```sql
   -- Step 1: Add new column
   ALTER TABLE model_metrics ADD COLUMN created_at_new TIMESTAMP;
   
   -- Step 2: Backfill
   UPDATE model_metrics SET created_at_new = CAST(created_at AS TIMESTAMP) WHERE true;
   
   -- Step 3: Update dashboards to use new column
   
   -- Step 4: Drop old column after validation period
   ALTER TABLE model_metrics DROP COLUMN created_at;
   ```

3. **Remove columns** → Deprecate first, remove after 90 days

4. **Change partitioning** → Requires table recreation

### 🔄 **Versioning Strategy**

For major schema changes, create versioned tables:

```
project.dataset.model_metrics_v1  (deprecated, keep 90 days)
project.dataset.model_metrics_v2  (current)
project.dataset.model_metrics     (view pointing to latest version)
```

---

## 📊 Dashboard-Ready Metrics Catalog

### Model Performance Metrics

| Metric Name | Calculation | Use Case |
|------------|-------------|----------|
| **Model Count** | `COUNT(DISTINCT model_id)` | Total models trained |
| **Avg Accuracy** | `AVG(accuracy)` | Overall model quality |
| **Accuracy Trend** | `AVG(accuracy) OVER (ORDER BY created_date)` | Performance over time |
| **Best Model** | `model_id WHERE accuracy = MAX(accuracy)` | Top performer |
| **Models by Type** | `COUNT(*) GROUP BY model_type` | Algorithm distribution |
| **Training Time** | `AVG(training_duration_seconds)` | Resource usage |
| **Recent Models** | `WHERE created_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)` | Latest activity |

### Feature Importance Metrics

| Metric Name | Calculation | Use Case |
|------------|-------------|----------|
| **Top Features** | `WHERE importance_rank <= 10` | Most impactful features |
| **Avg Importance** | `AVG(importance_score)` | Feature impact distribution |
| **Engineered Features** | `COUNT(*) WHERE is_engineered = true` | Feature engineering effectiveness |
| **Feature Stability** | `STDDEV(importance_score) GROUP BY feature_name` | Consistent predictors |

### Prediction Metrics

| Metric Name | Calculation | Use Case |
|------------|-------------|----------|
| **Accuracy Rate** | `AVG(CAST(is_correct AS FLOAT64))` | Real-world performance |
| **MAE** | `AVG(absolute_error)` | Average error magnitude |
| **RMSE** | `SQRT(AVG(squared_error))` | Error with outlier penalty |
| **Predictions/Day** | `COUNT(*) GROUP BY predicted_date` | Volume tracking |
| **Confidence Distribution** | `APPROX_QUANTILES(prediction_confidence, 10)` | Model calibration |
| **Segment Performance** | `AVG(is_correct) GROUP BY segment` | Fairness check |

### Data Quality Metrics

| Metric Name | Calculation | Use Case |
|------------|-------------|----------|
| **Data Quality Score** | `AVG(quality_score)` | Overall health |
| **Null Rate** | `AVG(null_percentage)` | Completeness |
| **Columns with Issues** | `COUNT(DISTINCT column_name) WHERE validation_status != 'pass'` | Problem areas |
| **Quality Trend** | `AVG(quality_score) OVER (ORDER BY profiled_date)` | Improving/degrading? |

---

## 🎯 Sample Looker Explores

### Explore 1: Model Performance Analysis

```lookml
explore: model_metrics {
  label: "Model Performance"
  description: "Track model accuracy, training time, and comparison"

  join: feature_importance {
    type: left_outer
    sql_on: ${model_metrics.model_id} = ${feature_importance.model_id} ;;
    relationship: one_to_many
  }
}
```

### Explore 2: Prediction Monitoring

```lookml
explore: predictions {
  label: "Prediction Monitoring"
  description: "Real-time prediction accuracy and drift"

  join: model_metrics {
    type: left_outer
    sql_on: ${predictions.model_id} = ${model_metrics.model_id} ;;
    relationship: many_to_one
  }
}
```

### Explore 3: Data Quality Dashboard

```lookml
explore: data_profile_summary {
  label: "Data Quality"
  description: "Monitor data health and schema drift"
}
```

---

## 📝 Implementation Checklist

### Phase 1: Setup (Week 1)
- [ ] Create all 4 BigQuery tables with partitioning
- [ ] Set up service account permissions
- [ ] Configure table expiration policies
- [ ] Document table owners and update SLAs

### Phase 2: Integration (Week 2)
- [ ] Update tools to write to these schemas
- [ ] Add schema validation in CI/CD
- [ ] Create data dictionary in Looker
- [ ] Set up table monitoring alerts

### Phase 3: BI Layer (Week 3)
- [ ] Create Looker views for all 4 tables
- [ ] Build explores with joins
- [ ] Create initial dashboards
- [ ] Set up scheduled data refreshes

### Phase 4: Validation (Week 4)
- [ ] Backfill historical data
- [ ] Verify dashboard accuracy
- [ ] Train stakeholders on dashboards
- [ ] Document runbooks for common issues

---

## 🔗 Related Tools

**BigQuery Write Tools** (src/bigquery/):
- `bigquery_write_results()` - Generic write function
- Helper: `bigquery_write_model_metrics()` - Specialized writer
- Helper: `bigquery_write_feature_importance()` - Specialized writer
- Helper: `bigquery_write_predictions()` - Specialized writer
- Helper: `bigquery_write_data_profile()` - Specialized writer

**Example Usage**:
```python
from src.bigquery import bigquery_write_results

# Write model metrics
bigquery_write_results(
    data=metrics_df,
    table_id="project.dataset.model_metrics",
    write_disposition="WRITE_APPEND"
)
```

---

## 📚 Additional Resources

- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [Looker LookML Reference](https://cloud.google.com/looker/docs/reference/lookml-quick-reference)
- [Schema Design for BI](https://cloud.google.com/architecture/bigquery-data-warehouse)

---

**Last Updated**: December 23, 2025  
**Schema Version**: 1.0.0  
**Maintained By**: Data Science Team  
**Review Cadence**: Quarterly
