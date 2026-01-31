# Supabase Database Setup

This document explains how to set up the Supabase tables for user authentication and analytics.

## Tables Created

### 1. user_profiles
Stores user onboarding data from the multi-step signup form.

**Columns:**
- `id` (UUID) - Primary key
- `user_id` (UUID) - References auth.users(id), unique
- `name` (TEXT) - Full name
- `email` (TEXT) - Email address
- `primary_goal` (TEXT) - User's primary goal
- `target_outcome` (TEXT) - What they want to achieve
- `data_types` (TEXT[]) - Array of data types they work with
- `profession` (TEXT) - Job title/role
- `experience` (TEXT) - Experience level
- `industry` (TEXT) - Industry they work in
- `onboarding_completed` (BOOLEAN) - Whether they finished onboarding
- `created_at` (TIMESTAMP) - Created timestamp
- `updated_at` (TIMESTAMP) - Last updated timestamp

### 2. usage_analytics
Tracks each query/interaction with the agent.

**Columns:**
- `id` (UUID) - Primary key
- `user_id` (TEXT) - User identifier
- `user_email` (TEXT) - User email
- `session_id` (TEXT) - Session UUID
- `query` (TEXT) - The user's query
- `agent_used` (TEXT) - Which specialist agent handled it
- `tools_executed` (TEXT[]) - Array of tools used
- `tokens_used` (INTEGER) - Tokens consumed
- `duration_ms` (INTEGER) - Time taken
- `success` (BOOLEAN) - Whether it succeeded
- `error_message` (TEXT) - Error details if failed
- `created_at` (TIMESTAMP) - When the query was made

### 3. user_sessions
Tracks user sessions for analytics.

**Columns:**
- `id` (UUID) - Primary key
- `user_id` (TEXT) - User identifier
- `user_email` (TEXT) - User email
- `started_at` (TIMESTAMP) - Session start time
- `ended_at` (TIMESTAMP) - Session end time
- `queries_count` (INTEGER) - Number of queries in session
- `browser_info` (TEXT) - User agent string
- `created_at` (TIMESTAMP) - Created timestamp

## Setup Instructions

### Step 1: Create Tables

1. Go to your Supabase project: https://supabase.com/dashboard
2. Click on **SQL Editor** in the left sidebar
3. Click **New Query**
4. Copy the entire contents of `supabase_schema.sql`
5. Paste into the query editor
6. Click **Run** to execute

### Step 2: Verify Tables

1. Click on **Table Editor** in the left sidebar
2. You should see:
   - `user_profiles`
   - `usage_analytics`
   - `user_sessions`

### Step 3: Test Authentication

1. Sign up via the app (email/password or OAuth)
2. Complete the multi-step onboarding form
3. Check the `user_profiles` table - you should see your data
4. Make a query to the agent
5. Check `usage_analytics` and `user_sessions` tables for tracking data

## Row Level Security (RLS)

The tables have RLS enabled with policies:
- **user_profiles**: Users can only read/insert/update their own profile
- **usage_analytics** and **user_sessions**: No RLS (for admin analytics)

## Querying Analytics

### Get total users
```sql
SELECT COUNT(DISTINCT user_id) FROM user_profiles;
```

### Get queries in last 7 days
```sql
SELECT COUNT(*) 
FROM usage_analytics 
WHERE created_at >= NOW() - INTERVAL '7 days';
```

### Get most active users
```sql
SELECT user_email, COUNT(*) as query_count
FROM usage_analytics
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY user_email
ORDER BY query_count DESC
LIMIT 10;
```

### Get success rate
```sql
SELECT 
  COUNT(*) as total_queries,
  SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
  ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate
FROM usage_analytics;
```

## Troubleshooting

### User profile not saving
- Check that Supabase URL and Anon Key are set as HuggingFace Secrets
- Check browser console for errors
- Verify the `user_profiles` table exists and RLS policies are correct

### OAuth users not showing onboarding form
- This is expected! OAuth users are prompted to complete the form after signing in
- The form auto-fills their email and name from OAuth data
- They only fill in: goals, profession, industry, experience

### Analytics not tracking
- Check that `usage_analytics` and `user_sessions` tables exist
- Verify no RLS blocking inserts (they should have no RLS for admin use)
- Check network tab for failed API calls to Supabase
