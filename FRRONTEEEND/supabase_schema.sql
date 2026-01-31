-- =====================================================
-- Supabase Database Schema for Data Science Agent Analytics
-- =====================================================
-- Run this in your Supabase SQL Editor: https://app.supabase.com/project/_/sql

-- 1. Usage Analytics Table
-- Tracks individual queries/requests made by users
CREATE TABLE IF NOT EXISTS usage_analytics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    user_email TEXT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    agent_used TEXT,
    tools_executed TEXT[],
    tokens_used INTEGER,
    duration_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. User Sessions Table
-- Tracks user sessions for engagement metrics
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    user_email TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    queries_count INTEGER DEFAULT 0,
    browser_info TEXT
);

-- 3. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_usage_analytics_user_id ON usage_analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_created_at ON usage_analytics(created_at);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_session_id ON usage_analytics(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_started_at ON user_sessions(started_at);

-- 4. Function to increment session query count atomically
CREATE OR REPLACE FUNCTION increment_session_queries(session_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE user_sessions 
    SET queries_count = queries_count + 1 
    WHERE id = session_id;
END;
$$ LANGUAGE plpgsql;

-- 5. Enable Row Level Security (RLS)
ALTER TABLE usage_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- 6. RLS Policies - Allow authenticated users to insert their own data
-- Policy for usage_analytics
CREATE POLICY "Users can insert their own analytics" ON usage_analytics
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Users can view their own analytics" ON usage_analytics
    FOR SELECT USING (auth.uid()::text = user_id OR user_id = 'anonymous');

-- Policy for user_sessions
CREATE POLICY "Users can insert their own sessions" ON user_sessions
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Users can update their own sessions" ON user_sessions
    FOR UPDATE USING (auth.uid()::text = user_id OR user_id = 'anonymous');

CREATE POLICY "Users can view their own sessions" ON user_sessions
    FOR SELECT USING (auth.uid()::text = user_id OR user_id = 'anonymous');

-- 7. Helpful Views for Analytics Dashboard

-- Daily active users
CREATE OR REPLACE VIEW daily_active_users AS
SELECT 
    DATE(created_at) as date,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(*) as total_queries
FROM usage_analytics
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Popular queries
CREATE OR REPLACE VIEW popular_queries AS
SELECT 
    query,
    COUNT(*) as count,
    COUNT(DISTINCT user_id) as unique_users
FROM usage_analytics
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY query
ORDER BY count DESC
LIMIT 50;

-- Agent usage stats
CREATE OR REPLACE VIEW agent_usage_stats AS
SELECT 
    agent_used,
    COUNT(*) as total_uses,
    AVG(duration_ms) as avg_duration_ms,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::float / COUNT(*) * 100 as success_rate
FROM usage_analytics
WHERE agent_used IS NOT NULL
GROUP BY agent_used
ORDER BY total_uses DESC;

-- =====================================================
-- SETUP INSTRUCTIONS:
-- =====================================================
-- 1. Go to https://app.supabase.com/ and create a new project
-- 2. Go to Settings > API to get your Project URL and anon key
-- 3. Create a .env file in FRRONTEEEND/ with:
--    VITE_SUPABASE_URL=your_project_url
--    VITE_SUPABASE_ANON_KEY=your_anon_key
-- 4. Go to Authentication > Providers and enable:
--    - Email (enabled by default)
--    - Google (optional - need OAuth credentials)
--    - GitHub (optional - need OAuth app)
-- 5. Run this SQL in the SQL Editor
-- 6. Done! Your analytics will start tracking automatically
-- =====================================================
