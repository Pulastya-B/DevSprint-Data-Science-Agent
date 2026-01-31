# Supabase Authentication & Analytics Setup

## Quick Setup (5 minutes)

### 1. Create Supabase Project
1. Go to [https://app.supabase.com/](https://app.supabase.com/)
2. Click "New Project"
3. Choose organization, name your project (e.g., "ds-agent-analytics")
4. Set a strong database password (save it!)
5. Choose a region close to your users
6. Click "Create new project" and wait ~2 minutes

### 2. Get Your API Keys
1. Go to **Settings** → **API**
2. Copy:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon/public key**: `eyJhbGciOi...` (long string)

### 3. Configure Environment
Create `.env` file in `FRRONTEEEND/`:
```bash
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
```

For HuggingFace Spaces, add these as **Secrets**:
1. Go to your Space → Settings → Repository secrets
2. Add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`

### 4. Set Up Database Tables
1. Go to **SQL Editor** in Supabase dashboard
2. Copy contents of `supabase_schema.sql`
3. Click "Run" to create tables and policies

### 5. Enable Authentication Providers (Optional)

#### Email (enabled by default)
- Works out of the box

#### Google OAuth
1. Go to **Authentication** → **Providers** → **Google**
2. Enable it
3. Create OAuth credentials at [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
4. Add your Supabase callback URL: `https://your-project.supabase.co/auth/v1/callback`
5. Copy Client ID and Secret to Supabase

#### GitHub OAuth
1. Go to **Authentication** → **Providers** → **GitHub**
2. Enable it
3. Create OAuth App at [GitHub Developer Settings](https://github.com/settings/developers)
4. Add callback URL: `https://your-project.supabase.co/auth/v1/callback`
5. Copy Client ID and Secret to Supabase

## Features Included

### Authentication
- ✅ Email/Password sign up & sign in
- ✅ Google OAuth (optional)
- ✅ GitHub OAuth (optional)
- ✅ Persistent sessions
- ✅ Guest mode (can use without signing in)

### Analytics Tracking
- ✅ Per-query tracking (user, session, query text, success/failure)
- ✅ Session tracking (start time, end time, query count)
- ✅ Browser info capture
- ✅ Anonymous user support

### Dashboard Views (in Supabase)
- `daily_active_users` - DAU metrics
- `popular_queries` - Most common queries
- `agent_usage_stats` - Which agents are used most

## Viewing Analytics

### Quick Stats in Supabase
1. Go to **Table Editor**
2. Select `usage_analytics` or `user_sessions`
3. Use filters and sorting to analyze data

### SQL Queries
```sql
-- Total users last 7 days
SELECT COUNT(DISTINCT user_id) 
FROM usage_analytics 
WHERE created_at > NOW() - INTERVAL '7 days';

-- Queries per day
SELECT DATE(created_at), COUNT(*) 
FROM usage_analytics 
GROUP BY DATE(created_at) 
ORDER BY 1 DESC;

-- Success rate
SELECT 
  SUM(CASE WHEN success THEN 1 ELSE 0 END)::float / COUNT(*) * 100 as success_rate
FROM usage_analytics;
```

## Free Tier Limits (Plenty for demos!)
- 50,000 monthly active users
- 500 MB database storage
- 1 GB file storage
- Unlimited API requests
- 2 projects

## Troubleshooting

### "Invalid API key"
- Check that your `.env` file has the correct values
- Make sure you're using the `anon` key, not the `service_role` key

### OAuth not working
- Verify callback URL is correct in provider settings
- Check that the provider is enabled in Supabase

### Data not appearing
- Check browser console for errors
- Verify RLS policies are created (run the SQL schema)
- Check Supabase logs for errors
