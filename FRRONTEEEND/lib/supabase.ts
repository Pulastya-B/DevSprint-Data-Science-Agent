import { createClient } from '@supabase/supabase-js';

// Supabase configuration - these will be loaded from environment variables
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

// Create Supabase client
export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Types for our analytics
export interface UsageAnalytics {
  id?: string;
  user_id: string;
  user_email?: string;
  session_id: string;
  query: string;
  agent_used?: string;
  tools_executed?: string[];
  tokens_used?: number;
  duration_ms?: number;
  success: boolean;
  error_message?: string;
  created_at?: string;
}

export interface UserSession {
  id?: string;
  user_id: string;
  user_email?: string;
  started_at: string;
  ended_at?: string;
  queries_count: number;
  browser_info?: string;
}

// Analytics functions
export const trackQuery = async (analytics: Omit<UsageAnalytics, 'id' | 'created_at'>) => {
  try {
    const { data, error } = await supabase
      .from('usage_analytics')
      .insert([{
        ...analytics,
        created_at: new Date().toISOString()
      }]);
    
    if (error) {
      console.error('Failed to track query:', error);
      return null;
    }
    return data;
  } catch (err) {
    console.error('Analytics tracking error:', err);
    return null;
  }
};

export const startUserSession = async (userId: string, userEmail?: string) => {
  try {
    const { data, error } = await supabase
      .from('user_sessions')
      .insert([{
        user_id: userId,
        user_email: userEmail,
        started_at: new Date().toISOString(),
        queries_count: 0,
        browser_info: typeof navigator !== 'undefined' ? navigator.userAgent : null
      }])
      .select()
      .single();
    
    if (error) {
      console.error('Failed to start session:', error);
      return null;
    }
    return data;
  } catch (err) {
    console.error('Session tracking error:', err);
    return null;
  }
};

export const endUserSession = async (sessionId: string) => {
  try {
    const { error } = await supabase
      .from('user_sessions')
      .update({ ended_at: new Date().toISOString() })
      .eq('id', sessionId);
    
    if (error) {
      console.error('Failed to end session:', error);
    }
  } catch (err) {
    console.error('Session end error:', err);
  }
};

export const incrementSessionQueries = async (sessionId: string) => {
  try {
    // Use RPC for atomic increment
    const { error } = await supabase.rpc('increment_session_queries', {
      session_id: sessionId
    });
    
    if (error) {
      // Fallback: fetch and update
      const { data } = await supabase
        .from('user_sessions')
        .select('queries_count')
        .eq('id', sessionId)
        .single();
      
      if (data) {
        await supabase
          .from('user_sessions')
          .update({ queries_count: (data.queries_count || 0) + 1 })
          .eq('id', sessionId);
      }
    }
  } catch (err) {
    console.error('Failed to increment queries:', err);
  }
};

// Get usage stats (for admin dashboard)
export const getUsageStats = async (days: number = 7) => {
  try {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    const { data, error } = await supabase
      .from('usage_analytics')
      .select('*')
      .gte('created_at', startDate.toISOString())
      .order('created_at', { ascending: false });
    
    if (error) {
      console.error('Failed to get stats:', error);
      return null;
    }
    return data;
  } catch (err) {
    console.error('Stats fetch error:', err);
    return null;
  }
};

// Get unique users count
export const getUniqueUsersCount = async (days: number = 7) => {
  try {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    const { data, error } = await supabase
      .from('user_sessions')
      .select('user_id')
      .gte('started_at', startDate.toISOString());
    
    if (error) {
      console.error('Failed to get unique users:', error);
      return 0;
    }
    
    // Count unique user IDs
    const uniqueUsers = new Set(data?.map(d => d.user_id));
    return uniqueUsers.size;
  } catch (err) {
    console.error('Unique users fetch error:', err);
    return 0;
  }
};
