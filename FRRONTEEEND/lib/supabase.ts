import { createClient } from '@supabase/supabase-js';

// Supabase configuration
// For HuggingFace Spaces: secrets are injected at runtime via window.__SUPABASE_CONFIG__
// For local dev: use import.meta.env (Vite build-time variables)
declare global {
  interface Window {
    __SUPABASE_CONFIG__?: {
      url: string;
      anonKey: string;
    };
  }
}

// Try to get config from runtime injection first (HuggingFace), then fall back to Vite env vars
const getSupabaseConfig = () => {
  // Check for runtime config (injected by server)
  if (typeof window !== 'undefined' && window.__SUPABASE_CONFIG__) {
    return {
      url: window.__SUPABASE_CONFIG__.url,
      anonKey: window.__SUPABASE_CONFIG__.anonKey
    };
  }
  
  // Fall back to Vite build-time env vars
  const url = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_SUPABASE_URL) || '';
  const anonKey = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_SUPABASE_ANON_KEY) || '';
  
  return { url, anonKey };
};

const config = getSupabaseConfig();
const supabaseUrl = config.url;
const supabaseAnonKey = config.anonKey;

// Check if Supabase is configured
export const isSupabaseConfigured = () => {
  const cfg = getSupabaseConfig();
  return !!(cfg.url && cfg.anonKey && cfg.url.includes('supabase') && !cfg.url.includes('placeholder'));
};

// Create Supabase client (use placeholder if not configured to avoid errors)
export const supabase = createClient(
  supabaseUrl || 'https://placeholder.supabase.co', 
  supabaseAnonKey || 'placeholder-key'
);

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

// User profile management
export interface UserProfile {
  id?: string;
  user_id: string;
  name: string;
  email: string;
  primary_goal?: string;
  target_outcome?: string;
  data_types?: string[];
  profession?: string;
  experience?: string;
  industry?: string;
  onboarding_completed: boolean;
  created_at?: string;
  updated_at?: string;
}

// Create or update user profile (for signup form data)
export const saveUserProfile = async (profile: Omit<UserProfile, 'id' | 'created_at' | 'updated_at'>) => {
  try {
    const { data, error } = await supabase
      .from('user_profiles')
      .upsert([{
        ...profile,
        updated_at: new Date().toISOString()
      }], {
        onConflict: 'user_id'
      })
      .select()
      .single();
    
    if (error) {
      console.error('Failed to save user profile:', error);
      return null;
    }
    return data;
  } catch (err) {
    console.error('Profile save error:', err);
    return null;
  }
};

// Check if user has completed onboarding
export const getUserProfile = async (userId: string) => {
  try {
    const { data, error } = await supabase
      .from('user_profiles')
      .select('*')
      .eq('user_id', userId)
      .single();
    
    if (error) {
      // User not found is not an error (first time user)
      if (error.code === 'PGRST116') {
        return null;
      }
      console.error('Failed to get user profile:', error);
      return null;
    }
    return data as UserProfile;
  } catch (err) {
    console.error('Profile fetch error:', err);
    return null;
  }
};

