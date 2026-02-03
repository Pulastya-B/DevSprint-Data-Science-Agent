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
  huggingface_token?: string;  // Encrypted HF token for storage integration
  huggingface_username?: string;
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

// Update HuggingFace token for a user (only updates existing profiles)
export const updateHuggingFaceToken = async (userId: string, hfToken: string, hfUsername?: string) => {
  console.log('[HF Token] Starting update for user:', userId);
  
  // Check if Supabase is properly configured
  if (!isSupabaseConfigured()) {
    console.error('[HF Token] Supabase not configured!');
    return null;
  }
  
  try {
    // Check if user is authenticated in Supabase
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    console.log('[HF Token] Current session:', session ? `User: ${session.user.id}` : 'NO SESSION');
    
    if (!session) {
      console.error('[HF Token] No active Supabase session! RLS will block the query.');
      console.error('[HF Token] User needs to be logged in via Supabase auth.');
      return null;
    }
    
    if (session.user.id !== userId) {
      console.warn('[HF Token] Session user ID mismatch:', session.user.id, '!=', userId);
    }
    
    console.log('[HF Token] Attempting update with authenticated session...');
    
    const updateData = { 
      huggingface_token: hfToken || null,
      huggingface_username: hfUsername || null,
      updated_at: new Date().toISOString()
    };
    console.log('[HF Token] Update payload:', { ...updateData, huggingface_token: hfToken ? '****' : null });
    
    const { error, count } = await supabase
      .from('user_profiles')
      .update(updateData)
      .eq('user_id', userId);
    
    if (error) {
      console.error('[HF Token] Update failed:', error.message, error.code, error.hint);
      return null;
    }
    
    console.log('[HF Token] Update successful!');
    return { ...updateData, user_id: userId };
  } catch (err: any) {
    console.error('[HF Token] Unexpected error:', err?.message || err);
    return null;
  }
};

// Get HuggingFace token for a user (returns masked token for security)
export const getHuggingFaceStatus = async (userId: string) => {
  try {
    const { data, error } = await supabase
      .from('user_profiles')
      .select('huggingface_token, huggingface_username')
      .eq('user_id', userId)
      .single();
    
    if (error) {
      return { connected: false };
    }
    
    return {
      connected: !!data?.huggingface_token,
      username: data?.huggingface_username,
      tokenMasked: data?.huggingface_token ? `hf_****${data.huggingface_token.slice(-4)}` : null
    };
  } catch (err) {
    console.error('HF status fetch error:', err);
    return { connected: false };
  }
};

