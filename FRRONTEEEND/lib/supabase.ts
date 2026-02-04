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
export const getSupabaseConfig = () => {
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

// Helper function to add timeout to any promise
const withTimeout = <T>(promise: Promise<T>, ms: number, errorMsg: string): Promise<T> => {
  const timeout = new Promise<never>((_, reject) => 
    setTimeout(() => reject(new Error(errorMsg)), ms)
  );
  return Promise.race([promise, timeout]);
};

// Update HuggingFace token for a user (uses dedicated hf_tokens table)
export const updateHuggingFaceToken = async (userId: string, hfToken: string, hfUsername?: string) => {
  console.log('[HF Token] Starting upsert for user:', userId);
  
  // Check if Supabase is properly configured
  if (!isSupabaseConfigured()) {
    console.error('[HF Token] Supabase not configured!');
    return null;
  }
  
  try {
    const tokenData = { 
      user_id: userId,
      huggingface_token: hfToken || null,
      huggingface_username: hfUsername || null,
      updated_at: new Date().toISOString()
    };
    console.log('[HF Token] Upsert payload:', { ...tokenData, huggingface_token: hfToken ? '****' : null });
    
    // Get current session for auth header
    const { data: sessionData } = await supabase.auth.getSession();
    const accessToken = sessionData?.session?.access_token;
    console.log('[HF Token] Has auth session:', !!accessToken);
    
    // Use direct REST API call instead of Supabase client (more reliable)
    const config = getSupabaseConfig();
    const response = await fetch(`${config.url}/rest/v1/hf_tokens`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'apikey': config.anonKey,
        'Authorization': `Bearer ${accessToken || config.anonKey}`,
        'Prefer': 'resolution=merge-duplicates,return=representation'
      },
      body: JSON.stringify(tokenData)
    });
    
    console.log('[HF Token] REST API response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[HF Token] REST API error:', response.status, errorText);
      return null;
    }
    
    const data = await response.json();
    console.log('[HF Token] Upsert successful!', data);
    
    // Clear cached status so next check fetches fresh data
    clearHfStatusCache();
    
    return Array.isArray(data) ? data[0] : data;
  } catch (err: any) {
    console.error('[HF Token] Error:', err?.message || err);
    return null;
  }
};

// Forward declaration for clearHfStatusCache (defined below getHuggingFaceStatus)
let clearHfStatusCache: () => void;

// Get HuggingFace token status for a user (from dedicated hf_tokens table)
// Debounce tracking to prevent multiple simultaneous calls
let hfStatusCheckInProgress = false;
let lastHfStatusResult: { connected: boolean; username?: string; tokenMasked?: string | null } | null = null;
let lastHfStatusUserId: string | null = null;

export const getHuggingFaceStatus = async (userId: string) => {
  console.log('[HF Status] Checking HF connection for user:', userId);
  
  // Return cached result if a check is already in progress for the same user
  if (hfStatusCheckInProgress && lastHfStatusUserId === userId && lastHfStatusResult !== null) {
    console.log('[HF Status] Check in progress, returning cached result');
    return lastHfStatusResult;
  }
  
  hfStatusCheckInProgress = true;
  lastHfStatusUserId = userId;
  
  try {
    // Get current session for auth header
    const { data: sessionData, error: sessionError } = await supabase.auth.getSession();
    
    if (sessionError) {
      console.error('[HF Status] Session error:', sessionError);
    }
    
    const accessToken = sessionData?.session?.access_token;
    console.log('[HF Status] Has valid session:', !!accessToken);
    
    // Use direct REST API call
    const config = getSupabaseConfig();
    const response = await fetch(
      `${config.url}/rest/v1/hf_tokens?user_id=eq.${userId}&select=huggingface_token,huggingface_username`,
      {
        method: 'GET',
        headers: {
          'apikey': config.anonKey,
          'Authorization': `Bearer ${accessToken || config.anonKey}`,
        }
      }
    );
    
    console.log('[HF Status] REST API response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[HF Status] REST API error:', response.status, errorText);
      lastHfStatusResult = { connected: false };
      return lastHfStatusResult;
    }
    
    const data = await response.json();
    console.log('[HF Status] Query result:', data);
    
    if (!data || data.length === 0) {
      console.log('[HF Status] No token found for user');
      lastHfStatusResult = { connected: false };
      return lastHfStatusResult;
    }
    
    const row = data[0];
    const result = {
      connected: !!row.huggingface_token,
      username: row.huggingface_username,
      tokenMasked: row.huggingface_token ? `hf_****${row.huggingface_token.slice(-4)}` : null
    };
    
    console.log('[HF Status] Result:', result.connected ? `Connected as ${result.username}` : 'Not connected');
    lastHfStatusResult = result;
    return result;
  } catch (err: any) {
    console.error('[HF Status] Error:', err?.message || err);
    lastHfStatusResult = { connected: false };
    return lastHfStatusResult;
  } finally {
    hfStatusCheckInProgress = false;
  }
};

// Clear cached HF status (call after token updates) - assign to the forward-declared variable
clearHfStatusCache = () => {
  console.log('[HF Status] Clearing cached status');
  lastHfStatusResult = null;
  lastHfStatusUserId = null;
  hfStatusCheckInProgress = false;
};

// Export the function for external use
export { clearHfStatusCache };

// Get the actual HuggingFace token (for export functionality)
export const getHuggingFaceToken = async (userId: string): Promise<string | null> => {
  console.log('[HF Token] Getting full token for user:', userId);
  
  try {
    const { data, error } = await supabase
      .from('hf_tokens')
      .select('huggingface_token')
      .eq('user_id', userId)
      .maybeSingle();
    
    if (error || !data) {
      console.error('[HF Token] Failed to get token:', error?.message);
      return null;
    }
    
    return data.huggingface_token;
  } catch (err: any) {
    console.error('[HF Token] Error:', err?.message || err);
    return null;
  }
};

