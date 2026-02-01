import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, Session, AuthChangeEvent } from '@supabase/supabase-js';
import { supabase, startUserSession, endUserSession, isSupabaseConfigured, getUserProfile } from './supabase';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  dbSessionId: string | null;
  loading: boolean;
  needsOnboarding: boolean;
  signIn: (email: string, password: string) => Promise<{ error: any }>;
  signUp: (email: string, password: string) => Promise<{ error: any }>;
  signInWithGoogle: () => Promise<{ error: any }>;
  signInWithGithub: () => Promise<{ error: any }>;
  signOut: () => Promise<void>;
  refreshOnboardingStatus: () => Promise<void>;
  isAuthenticated: boolean;
  isConfigured: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [dbSessionId, setDbSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);
  const configured = isSupabaseConfigured();

  useEffect(() => {
    // If Supabase is not configured, skip auth initialization
    if (!configured) {
      setLoading(false);
      return;
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
      
      // Start tracking session if user is logged in
      if (session?.user) {
        startUserSession(session.user.id, session.user.email).then((dbSession) => {
          if (dbSession) {
            setDbSessionId(dbSession.id);
          }
        
        // Check if user needs onboarding
        getUserProfile(session.user.id).then((profile) => {
          setNeedsOnboarding(!profile || !profile.onboarding_completed);
        });
        });
      }
    }).catch((err) => {
      console.error('Failed to get session:', err);
      setLoading(false);
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event: AuthChangeEvent, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);

        if (event === 'SIGNED_IN' && session?.user) {
          // Start new tracking session
          const dbSession = await startUserSession(session.user.id, session.user.email);
          if (dbSession) {
            setDbSessionId(dbSession.id);
          }
          
          // Check if user needs onboarding
          const profile = await getUserProfile(session.user.id);
          setNeedsOnboarding(!profile || !profile.onboarding_completed);
        } else if (event === 'SIGNED_OUT') {
          // End tracking session
          if (dbSessionId) {
            await endUserSession(dbSessionId);
            setDbSessionId(null);
          }
          setNeedsOnboarding(false);
        }
      }
    );

    // Cleanup on unmount
    return () => {
      subscription.unsubscribe();
      // End session when component unmounts
      if (dbSessionId) {
        endUserSession(dbSessionId);
      }
    };
  }, [configured]);

  const signIn = async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    return { error };
  };

  const signUp = async (email: string, password: string) => {
    const { error } = await supabase.auth.signUp({ email, password });
    return { error };
  };

  const signInWithGoogle = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: window.location.origin
      }
    });
    return { error };
  };

  const signInWithGithub = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'github',
      options: {
        redirectTo: window.location.origin
      }
    });
    return { error };
  };

  const refreshOnboardingStatus = async () => {
    if (user) {
      const profile = await getUserProfile(user.id);
      setNeedsOnboarding(!profile || !profile.onboarding_completed);
    }
  };

  const signOut = async () => {
    try {
      if (dbSessionId) {
        await endUserSession(dbSessionId);
        setDbSessionId(null);
      }
      const { error } = await supabase.auth.signOut();
      if (error) {
        console.error('Sign out error:', error);
        throw error;
      }
    } catch (error) {
      console.error('Sign out failed:', error);
      throw error;
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        dbSessionId,
        loading,
        needsOnboarding,
        signIn,
        signUp,
        signInWithGoogle,
        signInWithGithub,
        signOut,
        refreshOnboardingStatus,
        isAuthenticated: !!user,
        isConfigured: configured
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
