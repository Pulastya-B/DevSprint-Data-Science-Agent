import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, Session, AuthChangeEvent } from '@supabase/supabase-js';
import { supabase, startUserSession, endUserSession, isSupabaseConfigured } from './supabase';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  dbSessionId: string | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<{ error: any }>;
  signUp: (email: string, password: string) => Promise<{ error: any }>;
  signInWithGoogle: () => Promise<{ error: any }>;
  signInWithGithub: () => Promise<{ error: any }>;
  signOut: () => Promise<void>;
  isAuthenticated: boolean;
  isConfigured: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [dbSessionId, setDbSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
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
        } else if (event === 'SIGNED_OUT') {
          // End tracking session
          if (dbSessionId) {
            await endUserSession(dbSessionId);
            setDbSessionId(null);
          }
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

  const signOut = async () => {
    if (dbSessionId) {
      await endUserSession(dbSessionId);
      setDbSessionId(null);
    }
    await supabase.auth.signOut();
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        dbSessionId,
        loading,
        signIn,
        signUp,
        signInWithGoogle,
        signInWithGithub,
        signOut,
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
