
import React, { useState } from 'react';
import { HeroGeometric } from './components/HeroGeometric';
import ProblemSolution from './components/ProblemSolution';
import KeyCapabilities from './components/KeyCapabilities';
import Process from './components/Process';
import TechStack from './components/TechStack';
import Footer from './components/Footer';
import { BackgroundPaths } from './components/BackgroundPaths';
import { Logo } from './components/Logo';
import { ChatInterface } from './components/ChatInterface';
import { AuthPage } from './components/AuthPage';
import { AuthProvider, useAuth } from './lib/AuthContext';
import { User, LogOut, Loader2 } from 'lucide-react';

// Inner app component that uses auth context
const AppContent: React.FC = () => {
  const [view, setView] = useState<'landing' | 'chat' | 'auth'>('landing');
  const { user, isAuthenticated, loading, signOut, isConfigured, needsOnboarding } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);

  // Close user menu when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (showUserMenu && !target.closest('.user-menu-container')) {
        setShowUserMenu(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showUserMenu]);

  // If user is authenticated but needs onboarding, show auth page
  React.useEffect(() => {
    if (isAuthenticated && needsOnboarding && view !== 'auth') {
      console.log('User needs onboarding, showing form...');
      setView('auth');
    }
  }, [isAuthenticated, needsOnboarding, view]);

  // Handle launch console - redirect to auth if not logged in
  const handleLaunchConsole = () => {
    if (isAuthenticated) {
      setView('chat');
    } else {
      // Redirect to auth page first
      setView('auth');
    }
  };

  // Show loading state only briefly
  if (loading) {
    return (
      <div className="min-h-screen bg-[#030303] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Logo className="w-16 h-16 animate-pulse" />
          <div className="flex items-center gap-2 text-white/50">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Loading...</span>
          </div>
        </div>
      </div>
    );
  }

  // Show auth page
  if (view === 'auth') {
    return (
      <AuthPage 
        onSuccess={() => setView('chat')} 
        onSkip={() => setView('chat')}
      />
    );
  }

  if (view === 'chat') {
    return <ChatInterface onBack={() => setView('landing')} />;
  }

  return (
    <div className="min-h-screen bg-[#030303] text-white selection:bg-indigo-500/30">
      {/* Navigation (Overlay) */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex justify-between items-center px-6 py-4 backdrop-blur-md bg-[#030303]/20 border-b border-white/5">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => setView('landing')}>
          <Logo className="w-10 h-10" />
          <span className="font-bold tracking-tight text-lg hidden sm:block uppercase text-white">
            DATA SCIENCE AGENT
          </span>
        </div>
        
        <div className="flex items-center gap-3">
          {/* User menu - only show if Supabase is configured */}
          {isAuthenticated ? (
            <div className="relative user-menu-container">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm transition-all"
              >
                <User className="w-4 h-4" />
                <span className="hidden sm:block max-w-[120px] truncate">
                  {user?.email?.split('@')[0]}
                </span>
              </button>
              
              {showUserMenu && (
                <div className="absolute right-0 mt-2 w-48 bg-[#1a1a1a] border border-white/10 rounded-lg shadow-xl py-1 z-50">
                  <div className="px-4 py-2 border-b border-white/10">
                    <p className="text-xs text-white/50">Signed in as</p>
                    <p className="text-sm text-white truncate">{user?.email}</p>
                  </div>
                  <button
                    onClick={async () => {
                      console.log('Sign out clicked');
                      setShowUserMenu(false);
                      try {
                        await signOut();
                        console.log('Sign out successful');
                        setView('landing');
                      } catch (error) {
                        console.error('Sign out failed:', error);
                        alert('Failed to sign out. Please try again.');
                      }
                    }}
                    className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-400 hover:bg-white/5 transition-colors text-left"
                  >
                    <LogOut className="w-4 h-4" />
                    Sign Out
                  </button>
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={() => setView('auth')}
              className="px-4 py-2 text-sm text-white/70 hover:text-white transition-colors"
            >
              Sign In
            </button>
          )}
          
          <button 
            onClick={handleLaunchConsole}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 border border-indigo-500/50 rounded-lg text-sm font-medium transition-all text-white"
          >
            {isAuthenticated ? 'Launch Console' : 'Get Started'}
          </button>
        </div>
      </nav>

      <main>
        <HeroGeometric onChatClick={handleLaunchConsole} />
        <TechStack />
        <ProblemSolution />
        <KeyCapabilities />
        
        {/* Transitional background paths section */}
        <BackgroundPaths 
            title="Intelligence Without Limits" 
            subtitle="The agent continuously learns from your specific domain, optimizing its own tools and reasoning strategies to solve your hardest data challenges."
        />

        <Process />
      </main>

      <Footer />
    </div>
  );
};

// Wrap with AuthProvider
const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;