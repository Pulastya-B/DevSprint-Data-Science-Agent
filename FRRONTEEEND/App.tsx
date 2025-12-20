
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

const App: React.FC = () => {
  const [view, setView] = useState<'landing' | 'chat'>('landing');

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
        
        <button 
          onClick={() => setView('chat')}
          className="px-5 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm font-medium transition-all"
        >
          Launch Console
        </button>
      </nav>

      <main>
        <HeroGeometric onChatClick={() => setView('chat')} />
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

export default App;
