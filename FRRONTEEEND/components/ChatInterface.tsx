
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Plus, Search, Settings, MoreHorizontal, User, Bot, ArrowLeft, Paperclip, Sparkles, Trash2, X, Upload, Package, FileText, BarChart3, ChevronRight, LogOut, AlertTriangle, Loader2, Check, CloudUpload } from 'lucide-react';
import { cn } from '../lib/utils';
import { Logo } from './Logo';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useAuth } from '../lib/AuthContext';
import { trackQuery, incrementSessionQueries, getHuggingFaceStatus } from '../lib/supabase';
import { SettingsModal } from './SettingsModal';
import { PipelineView, PipelineStep } from './PipelineView';

// HuggingFace logo SVG component for the export button
const HuggingFaceLogo = ({ className = "w-4 h-4" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 120 120" fill="currentColor">
    <path d="M60 0C26.863 0 0 26.863 0 60s26.863 60 60 60 60-26.863 60-60S93.137 0 60 0zm0 10c27.614 0 50 22.386 50 50s-22.386 50-50 50S10 87.614 10 60 32.386 10 60 10z"/>
    <circle cx="40" cy="50" r="8"/>
    <circle cx="80" cy="50" r="8"/>
    <path d="M40 75c0 11.046 8.954 20 20 20s20-8.954 20-20H40z"/>
  </svg>
);

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  file?: {
    name: string;
    size: number;
  };
  reports?: Array<{
    name: string;
    path: string;
  }>;
  plots?: Array<{
    title: string;
    url: string;
    type?: 'image' | 'html';
  }>;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: Date;
}

// Generate a unique local session ID (not a backend UUID)
const generateLocalSessionId = () => `local_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

// Initial session ID - generated once when module loads
const INITIAL_SESSION_ID = generateLocalSessionId();

// LocalStorage key for persisting sessions
const SESSIONS_STORAGE_KEY = 'ds_agent_chat_sessions';
const ACTIVE_SESSION_STORAGE_KEY = 'ds_agent_active_session';

// Clean up malformed markdown - fixes common LLM formatting issues
// This is CRITICAL for proper rendering of inline code and lists
const cleanMarkdown = (content: string): string => {
  if (!content) return '';
  
  let cleaned = content;
  
  // PHASE 0: Strip wrapping code fences that LLMs add around markdown
  // This causes ReactMarkdown to render the ENTIRE response as a <code> block
  // instead of parsing the markdown. Must be done FIRST.
  cleaned = cleaned.replace(/^\s*```(?:markdown|md|text)?\s*\n/, '');
  cleaned = cleaned.replace(/\n\s*```\s*$/, '');
  
  // PHASE 1: Fix inline code that got split across lines
  // Pattern: `code` followed by newline(s) then comma
  cleaned = cleaned.replace(/`([^`\n]+)`\s*\n+\s*,/g, '`$1`, ');
  
  // Pattern: comma followed by newline(s) then `code`
  cleaned = cleaned.replace(/,\s*\n+\s*`([^`\n]+)`/g, ', `$1`');
  
  // Pattern: `code` followed by newline(s) then ", and"
  cleaned = cleaned.replace(/`([^`\n]+)`\s*\n+\s*,\s*and\s/gi, '`$1`, and ');
  
  // Pattern: `code` newline(s) `code` (consecutive code blocks that should be inline)
  cleaned = cleaned.replace(/`([^`\n]+)`\s*\n+\s*`([^`\n]+)`/g, '`$1` `$2`');
  
  // PHASE 2: Fix text that got split from inline code
  // Pattern: word(s) + newline + `code` + newline + word(s)
  cleaned = cleaned.replace(/(\w+)\s*\n+\s*`([^`\n]+)`\s*\n+\s*(\w+)/g, '$1 `$2` $3');
  
  // Pattern: "the" or "on" or "for" etc + newline + `code`
  cleaned = cleaned.replace(/(the|on|for|with|using|from|in|and|or|to|of|between)\s*\n+\s*`([^`\n]+)`/gi, '$1 `$2`');
  
  // Pattern: `code` + newline + common follow words
  cleaned = cleaned.replace(/`([^`\n]+)`\s*\n+\s*(column|target|feature|variable|field|and|or|to|from|using|with)/gi, '`$1` $2');
  
  // PHASE 3: Fix orphaned punctuation
  // Comma on its own line
  cleaned = cleaned.replace(/\n\s*,\s*\n/g, ', ');
  cleaned = cleaned.replace(/\n\s*,\s*$/gm, ', ');
  
  // Period on its own line (but preserve paragraph breaks)
  cleaned = cleaned.replace(/\n\s*\.\s*\n(?!\n)/g, '. ');
  
  // "and" on its own line between code blocks
  cleaned = cleaned.replace(/`([^`\n]+)`\s*\n+\s*,?\s*and\s*\n+\s*`([^`\n]+)`/gi, '`$1`, and `$2`');
  
  // PHASE 4: Fix list items with broken inline code
  // Pattern: "• Encoded" + newline + `code` + newline + ","
  cleaned = cleaned.replace(/([-•*]\s*\w+[^`\n]*)\n+\s*`([^`\n]+)`\s*\n+\s*,/g, '$1 `$2`,');
  
  // Pattern: list item ending with newline before `code`
  cleaned = cleaned.replace(/([-•*]\s*[^`\n]+)\n+\s*`([^`\n]+)`/g, '$1 `$2`');
  
  // PHASE 5: Fix specific patterns from the screenshots
  // "Encoded" + newline + `code`
  cleaned = cleaned.replace(/Encoded\s*\n+\s*`/gi, 'Encoded `');
  
  // "using" + `code`
  cleaned = cleaned.replace(/using\s*\n+\s*`/gi, 'using `');
  
  // "Output:" + newline + `path`
  cleaned = cleaned.replace(/Output:\s*\n+\s*`/gi, 'Output: `');
  
  // "on the" + newline + `code` + newline + "target"
  cleaned = cleaned.replace(/on the\s*\n+\s*`([^`\n]+)`\s*\n+\s*target/gi, 'on the `$1` target');
  
  // "between" + newline + `code`
  cleaned = cleaned.replace(/between\s*\n+\s*`([^`\n]+)`/gi, 'between `$1`');
  
  // PHASE 6: Aggressive inline code joining
  // If we have `code` on its own line (preceded and followed by non-code text), join it
  // Match: text\n`code`\n or text\n`code`\ntext
  cleaned = cleaned.replace(/([^\n`])\n+`([^`\n]{1,50})`\n+([^\n`])/g, '$1 `$2` $3');
  
  // PHASE 7: Clean up excessive whitespace
  // Multiple spaces to single space (but preserve newlines)
  cleaned = cleaned.replace(/[ \t]+/g, ' ');
  
  // More than 2 consecutive newlines to just 2
  cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
  
  // PHASE 8: Fix table formatting
  // Table cells with broken inline code
  cleaned = cleaned.replace(/\|\s*\n+\s*`([^`]+)`\s*\n+\s*\|/g, '| `$1` |');
  cleaned = cleaned.replace(/\|\s*`([^`]+)`\s*\n+\s*\|/g, '| `$1` |');
  
  // PHASE 9: One more pass for any remaining patterns
  // `code` followed by newline and then more text on same logical line
  cleaned = cleaned.replace(/`([^`\n]+)`\n+(?=[a-z])/gi, '`$1` ');
  
  return cleaned;
};

// Load sessions from localStorage
const loadSessionsFromStorage = (): ChatSession[] => {
  try {
    const stored = localStorage.getItem(SESSIONS_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      // Convert ISO date strings back to Date objects
      return parsed.map((s: any) => ({
        ...s,
        updatedAt: new Date(s.updatedAt),
        messages: s.messages.map((m: any) => ({
          ...m,
          timestamp: new Date(m.timestamp)
        }))
      }));
    }
  } catch (err) {
    console.error('Failed to load sessions from localStorage:', err);
  }
  // Return default session if loading fails
  return [{
    id: INITIAL_SESSION_ID,
    title: 'New Chat',
    messages: [],
    updatedAt: new Date(),
  }];
};

// Save sessions to localStorage
const saveSessionsToStorage = (sessions: ChatSession[]) => {
  try {
    localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(sessions));
  } catch (err) {
    console.error('Failed to save sessions to localStorage:', err);
  }
};

export const ChatInterface: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessionsFromStorage);
  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    // Try to restore last active session
    try {
      const stored = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
      if (stored && sessions.some(s => s.id === stored)) {
        return stored;
      }
    } catch (err) {
      console.error('Failed to load active session:', err);
    }
    return sessions[0]?.id || INITIAL_SESSION_ID;
  });
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [reportModalUrl, setReportModalUrl] = useState<string | null>(null);
  const [reportModalTitle, setReportModalTitle] = useState<string>('Visualization');
  const [showAssets, setShowAssets] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [hfConnected, setHfConnected] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const processedAnalysisRef = useRef<Set<string>>(new Set()); // Track processed analysis_complete events
  const [sseReconnectTrigger, setSseReconnectTrigger] = useState(0); // Force SSE reconnection for follow-up queries
  
  // Pipeline visualization state (reasoning loop)
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([]);
  const [pipelineMode, setPipelineMode] = useState<string | null>(null);
  const [pipelineHypotheses, setPipelineHypotheses] = useState<string[]>([]);
  const pipelineStepCounterRef = useRef(0); // Unique step ID counter
  
  // Auth context for user tracking
  const { user, isAuthenticated, dbSessionId, signOut } = useAuth();
  
  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];
  const hfStatusCheckedRef = useRef(false); // Prevent multiple HF status checks

  // Check HuggingFace connection status (only once on mount or when user changes)
  useEffect(() => {
    const checkHfStatus = async () => {
      if (user?.id && !hfStatusCheckedRef.current) {
        hfStatusCheckedRef.current = true;
        const status = await getHuggingFaceStatus(user.id);
        setHfConnected(status.connected);
      }
    };
    checkHfStatus();
    
    // Reset the ref when user changes (e.g., logout/login)
    return () => {
      if (!user?.id) {
        hfStatusCheckedRef.current = false;
      }
    };
  }, [user?.id]); // Only depend on user.id, not the whole user object

  // Persist sessions to localStorage whenever they change
  useEffect(() => {
    saveSessionsToStorage(sessions);
  }, [sessions]);

  // Persist active session ID
  useEffect(() => {
    try {
      localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, activeSessionId);
    } catch (err) {
      console.error('Failed to save active session:', err);
    }
  }, [activeSessionId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activeSession.messages, isTyping]);

  // Clear state when switching sessions
  useEffect(() => {
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    // Clear processed results tracker for new session
    // (keeps results from old session from blocking new ones)
    processedAnalysisRef.current.clear();
  }, [activeSessionId]);

  // Track which session the current SSE connection is for
  const sseSessionRef = useRef<string | null>(null);
  const isCleaningUpRef = useRef<boolean>(false); // Prevent race conditions during cleanup

  // Connect to SSE when we receive a valid backend UUID
  useEffect(() => {
    // Only connect if we have a backend UUID (contains hyphens, not a local_ ID)
    const isBackendUUID = activeSessionId && activeSessionId.includes('-') && !activeSessionId.startsWith('local_');
    
    if (!isBackendUUID) {
      // No backend session yet - close any existing connection
      if (eventSourceRef.current && !isCleaningUpRef.current) {
        console.log('🔌 Closing SSE - no backend session');
        isCleaningUpRef.current = true;
        eventSourceRef.current.close();
        eventSourceRef.current = null;
        sseSessionRef.current = null;
        isCleaningUpRef.current = false;
      }
      return;
    }

    // Check if we're already connected to the correct session
    // BUT: If sseReconnectTrigger changed, we MUST reconnect (follow-up query sent)
    if (sseSessionRef.current === activeSessionId && sseReconnectTrigger === 0) {
      // Same session - check if connection is still alive
      if (eventSourceRef.current && eventSourceRef.current.readyState !== 2) {
        console.log('♻️ Reusing existing SSE connection for same session');
        return;
      }
    }
    
    // If reconnect was triggered, log it
    if (sseReconnectTrigger > 0) {
      console.log(`🔄 SSE reconnect triggered (trigger=${sseReconnectTrigger})`);
    }

    // Different session or connection is closed - need new connection
    // First, close any existing connection
    if (eventSourceRef.current && !isCleaningUpRef.current) {
      const oldSession = sseSessionRef.current?.slice(0, 8) || 'unknown';
      console.log(`🔄 Closing SSE for ${oldSession}... before switching to ${activeSessionId.slice(0, 8)}...`);
      isCleaningUpRef.current = true;
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      isCleaningUpRef.current = false;
    }

    // Small delay to ensure old connection is fully closed
    const timeoutId = setTimeout(() => {
      // Double-check we're still on the same session (might have switched again)
      if (activeSessionId !== sseSessionRef.current) {
        console.log(`🔌 Opening new SSE connection to session: ${activeSessionId.slice(0, 8)}...`);
        
        const API_URL = window.location.origin;
        const eventSource = new EventSource(`${API_URL}/api/progress/stream/${activeSessionId}`);
        sseSessionRef.current = activeSessionId;
        eventSourceRef.current = eventSource;

        eventSource.onopen = () => {
          console.log('✅ SSE connection established');
        };

        // Handle all incoming messages
        eventSource.onmessage = (e) => {
          console.log('📨 SSE received:', e.data);
          try {
            const data = JSON.parse(e.data);
            
            // Handle different event types
            if (data.type === 'connected') {
              console.log('🔗 Connected to progress stream');
            } else if (data.type === 'agent_assigned') {
              // 🤖 Multi-Agent: Display which specialist agent is handling the task
              const agentMessage = `${data.emoji} **${data.agent}** assigned\n_${data.description}_`;
              setCurrentStep(agentMessage);
              console.log(`🤖 Agent assigned: ${data.agent}`);
            } else if (data.type === 'tool_executing') {
              setCurrentStep(data.message || `🔧 Executing: ${data.tool}`);
              // Add pipeline step if in reasoning mode
              if (pipelineMode) {
                const stepId = `act-${++pipelineStepCounterRef.current}`;
                setPipelineSteps(prev => [...prev, {
                  id: stepId,
                  type: 'act',
                  status: 'active',
                  title: `Executing: ${data.tool}`,
                  subtitle: data.message || '',
                  tool: data.tool,
                  timestamp: new Date()
                }]);
              }
            } else if (data.type === 'tool_completed') {
              setCurrentStep(data.message || `✓ Completed: ${data.tool}`);
              // Update pipeline step status
              if (pipelineMode) {
                setPipelineSteps(prev => prev.map(s =>
                  s.type === 'act' && s.status === 'active' ? { ...s, status: 'completed' as const } : s
                ));
              }
            } else if (data.type === 'tool_failed') {
              setCurrentStep(data.message || `❌ Failed: ${data.tool}`);
              // Update pipeline step status
              if (pipelineMode) {
                setPipelineSteps(prev => prev.map(s =>
                  s.type === 'act' && s.status === 'active' ? { ...s, status: 'failed' as const, subtitle: data.message || 'Tool failed' } : s
                ));
              }
            } else if (data.type === 'token_update') {
              // Optional: Display token budget updates
              console.log('💰 Token update:', data.message);
            } else if (data.type === 'intent_classified') {
              // 🎯 Reasoning Loop: Intent classification result
              console.log(`🎯 Intent: ${data.mode} (${Math.round(data.confidence * 100)}%)`);
              setPipelineMode(data.mode);
              const stepId = `intent-${++pipelineStepCounterRef.current}`;
              setPipelineSteps(prev => [...prev, {
                id: stepId,
                type: 'intent',
                status: 'completed',
                title: `Intent: ${data.mode.charAt(0).toUpperCase() + data.mode.slice(1)}`,
                subtitle: data.sub_intent || data.reasoning,
                detail: data.reasoning,
                confidence: data.confidence,
                timestamp: new Date()
              }]);
            } else if (data.type === 'reasoning_mode') {
              // 🧠 Reasoning Loop activated
              console.log(`🧠 Reasoning mode: ${data.mode}`);
              setPipelineMode(data.mode);
              setCurrentStep(data.message || `🧠 Reasoning Loop (${data.mode})`);
            } else if (data.type === 'hypotheses_generated') {
              // 💡 Exploratory mode: hypotheses generated
              console.log(`💡 ${data.count} hypotheses generated`);
              setPipelineHypotheses(data.hypotheses || []);
              const stepId = `hyp-${++pipelineStepCounterRef.current}`;
              setPipelineSteps(prev => [...prev, {
                id: stepId,
                type: 'hypothesis',
                status: 'completed',
                title: `${data.count} Hypotheses Generated`,
                subtitle: data.hypotheses?.[0] || '',
                detail: (data.hypotheses || []).map((h: string, i: number) => `${i + 1}. ${h}`).join('\n'),
                timestamp: new Date()
              }]);
            } else if (data.type === 'reasoning_step') {
              // 🤔 Reasoning step: LLM decided next action
              console.log(`🤔 Iteration ${data.iteration}: ${data.tool}`);
              // Mark previous "reason" steps as completed
              setPipelineSteps(prev => prev.map(s => 
                s.type === 'reason' && s.status === 'active' ? { ...s, status: 'completed' as const } : s
              ));
              const stepId = `reason-${++pipelineStepCounterRef.current}`;
              setPipelineSteps(prev => [...prev, {
                id: stepId,
                type: 'reason',
                status: 'completed',
                title: `Reason → ${data.tool}`,
                subtitle: data.hypothesis || '',
                detail: data.reasoning,
                iteration: data.iteration,
                tool: data.tool,
                timestamp: new Date()
              }]);
            } else if (data.type === 'finding_discovered') {
              // 🔬 Finding from evaluation step
              console.log(`🔬 Finding (confidence: ${Math.round(data.confidence * 100)}%)`);
              const stepId = `finding-${++pipelineStepCounterRef.current}`;
              setPipelineSteps(prev => [...prev, {
                id: stepId,
                type: 'finding',
                status: 'completed',
                title: data.answered ? '✓ Question Answered' : 'Finding Discovered',
                subtitle: data.interpretation?.substring(0, 100) || '',
                detail: data.interpretation,
                confidence: data.confidence,
                iteration: data.iteration,
                timestamp: new Date()
              }]);
            } else if (data.type === 'analysis_failed') {
              console.log('❌ Analysis failed', data);
              setIsTyping(false);
              
              // Reset pipeline state
              setPipelineSteps([]);
              setPipelineMode(null);
              setPipelineHypotheses([]);
              
              // Show error message to user - add to sessions
              setSessions(prev => prev.map(s => {
                if (s.id === activeSessionId) {
                  return {
                    ...s,
                    messages: [...s.messages, {
                      id: Date.now().toString(),
                      role: 'assistant' as const,
                      content: data.message || data.error || '❌ Analysis failed',
                      timestamp: new Date(),
                    }],
                    updatedAt: new Date()
                  };
                }
                return s;
              }));
              
              setCurrentStep('');
            } else if (data.type === 'analysis_complete') {
              console.log('✅ Analysis completed', data.result);
              setIsTyping(false);
              
              // Reset pipeline state
              setPipelineSteps([]);
              setPipelineMode(null);
              setPipelineHypotheses([]);
              
              // Create a unique key based on actual workflow content to prevent duplicates
              // Use the last tool executed + summary hash for uniqueness
              const lastTool = data.result?.workflow_history?.[data.result.workflow_history.length - 1]?.tool || 'unknown';
              const summarySnippet = (data.result?.summary || '').substring(0, 50);
              const resultKey = `${activeSessionId}-${lastTool}-${summarySnippet}`;
              
              // Only process if we haven't seen this exact result before
              if (!processedAnalysisRef.current.has(resultKey)) {
                console.log('🆕 New analysis result, processing...', resultKey);
                processedAnalysisRef.current.add(resultKey);
                
                // Process the final result with the current session ID
                if (data.result) {
                  processAnalysisResult(data.result, activeSessionId);
                }
                
                // Close SSE connection after receiving final result to prevent duplicate events
                console.log('🔒 Closing SSE connection after analysis complete');
                if (eventSourceRef.current) {
                  eventSourceRef.current.close();
                  eventSourceRef.current = null;
                  sseSessionRef.current = null;
                }
              } else {
                console.log('⏭️ Skipping duplicate analysis result', resultKey);
                // MUST close EventSource on duplicates to prevent reconnect loop
                if (eventSourceRef.current) {
                  eventSourceRef.current.close();
                  eventSourceRef.current = null;
                  sseSessionRef.current = null;
                }
              }
            }
          } catch (err) {
            console.error('❌ Error parsing SSE event:', err, e.data);
          }
        };

        // Handle errors - DON'T immediately close, just log
        eventSource.onerror = (err) => {
          console.error('❌ SSE connection error/closed:', err);
        };
      }
    }, 50); // 50ms delay to ensure old connection closes

    // Cleanup on unmount or session change
    return () => {
      clearTimeout(timeoutId); // Clear timeout if component unmounts
      if (eventSourceRef.current && !isCleaningUpRef.current) {
        console.log('🧹 Cleaning up SSE connection on unmount/session change');
        isCleaningUpRef.current = true;
        eventSourceRef.current.close();
        eventSourceRef.current = null;
        sseSessionRef.current = null;
        isCleaningUpRef.current = false;
      }
    };
  }, [activeSessionId, sseReconnectTrigger]); // 🔄 Also reconnect when trigger changes

  const processAnalysisResult = (result: any, sessionId: string) => {
    // Extract and display the analysis result from SSE
    let assistantContent = '✅ Analysis Complete!\n\n';
    let reports: Array<{name: string, path: string}> = [];
    let plots: Array<{title: string, url: string, type?: 'image' | 'html'}> = [];
    
    // PRIORITY 1: Extract plots from main result.plots array (backend enhanced summary)
    if (result.plots && Array.isArray(result.plots)) {
      result.plots.forEach((plot: any) => {
        plots.push({
          title: plot.title || 'Visualization',
          url: plot.url || plot.path,
          type: plot.type || (plot.url?.endsWith('.html') ? 'html' : 'image')
        });
      });
    }
    
    // PRIORITY 2: Extract plots and reports from workflow_history (for backward compatibility)
    if (result.workflow_history) {
      const reportTools = ['generate_ydata_profiling_report', 'generate_plotly_dashboard', 'generate_all_plots'];
      const plotTools = [
        'generate_interactive_correlation_heatmap',
        'generate_interactive_scatter',
        'generate_interactive_histogram',
        'generate_interactive_box_plots',
        'generate_interactive_time_series',
        'generate_eda_plots',
        'generate_data_quality_plots',
        'analyze_correlations'
      ];
      
      result.workflow_history.forEach((step: any) => {
        if (reportTools.includes(step.tool)) {
          const reportPath = step.result?.output_path || step.result?.report_path || step.arguments?.output_path;
          if (reportPath && (step.result?.success !== false)) {
            reports.push({
              name: step.tool.replace('generate_', '').replace(/_/g, ' ').trim(),
              path: reportPath
            });
          }
        }
        
        // Only extract from workflow if not already in result.plots
        if (plotTools.includes(step.tool) && step.result?.result?.output_path && plots.length === 0) {
          const outputPath = step.result.result.output_path;
          plots.push({
            title: step.tool.replace('generate_', '').replace('interactive_', '').replace(/_/g, ' ').trim(),
            url: outputPath.startsWith('/') ? outputPath : `/outputs/${outputPath.replace('./outputs/', '')}`,
            type: outputPath.endsWith('.html') ? 'html' : 'image'
          });
        }
      });
    }
    
    if (reports.length > 0) {
      assistantContent += '📊 **Generated Reports:**\n';
      reports.forEach(r => assistantContent += `- ${r.name}\n`);
      assistantContent += '\n';
    }
    
    if (plots.length > 0) {
      assistantContent += `📈 **Generated ${plots.length} Visualizations**\n\n`;
    }
    
    // Extract summary from backend (field changed from final_answer to summary)
    const summaryText = result.summary || result.final_answer || 'Analysis complete. Check the generated artifacts.';
    assistantContent += summaryText;
    
    // Add assistant message with result
    const assistantMessage: Message = {
      id: Date.now().toString(),
      role: 'assistant',
      content: assistantContent,
      timestamp: new Date(),
      reports,
      plots
    };
    
    // Get current session and add message
    setSessions(prev => prev.map(s => {
      if (s.id === sessionId) {
        return { ...s, messages: [...s.messages, assistantMessage], updatedAt: new Date() };
      }
      return s;
    }));
  };

  const handleSend = async () => {
    if ((!input.trim() && !uploadedFile) || isTyping) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input || (uploadedFile ? `Uploaded: ${uploadedFile.name}` : ''),
      timestamp: new Date(),
      file: uploadedFile ? { name: uploadedFile.name, size: uploadedFile.size } : undefined,
    };

    const newMessages = [...activeSession.messages, userMessage];
    updateSession(activeSessionId, newMessages);
    setInput('');
    
    // Show loading indicator immediately (for UI feedback)
    setIsTyping(true);
    
    // Reset pipeline state for new analysis
    setPipelineSteps([]);
    setPipelineMode(null);
    setPipelineHypotheses([]);
    pipelineStepCounterRef.current = 0;

    try {
      
      // Use the current origin if running on same server, otherwise use env variable
      const API_URL = window.location.origin;
      console.log('API URL:', API_URL);
      
      let response;
      const sessionKey = activeSessionId || 'default';
      
      // Detect if we have an active backend session (UUID format)
      const hasBackendSession = sessionKey.includes('-') && sessionKey.length > 20;
      
      // Check if there's a recent file analysis in the conversation
      const recentFileMessage = newMessages.find(m => m.file || m.content.includes('Uploaded:'));
      const isFileAnalysis = uploadedFile || recentFileMessage;
      
      // 🔑 KEY CHANGE: Always use /run-async if:
      // 1. User is uploading a file, OR
      // 2. We have an active backend session (meaning we've done file analysis before)
      if (uploadedFile || hasBackendSession) {
        // Use /run-async endpoint for file analysis or follow-up questions
        const formData = new FormData();
        
        if (uploadedFile) {
          formData.append('file', uploadedFile);
          formData.append('task_description', input || 'Analyze this dataset and provide insights');
        } else {
          // Follow-up query - send task description only, backend will use cached dataset
          formData.append('task_description', input);
          console.log(`📤 Follow-up query for session ${sessionKey.slice(0, 8)}...`);
          
          // 🔄 CRITICAL: Force SSE reconnection for follow-up queries
          // The previous SSE was closed after analysis_complete, need new connection
          console.log('🔄 Triggering SSE reconnection for follow-up query...');
          setSseReconnectTrigger(prev => prev + 1);
        }
        
        formData.append('session_id', sessionKey);
        formData.append('use_cache', 'false');  // Disabled to show multi-agent execution
        formData.append('max_iterations', '20');
        
        // Track query start time for analytics
        const queryStartTime = Date.now();
        
        response = await fetch(`${API_URL}/run-async`, {
          method: 'POST',
          body: formData
        });
        
        // 📊 Track analytics (non-blocking)
        if (user || dbSessionId) {
          trackQuery({
            user_id: user?.id || 'anonymous',
            user_email: user?.email,
            session_id: sessionKey,
            query: input || 'File analysis',
            success: response.ok,
            error_message: response.ok ? undefined : `HTTP ${response.status}`
          }).catch(console.error);
          
          if (dbSessionId) {
            incrementSessionQueries(dbSessionId).catch(console.error);
          }
        }
        
        setUploadedFile(null);
      } else {
        // No file and no backend session - use simple chat endpoint
        response = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messages: newMessages.map(m => ({
              role: m.role,
              content: m.content
            })),
            stream: false
          })
        });
      }

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      // Store UUID from backend to trigger SSE connection
      if (data.session_id) {
        console.log(`🔑 Session UUID from backend: ${data.session_id}`);
        
        const newSessionId = data.session_id;
        
        // CRITICAL: Update sessions first, then activeSessionId
        // React 18 batches these updates automatically, preventing flicker
        setSessions(prev => prev.map(s => 
          s.id === activeSessionId ? { ...s, id: newSessionId } : s
        ));
        setActiveSessionId(newSessionId);
      }
      
      // For async endpoint, result comes via SSE analysis_complete event
      // For now, just wait for SSE to deliver the result
      if (data.status === 'started') {
        console.log('🚀 Analysis started, waiting for SSE events...');
        return; // Don't process result here, will come via SSE
      }
      
      // Legacy sync endpoint handling (if data.result exists)
      let assistantContent = '';
      let reports: Array<{name: string, path: string}> = [];
      let plots: Array<{title: string, url: string, type?: 'image' | 'html'}> = [];
      
      // Check for reports in any /run endpoint response (not just when file is uploaded)
      if (data.result) {
        const result = data.result;
        assistantContent = `✅ Analysis Complete!\n\n`;
        
        // Extract plots from workflow_history (PRIMARY SOURCE)
        if (result.workflow_history) {
          const reportTools = ['generate_ydata_profiling_report', 'generate_plotly_dashboard', 'generate_all_plots'];
          const plotTools = [
            'generate_interactive_correlation_heatmap',
            'generate_interactive_scatter',
            'generate_interactive_histogram',
            'generate_interactive_box_plots',
            'generate_interactive_time_series',
            'generate_eda_plots',
            'generate_data_quality_plots',
            'analyze_correlations'
          ];
          
          result.workflow_history.forEach((step: any) => {
            // Extract reports
            if (reportTools.includes(step.tool)) {
              const reportPath = step.result?.output_path || step.result?.report_path || step.arguments?.output_path;
              
              if (reportPath && (step.result?.success !== false)) {
                reports.push({
                  name: step.tool.replace('generate_', '').replace(/_/g, ' ').replace('report', '').trim(),
                  path: reportPath
                });
              }
            }
            
            // Extract plots
            if (plotTools.includes(step.tool)) {
              const plotPath = step.result?.output_path || step.arguments?.output_path;
              
              if (plotPath && (step.result?.success !== false)) {
                const plotTitle = step.tool
                  .replace('generate_', '')
                  .replace('interactive_', '')
                  .replace(/_/g, ' ')
                  .replace('plots', 'plot')
                  .trim();
                
                plots.push({
                  title: plotTitle.charAt(0).toUpperCase() + plotTitle.slice(1),
                  url: plotPath.replace('./outputs/', '/outputs/'),
                  type: plotPath.endsWith('.html') ? 'html' : 'image'
                });
              }
            }
          });
        }
        
        // Also check for report paths mentioned in the summary text
        if (result.summary && !reports.length) {
          const reportPathMatch = result.summary.match(/\.(\/outputs\/reports\/[^\s]+\.html)/);
          if (reportPathMatch) {
            reports.push({
              name: 'ydata profiling',
              path: reportPathMatch[1]
            });
          }
        }
        
        if (result.summary) {
          assistantContent += `**Summary:**\n${result.summary}\n\n`;
        }
        
        if (result.workflow_history && result.workflow_history.length > 0) {
          assistantContent += `**Tools Used:** ${result.workflow_history.length} steps\n\n`;
          assistantContent += `**Final Result:**\n${result.final_result || 'Analysis completed successfully'}`;
        }
      } else if (data.success && data.message) {
        // /chat endpoint response
        assistantContent = data.message;
        setIsTyping(false); // Stop thinking animation for /chat responses
      } else {
        throw new Error('Invalid response from API');
      }

      // Aggressive text cleaning to remove malformed content
      assistantContent = assistantContent
        // Remove broken markdown tables (lines with just | symbols)
        .replace(/^\s*\|\s*\|\s*$/gm, '')
        // Remove confusing phrases
        .replace(/Printed in logs \(see above\)/gi, '')
        .replace(/\(see above\)/gi, '')
        .replace(/see above/gi, '')
        // Remove broken table rows (just dashes and pipes)
        .replace(/^\s*[-|]+\s*$/gm, '')
        // Remove code block markers without content
        .replace(/```\s*```/g, '')
        // Remove empty markdown sections
        .replace(/\n{3,}/g, '\n\n')
        // Clean up broken table syntax
        .replace(/\|\s*\n\s*\|/g, '')
        .trim();
      
      updateSession(activeSessionId, [...newMessages, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: assistantContent,
        timestamp: new Date(),
        reports: reports.length > 0 ? reports : undefined,
        plots: plots.length > 0 ? plots : undefined
      }]);
    } catch (error: any) {
      console.error("Chat Error:", error);
      
      let errorMessage = "I'm sorry, I encountered an error processing your request.";
      
      if (error.message) {
        errorMessage += `\n\n**Error:** ${error.message}`;
      }
      
      // Try to parse response error
      try {
        const errorText = await error.text?.();
        if (errorText) {
          const errorData = JSON.parse(errorText);
          if (errorData.detail) {
            errorMessage = `**Error:** ${typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail)}`;
          }
        }
      } catch (e) {
        // Ignore parsing errors
      }
      
      updateSession(activeSessionId, [...newMessages, {
        id: 'err-' + Date.now(),
        role: 'assistant',
        content: errorMessage,
        timestamp: new Date()
      }]);
      
      // On error, stop loading indicator
      setIsTyping(false);
    }
    // NOTE: No finally block - isTyping is set to false by SSE analysis_complete event
  };

  const updateSession = (id: string, messages: Message[]) => {
    setSessions(prev => prev.map(s => {
      if (s.id === id) {
        return { ...s, messages, updatedAt: new Date() };
      }
      return s;
    }));
  };

  const createNewChat = () => {
    // Generate a unique local ID for this chat session
    // The backend will generate the real UUID when the first request is made
    const newId = generateLocalSessionId();
    const newSession: ChatSession = {
      id: newId,
      title: 'New Chat',
      messages: [],
      updatedAt: new Date()
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newId);
    
    // Clear file upload state for new chat
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    // Close any existing SSE connection since this is a fresh chat
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };

  const deleteSession = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (sessions.length === 1) return;
    setSessions(prev => prev.filter(s => s.id !== id));
    if (activeSessionId === id) {
      setActiveSessionId(sessions.find(s => s.id !== id)?.id || '');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const validTypes = ['.csv', '.parquet'];
      const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      
      if (validTypes.includes(fileExt)) {
        setUploadedFile(file);
      } else {
        alert('Please upload a CSV or Parquet file');
      }
    }
  };

  const removeFile = () => {
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#050505] overflow-hidden text-white/90">
      {/* Sidebar */}
      <aside className="w-[280px] hidden md:flex flex-col border-r border-white/5 bg-[#0a0a0a]/50 backdrop-blur-xl">
        <div className="p-4 flex flex-col h-full">
          <div className="flex items-center gap-3 mb-8 px-2">
            <Logo className="w-8 h-8" />
            <span className="font-bold tracking-tight text-sm uppercase">Console</span>
          </div>

          <button 
            onClick={createNewChat}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-sm font-medium mb-6 group"
          >
            <Plus className="w-4 h-4 group-hover:scale-110 transition-transform" />
            New Conversation
          </button>

          <div className="flex-1 overflow-y-auto space-y-2 custom-scrollbar">
            <p className="px-3 text-[10px] uppercase tracking-widest text-white/30 font-bold mb-2">History</p>
            {sessions.map(session => (
              <div
                key={session.id}
                onClick={() => setActiveSessionId(session.id)}
                className={cn(
                  "group flex items-center justify-between px-4 py-3 rounded-xl cursor-pointer transition-all text-sm",
                  activeSessionId === session.id 
                    ? "bg-white/10 text-white border border-white/10 shadow-lg" 
                    : "text-white/40 hover:text-white/70 hover:bg-white/5"
                )}
              >
                <span className="truncate flex-1 pr-2">{session.title}</span>
                <Trash2 
                  onClick={(e) => deleteSession(e, session.id)}
                  className="w-4 h-4 opacity-0 group-hover:opacity-100 hover:text-rose-400 transition-all" 
                />
              </div>
            ))}
          </div>

          <div className="mt-auto pt-4 border-t border-white/5">
            {/* User info section */}
            {isAuthenticated && user ? (
              <div className="flex items-center gap-3 px-2 py-2 mb-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-xs font-bold">
                  {user.email?.[0]?.toUpperCase() || 'U'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">{user.email?.split('@')[0]}</p>
                  <p className="text-xs text-white/40 truncate">{user.email}</p>
                </div>
                <button 
                  onClick={() => signOut()}
                  className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/40 hover:text-red-400"
                  title="Sign Out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-3 px-2 py-2 mb-2">
                <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
                  <User className="w-4 h-4 text-white/50" />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-white/50">Guest User</p>
                  <p className="text-xs text-white/30">Sign in to save history</p>
                </div>
              </div>
            )}
            
            <div className="flex items-center justify-between px-2">
              <button onClick={onBack} className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/40 hover:text-white">
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="flex gap-2">
                <button 
                  onClick={() => setShowSettings(true)}
                  className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/40 hover:text-white"
                  title="Settings"
                >
                  <Settings className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative bg-gradient-to-b from-[#080808] to-[#050505]">
        {/* Top Header */}
        <header className="h-16 flex items-center justify-between px-6 border-b border-white/5 backdrop-blur-md bg-black/20 sticky top-0 z-10">
          <div className="flex items-center gap-4">
             <button onClick={onBack} className="md:hidden p-2 hover:bg-white/5 rounded-lg">
               <ArrowLeft className="w-5 h-5" />
             </button>
             <div>
               <h2 className="text-sm font-bold text-white tracking-tight">{activeSession.title}</h2>
               <p className="text-[10px] text-white/30 font-medium">{activeSession.messages.length} messages in session</p>
             </div>
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setShowAssets(!showAssets)}
              className={cn(
                "p-2 transition-colors rounded-lg",
                showAssets ? "text-emerald-400 bg-emerald-500/10" : "text-white/40 hover:text-white"
              )}
            >
              <Package className="w-5 h-5" />
            </button>
            <button className="p-2 text-white/40 hover:text-white transition-colors">
              <Search className="w-5 h-5" />
            </button>
            <button className="p-2 text-white/40 hover:text-white transition-colors">
              <MoreHorizontal className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Message List */}
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 scroll-smooth"
        >
          {activeSession.messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-4">
               <motion.div 
                 initial={{ opacity: 0, scale: 0.9 }}
                 animate={{ opacity: 1, scale: 1 }}
                 className="w-16 h-16 bg-gradient-to-br from-indigo-500/20 to-rose-500/20 rounded-2xl flex items-center justify-center mb-6 border border-white/10"
               >
                 <Sparkles className="w-8 h-8 text-indigo-400" />
               </motion.div>
               <h1 className="text-2xl font-extrabold text-white mb-3">Welcome, Data Scientist</h1>
               <p className="text-white/40 max-w-sm leading-relaxed text-sm">
                 I'm your autonomous agent ready to profile data, train models, or build dashboards. 
                 Try uploading a dataset or describing your ML objective.
               </p>
               <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-8 w-full max-w-lg">
                  {[
                    "Profile my sales.csv",
                    "Train a XGBoost classifier",
                    "Generate a correlation heatmap",
                    "Explain feature importance"
                  ].map(prompt => (
                    <button 
                      key={prompt}
                      onClick={() => setInput(prompt)}
                      className="text-left px-4 py-3 rounded-xl bg-white/[0.03] border border-white/5 hover:bg-white/5 transition-all text-xs text-white/60 hover:text-white"
                    >
                      "{prompt}"
                    </button>
                  ))}
               </div>
            </div>
          ) : (
            activeSession.messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn(
                  "flex w-full gap-4",
                  msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                )}
              >
                <div className={cn(
                  "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border border-white/10",
                  msg.role === 'user' ? "bg-indigo-500/20" : "bg-white/5"
                )}>
                  {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4 text-indigo-400" />}
                </div>
                <div className={cn(
                  "max-w-[80%] md:max-w-[70%] p-4 rounded-2xl text-sm leading-relaxed",
                  msg.role === 'user' 
                    ? "bg-indigo-600/20 text-indigo-50 border border-indigo-500/20" 
                    : "bg-white/[0.03] text-white/80 border border-white/5"
                )}>
                  {msg.file && (
                    <div className="mb-2 flex items-center gap-2 text-xs bg-white/5 rounded-lg px-3 py-2 border border-white/10">
                      <Paperclip className="w-3 h-3" />
                      <span className="font-medium">{msg.file.name}</span>
                      <span className="text-white/40">({(msg.file.size / 1024).toFixed(1)} KB)</span>
                    </div>
                  )}
                  {msg.role === 'assistant' ? (
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      className="prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-pre:bg-black/40 prose-pre:border prose-pre:border-white/10 prose-headings:text-white prose-strong:text-white prose-li:text-white/80"
                      components={{
                        // Headings
                        h1: ({node, ...props}) => <h1 className="text-xl font-bold text-white mb-4 mt-6 first:mt-0 border-b border-white/10 pb-2" {...props} />,
                        h2: ({node, ...props}) => <h2 className="text-lg font-semibold text-white mb-3 mt-5 first:mt-0" {...props} />,
                        h3: ({node, ...props}) => <h3 className="text-base font-semibold text-indigo-300 mb-2 mt-4 first:mt-0" {...props} />,
                        h4: ({node, ...props}) => <h4 className="text-sm font-semibold text-indigo-200 mb-2 mt-3 first:mt-0" {...props} />,
                        
                        // Paragraphs and text
                        p: ({node, ...props}) => <p className="mb-3 last:mb-0 text-white/80 leading-relaxed" {...props} />,
                        strong: ({node, ...props}) => <strong className="font-semibold text-white" {...props} />,
                        em: ({node, ...props}) => <em className="text-indigo-200 italic" {...props} />,
                        
                        // Lists
                        ul: ({node, ...props}) => <ul className="mb-3 space-y-1.5 list-disc list-outside ml-4" {...props} />,
                        ol: ({node, ...props}) => <ol className="mb-3 space-y-1.5 list-decimal list-outside ml-4" {...props} />,
                        li: ({node, ...props}) => <li className="text-white/80 pl-1" {...props} />,
                        
                        // Code - Smart inline detection
                        // The key insight: react-markdown sets inline=true for `code` inside paragraphs
                        // But if `code` is on its own line, it gets wrapped in <pre> and inline=false
                        // We need to detect "should be inline" cases even when inline=false
                        code: ({node, inline, className, children, ...props}: any) => {
                          const codeContent = String(children).replace(/\n$/, '');
                          const hasLanguage = className && className.startsWith('language-');
                          
                          // Force inline rendering for short, single-line, non-language code
                          // This catches cases where the LLM put `columnName` on its own line
                          const shouldBeInline = inline || (
                            !hasLanguage && 
                            !codeContent.includes('\n') && 
                            codeContent.length < 80 &&
                            // Common patterns that should always be inline
                            (codeContent.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/) ||  // variable names
                             codeContent.match(/^[a-zA-Z_][a-zA-Z0-9_]*\s*[+\-*\/]\s*[a-zA-Z_][a-zA-Z0-9_]*$/) ||  // expressions like depth * mag
                             codeContent.match(/^\.?\/[^\s]+$/) ||  // file paths
                             !codeContent.includes(' ') ||  // single words
                             codeContent.split(' ').length <= 5)  // short phrases
                          );
                          
                          if (shouldBeInline) {
                            return (
                              <code className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-xs font-mono border border-indigo-500/20 inline" {...props}>
                                {children}
                              </code>
                            );
                          }
                          return (
                            <code className="block p-4 rounded-lg bg-black/60 border border-white/10 text-xs font-mono overflow-x-auto text-emerald-300 my-3 whitespace-pre-wrap" {...props}>
                              {children}
                            </code>
                          );
                        },
                        // Pre wrapper - make it inline-friendly when containing short code
                        pre: ({node, children, ...props}: any) => {
                          // Check if this pre contains just a short code element
                          // If so, render without block styling to allow inline flow
                          const childContent = node?.children?.[0]?.children?.[0]?.value || '';
                          const isShortCode = !childContent.includes('\n') && childContent.length < 80;
                          
                          if (isShortCode) {
                            // Return children directly without pre wrapper for short inline code
                            return <>{children}</>;
                          }
                          return <pre className="bg-transparent p-0 m-0 overflow-visible" {...props}>{children}</pre>;
                        },
                        
                        // Tables - CRITICAL for proper table rendering
                        table: ({node, ...props}) => (
                          <div className="my-4 overflow-x-auto rounded-lg border border-white/10">
                            <table className="w-full text-sm border-collapse" {...props} />
                          </div>
                        ),
                        thead: ({node, ...props}) => <thead className="bg-white/5 border-b border-white/10" {...props} />,
                        tbody: ({node, ...props}) => <tbody className="divide-y divide-white/5" {...props} />,
                        tr: ({node, ...props}) => <tr className="hover:bg-white/[0.02] transition-colors" {...props} />,
                        th: ({node, ...props}) => <th className="px-4 py-3 text-left text-xs font-semibold text-indigo-300 uppercase tracking-wider whitespace-nowrap" {...props} />,
                        td: ({node, ...props}) => <td className="px-4 py-3 text-white/70 text-sm align-top" {...props} />,
                        
                        // Blockquotes
                        blockquote: ({node, ...props}) => (
                          <blockquote className="border-l-4 border-indigo-500/50 pl-4 py-2 my-3 bg-indigo-500/5 rounded-r-lg text-white/70 italic" {...props} />
                        ),
                        
                        // Horizontal rule
                        hr: ({node, ...props}) => <hr className="my-6 border-white/10" {...props} />,
                        
                        // Links
                        a: ({node, ...props}) => <a className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2" {...props} />,
                      }}
                    >
                      {cleanMarkdown(msg.content || '')}
                    </ReactMarkdown>
                  ) : (
                    msg.content
                  )}
                  {msg.reports && msg.reports.length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {msg.reports.map((report, idx) => {
                        // Normalize the report path: remove leading ./ and ensure it starts with /
                        const normalizedPath = report.path.replace(/^\.\//, '/');
                        return (
                          <button
                            key={idx}
                            onClick={() => { setReportModalUrl(`${window.location.origin}${normalizedPath}`); setReportModalTitle(report.name || 'Report'); }}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-500/20 hover:bg-indigo-500/30 border border-indigo-500/30 text-indigo-200 text-xs font-medium transition-all group"
                          >
                            <Sparkles className="w-3.5 h-3.5 group-hover:scale-110 transition-transform" />
                            View {report.name} Report
                          </button>
                        );
                      })}
                    </div>
                  )}
                  {msg.plots && msg.plots.length > 0 && (
                    <>
                      <div className="mt-4 space-y-3">
                        <div className="text-xs font-semibold text-white/60 mb-2">
                          📊 Generated Visualizations ({msg.plots.length})
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {msg.plots.map((plot, idx) => (
                          <button
                            key={idx}
                            onClick={() => { setReportModalUrl(`${window.location.origin}${plot.url}`); setReportModalTitle(plot.title || 'Visualization'); }}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-200 text-xs font-medium transition-all group"
                          >
                            <svg className="w-3.5 h-3.5 group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                            </svg>
                            View {plot.title}
                          </button>
                        ))}
                      </div>
                    </div>
                    </>
                  )}
                  <div className="mt-2 text-[10px] opacity-20 font-mono">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </motion.div>
            ))
          )}
          {isTyping && (
             <PipelineView
               steps={pipelineSteps}
               mode={pipelineMode}
               currentStep={currentStep}
               isActive={isTyping}
               hypotheses={pipelineHypotheses}
             />
          )}
        </div>

        {/* Input Bar */}
        <div className="p-4 md:p-8 pt-0">
          <div className="max-w-4xl mx-auto relative">
            <div className="absolute -top-10 left-4 flex gap-2">
               <input
                 ref={fileInputRef}
                 type="file"
                 accept=".csv,.parquet"
                 onChange={handleFileSelect}
                 className="hidden"
                 id="file-upload"
               />
               <label
                 htmlFor="file-upload"
                 className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/[0.03] border border-white/5 text-[10px] text-white/40 hover:text-white hover:bg-white/5 transition-all cursor-pointer"
               >
                  <Upload className="w-3 h-3" /> Upload Dataset
               </label>
               {uploadedFile && (
                 <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-500/30 text-[10px] text-indigo-200">
                   <Paperclip className="w-3 h-3" />
                   <span className="max-w-[150px] truncate">{uploadedFile.name}</span>
                   <button onClick={removeFile} className="hover:text-white transition-colors">
                     <X className="w-3 h-3" />
                   </button>
                 </div>
               )}
            </div>
            <div className="relative group">
               <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={uploadedFile ? "Describe what you want to do with this dataset..." : "Ask your agent anything or upload a dataset..."}
                className="w-full bg-[#0d0d0d] border border-white/10 rounded-2xl p-4 pr-16 text-sm min-h-[56px] max-h-48 resize-none focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-all text-white/90 placeholder:text-white/20 shadow-2xl"
              />
              <button
                onClick={handleSend}
                disabled={(!input.trim() && !uploadedFile) || isTyping}
                className={cn(
                  "absolute right-3 bottom-3 p-2.5 rounded-xl transition-all",
                  (input.trim() || uploadedFile) && !isTyping 
                    ? "bg-white text-black hover:scale-105 active:scale-95" 
                    : "bg-white/5 text-white/20 cursor-not-allowed"
                )}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            <p className="text-center mt-3 text-[10px] text-white/20 font-medium">
              Enterprise Data Agent v3.1 | Secured with end-to-end encryption
            </p>
          </div>
        </div>
      </main>
      
      {/* Assets Sidebar */}
      <AnimatePresence>
        {showAssets && (
          <motion.aside
            initial={{ x: 320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 320, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="w-[320px] border-l border-white/5 bg-[#0a0a0a]/95 backdrop-blur-xl flex flex-col"
          >
            <div className="p-4 border-b border-white/5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Package className="w-5 h-5 text-emerald-400" />
                  <h3 className="font-bold text-sm">Assets</h3>
                </div>
                <button 
                  onClick={() => setShowAssets(false)}
                  className="p-1.5 rounded-lg hover:bg-white/5 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              {/* Export to HuggingFace Button */}
              {hfConnected ? (
                <button
                  onClick={async () => {
                    setIsExporting(true);
                    setExportError(null);
                    setExportSuccess(false);
                    try {
                      // Call export API endpoint
                      const response = await fetch('/api/export/huggingface', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          user_id: user?.id,
                          session_id: activeSessionId
                        })
                      });
                      if (response.ok) {
                        setExportSuccess(true);
                        setTimeout(() => setExportSuccess(false), 5000);
                      } else {
                        const data = await response.json();
                        setExportError(data.detail || 'Export failed');
                      }
                    } catch (err) {
                      setExportError('Failed to export. Please try again.');
                    } finally {
                      setIsExporting(false);
                    }
                  }}
                  disabled={isExporting}
                  className={cn(
                    "w-full py-2.5 px-3 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all",
                    isExporting 
                      ? "bg-yellow-500/20 text-yellow-300/60 cursor-wait"
                      : exportSuccess
                        ? "bg-green-500/20 text-green-400 border border-green-500/30"
                        : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/30"
                  )}
                >
                  {isExporting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Exporting...
                    </>
                  ) : exportSuccess ? (
                    <>
                      <Check className="w-4 h-4" />
                      Exported to HuggingFace!
                    </>
                  ) : (
                    <>
                      <HuggingFaceLogo className="w-4 h-4" />
                      Export to HuggingFace
                    </>
                  )}
                </button>
              ) : (
                <button
                  onClick={() => setShowSettings(true)}
                  className="w-full py-2.5 px-3 rounded-lg text-sm font-medium flex items-center justify-center gap-2 bg-white/5 text-white/50 border border-white/10 hover:bg-white/10 hover:text-white/70 transition-all"
                >
                  <HuggingFaceLogo className="w-4 h-4" />
                  Export to HuggingFace
                </button>
              )}
              
              {exportError && (
                <p className="text-xs text-red-400 mt-2 text-center">{exportError}</p>
              )}
            </div>
            
            {/* Warning Banner */}
            {!hfConnected && (
              <div className="px-4 py-3 bg-amber-500/10 border-y border-amber-500/20">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs text-amber-300 font-medium">Assets are temporary</p>
                    <p className="text-[10px] text-amber-300/60 mt-0.5">
                      Connect HuggingFace in Settings to permanently save your visualizations, models, and datasets.
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
              {(() => {
                const allPlots: Array<{title: string, url: string, type?: string}> = [];
                const allReports: Array<{name: string, path: string}> = [];
                const allDataFiles: string[] = [];
                const baselineModels = ['xgboost', 'random_forest', 'lightgbm', 'ridge', 'lasso'];
                const foundModels = new Set<string>();
                
                activeSession.messages.forEach(msg => {
                  if (msg.plots) allPlots.push(...msg.plots);
                  if (msg.reports) allReports.push(...msg.reports);
                  
                  baselineModels.forEach(model => {
                    if (msg.content.toLowerCase().includes(model)) {
                      const displayName = model.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                      foundModels.add(displayName);
                    }
                  });
                  
                  if (msg.content.includes('Cleaned') || msg.content.includes('encoded')) {
                    allDataFiles.push('Cleaned & Encoded Dataset');
                  }
                });
                
                const uniqueDataFiles = [...new Set(allDataFiles)];
                const uniqueModels = Array.from(foundModels);
                
                return (
                  <>
                    {/* Plots Section FIRST */}
                    {allPlots.length > 0 && (
                      <div>
                        <div className="flex items-center gap-2 mb-3">
                          <BarChart3 className="w-4 h-4 text-emerald-400" />
                          <h4 className="text-xs font-bold uppercase tracking-wider text-white/60">Visualizations ({allPlots.length})</h4>
                        </div>
                        <div className="space-y-2">
                          {allPlots.map((plot, idx) => {
                            // Ensure URL is properly formatted
                            let plotUrl = plot.url;
                            if (plotUrl && plotUrl.startsWith('./outputs/')) {
                              plotUrl = plotUrl.replace('./outputs/', '/outputs/');
                            } else if (plotUrl && !plotUrl.startsWith('/outputs/')) {
                              plotUrl = `/outputs/${plotUrl.replace(/^outputs\//, '')}`;
                            }
                            
                            return (
                              <button
                                key={idx}
                                onClick={() => { setReportModalUrl(plotUrl || plot.url); setReportModalTitle(plot.title || 'Visualization'); }}
                                className="w-full p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-emerald-500/10 hover:border-emerald-500/30 transition-all text-left group"
                              >
                                <div className="flex items-center justify-between">
                                  <span className="text-sm text-white/80 truncate flex-1">{plot.title}</span>
                                  <ChevronRight className="w-4 h-4 text-white/40 group-hover:text-emerald-400 transition-all" />
                                </div>
                                <span className="text-xs text-white/40 mt-1 block">{plot.type || 'interactive'}</span>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    
                    {/* Data Files Section */}
                    {uniqueDataFiles.length > 0 && (
                      <div>
                        <div className="flex items-center gap-2 mb-3">
                          <FileText className="w-4 h-4 text-blue-400" />
                          <h4 className="text-xs font-bold uppercase tracking-wider text-white/60">Data Files ({uniqueDataFiles.length})</h4>
                        </div>
                        <div className="space-y-2">
                          {uniqueDataFiles.map((file, idx) => {
                            // Extract filename from path
                            const fileName = file.split('/').pop() || file;
                            // Create proper download URL
                            let downloadUrl = file;
                            if (downloadUrl.startsWith('./outputs/')) {
                              downloadUrl = downloadUrl.replace('./outputs/', '/outputs/');
                            } else if (!downloadUrl.startsWith('/outputs/')) {
                              downloadUrl = `/outputs/${file.replace(/^outputs\//, '')}`;
                            }
                            
                            return (
                              <a
                                key={idx}
                                href={downloadUrl}
                                download={fileName}
                                className="block w-full p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-blue-500/10 hover:border-blue-500/30 transition-all group"
                              >
                                <div className="flex items-center justify-between">
                                  <span className="text-sm text-white/80 truncate flex-1">{fileName}</span>
                                  <ChevronRight className="w-4 h-4 text-white/40 group-hover:text-blue-400 transition-all" />
                                </div>
                                <span className="text-xs text-white/40 mt-1 block">Click to download</span>
                              </a>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    
                    {/* Models Section */}
                    {uniqueModels.length > 0 && (
                      <div>
                        <div className="flex items-center gap-2 mb-3">
                          <FileText className="w-4 h-4 text-purple-400" />
                          <h4 className="text-xs font-bold uppercase tracking-wider text-white/60">Models ({uniqueModels.length})</h4>
                        </div>
                        <div className="space-y-2">
                          {uniqueModels.map((model, idx) => {
                            // Find the model file path from workflow history
                            let modelPath = '';
                            activeSession.messages.forEach(msg => {
                              if (msg.role === 'assistant' && msg.content.includes(model)) {
                                // Try to extract model path (typically in ./outputs/models/)
                                const match = msg.content.match(/\.\/outputs\/models\/[^\s)]+\.pkl/);
                                if (match) modelPath = match[0].replace('./', '/');
                              }
                            });
                            
                            // Fallback: construct typical path
                            if (!modelPath) {
                              modelPath = `/outputs/models/${model.toLowerCase().replace(/\s+/g, '_')}_model.pkl`;
                            }
                            
                            return (
                              <button
                                key={idx}
                                onClick={() => {
                                  // Trigger download
                                  const link = document.createElement('a');
                                  link.href = modelPath;
                                  link.download = `${model.toLowerCase().replace(/\s+/g, '_')}_model.pkl`;
                                  link.click();
                                }}
                                className="w-full p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-purple-500/10 hover:border-purple-500/30 transition-all text-left group"
                              >
                                <div className="flex items-center justify-between">
                                  <span className="text-sm text-white/80 truncate flex-1">{model}</span>
                                  <ChevronRight className="w-4 h-4 text-white/40 group-hover:text-purple-400 transition-all" />
                                </div>
                                <span className="text-xs text-white/40 mt-1 block">Click to download</span>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    
                    {/* Reports Section */}
                    {allReports.length > 0 && (
                      <div>
                        <div className="flex items-center gap-2 mb-3">
                          <FileText className="w-4 h-4 text-purple-400" />
                          <h4 className="text-xs font-bold uppercase tracking-wider text-white/60">Reports ({allReports.length})</h4>
                        </div>
                        <div className="space-y-2">
                          {allReports.map((report, idx) => (
                            <button
                              key={idx}
                              onClick={() => { setReportModalUrl(report.path); setReportModalTitle(report.name || 'Report'); }}
                              className="w-full p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-purple-500/10 hover:border-purple-500/30 transition-all text-left group"
                            >
                              <div className="flex items-center justify-between">
                                <span className="text-sm text-white/80 truncate flex-1">{report.name}</span>
                                <ChevronRight className="w-4 h-4 text-white/40 group-hover:text-purple-400 transition-all" />
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Empty State */}
                    {allPlots.length === 0 && allReports.length === 0 && uniqueModels.length === 0 && (
                      <div className="flex flex-col items-center justify-center h-full text-center p-8">
                        <Package className="w-12 h-12 text-white/10 mb-3" />
                        <p className="text-sm text-white/40 mb-1">No assets yet</p>
                        <p className="text-xs text-white/30">Upload a dataset to generate visualizations and models</p>
                      </div>
                    )}
                  </>
                );
              })()}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
      
      {/* Report Modal */}
      <AnimatePresence>
        {reportModalUrl && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => { setReportModalUrl(null); setReportModalTitle('Visualization'); }}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-[#0a0a0a] border border-white/10 rounded-2xl w-full max-w-7xl h-[90vh] flex flex-col overflow-hidden shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between p-4 border-b border-white/5">
                <h3 className="text-lg font-semibold text-white">{reportModalTitle}</h3>
                <button
                  onClick={() => { setReportModalUrl(null); setReportModalTitle('Visualization'); }}
                  className="p-2 rounded-lg hover:bg-white/5 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <iframe
                src={reportModalUrl}
                className="flex-1 w-full bg-white"
                title="Report Viewer"
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.1);
        }
      `}</style>
      
      {/* Settings Modal */}
      <SettingsModal 
        isOpen={showSettings} 
        onClose={() => {
          setShowSettings(false);
          // Refresh HF connection status when settings modal closes
          // Reset the ref to allow a fresh check
          if (user?.id) {
            hfStatusCheckedRef.current = false;
            getHuggingFaceStatus(user.id).then(status => setHfConnected(status.connected));
          }
        }} 
      />
    </div>
  );
};
