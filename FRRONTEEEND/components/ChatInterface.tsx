
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Plus, Search, Settings, MoreHorizontal, User, Bot, ArrowLeft, Paperclip, Sparkles, Trash2, X, Upload } from 'lucide-react';
import { cn } from '../lib/utils';
import { Logo } from './Logo';
import ReactMarkdown from 'react-markdown';

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
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: Date;
}

export const ChatInterface: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  const [sessions, setSessions] = useState<ChatSession[]>([
    {
      id: '1',
      title: 'ML Model Analysis',
      messages: [],
      updatedAt: new Date(),
    }
  ]);
  const [activeSessionId, setActiveSessionId] = useState('1');
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [reportModalUrl, setReportModalUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  
  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activeSession.messages, isTyping]);

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
    setIsTyping(true);

    try {
      // Use the current origin if running on same server, otherwise use env variable
      const API_URL = window.location.origin;
      console.log('API URL:', API_URL);
      
      let response;
      
      // Check if there's a recent file analysis in the conversation
      const recentFileMessage = newMessages.slice(-5).find(m => m.file || m.content.includes('Uploaded:'));
      const hasRecentFile = recentFileMessage && !uploadedFile;
      
      if (uploadedFile || hasRecentFile) {
        // Use /run endpoint for file analysis or follow-up questions about uploaded data
        const formData = new FormData();
        
        if (uploadedFile) {
          formData.append('file', uploadedFile);
          formData.append('task_description', input || 'Analyze this dataset and provide insights');
        } else if (hasRecentFile) {
          // For follow-up questions, extract the filename from recent context
          const fileNameMatch = recentFileMessage?.content.match(/Uploaded: (.+)/);
          const fileName = fileNameMatch ? fileNameMatch[1] : 'dataset.csv';
          
          // Send follow-up request as a new task description
          formData.append('task_description', input);
          formData.append('session_id', activeSessionId);
          
          // Note: Backend needs to support session-based file context
          // For now, just send the task which should work with session memory
        }
        
        formData.append('use_cache', 'true');
        formData.append('max_iterations', '20');
        
        response = await fetch(`${API_URL}/run`, {
          method: 'POST',
          body: formData
        });
        
        setUploadedFile(null);
      } else {
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
      
      let assistantContent = '';
      let reports: Array<{name: string, path: string}> = [];
      
      // Check for reports in any /run endpoint response (not just when file is uploaded)
      if (data.result) {
        const result = data.result;
        assistantContent = `✅ Analysis Complete!\n\n`;
        
        // Extract report paths from workflow history
        if (result.workflow_history) {
          const reportTools = ['generate_ydata_profiling_report'];
          result.workflow_history.forEach((step: any) => {
            if (reportTools.includes(step.tool)) {
              // Check multiple possible locations for the report path
              const reportPath = step.result?.output_path || step.result?.report_path || step.arguments?.output_path;
              
              if (reportPath && (step.result?.success !== false)) {
                reports.push({
                  name: step.tool.replace('generate_', '').replace(/_/g, ' ').replace('report', '').trim(),
                  path: reportPath
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
        assistantContent = data.message;
      } else {
        throw new Error('Invalid response from API');
      }
      
      updateSession(activeSessionId, [...newMessages, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: assistantContent,
        timestamp: new Date(),
        reports: reports.length > 0 ? reports : undefined
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
    } finally {
      setIsTyping(false);
    }
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
    const newId = Date.now().toString();
    const newSession: ChatSession = {
      id: newId,
      title: 'New Chat',
      messages: [],
      updatedAt: new Date()
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newId);
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

          <div className="mt-auto pt-4 border-t border-white/5 flex items-center justify-between px-2">
            <button onClick={onBack} className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/40 hover:text-white">
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div className="flex gap-2">
              <button className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/40 hover:text-white">
                <Settings className="w-5 h-5" />
              </button>
              <button className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/40 hover:text-white">
                <User className="w-5 h-5" />
              </button>
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
                      className="prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-pre:bg-black/40 prose-pre:border prose-pre:border-white/10 prose-headings:text-white prose-strong:text-white prose-li:text-white/80"
                      components={{
                        p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                        ul: ({node, ...props}) => <ul className="mb-3 space-y-1" {...props} />,
                        ol: ({node, ...props}) => <ol className="mb-3 space-y-1" {...props} />,
                        li: ({node, ...props}) => <li className="ml-4" {...props} />,
                        strong: ({node, ...props}) => <strong className="font-semibold text-white" {...props} />,
                        code: ({node, inline, ...props}: any) => 
                          inline ? 
                            <code className="px-1.5 py-0.5 rounded bg-white/10 text-indigo-300 text-xs font-mono" {...props} /> :
                            <code className="block p-3 rounded-lg bg-black/40 border border-white/10 text-xs font-mono overflow-x-auto" {...props} />
                      }}
                    >
                      {msg.content || ''}
                    </ReactMarkdown>
                  ) : (
                    msg.content || (msg.role === 'assistant' && isTyping && "...")
                  )}
                  {msg.reports && msg.reports.length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {msg.reports.map((report, idx) => {
                        // Normalize the report path: remove leading ./ and ensure it starts with /
                        const normalizedPath = report.path.replace(/^\.\//, '/');
                        return (
                          <button
                            key={idx}
                            onClick={() => setReportModalUrl(`${window.location.origin}${normalizedPath}`)}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-500/20 hover:bg-indigo-500/30 border border-indigo-500/30 text-indigo-200 text-xs font-medium transition-all group"
                          >
                            <Sparkles className="w-3.5 h-3.5 group-hover:scale-110 transition-transform" />
                            View {report.name} Report
                          </button>
                        );
                      })}
                    </div>
                  )}
                  <div className="mt-2 text-[10px] opacity-20 font-mono">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </motion.div>
            ))
          )}
          {isTyping && activeSession.messages[activeSession.messages.length - 1]?.role === 'user' && (
             <div className="flex gap-4">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-white/5 border border-white/10">
                  <Bot className="w-4 h-4 text-indigo-400" />
                </div>
                <div className="bg-white/[0.03] p-4 rounded-2xl border border-white/5">
                  <div className="flex gap-1">
                    <span className="w-1.5 h-1.5 bg-white/20 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                    <span className="w-1.5 h-1.5 bg-white/20 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                    <span className="w-1.5 h-1.5 bg-white/20 rounded-full animate-bounce"></span>
                  </div>
                </div>
             </div>
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
      
      {/* Report Modal */}
      <AnimatePresence>
        {reportModalUrl && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setReportModalUrl(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-[#0a0a0a] border border-white/10 rounded-2xl w-full max-w-7xl h-[90vh] flex flex-col overflow-hidden shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between p-4 border-b border-white/5">
                <h3 className="text-lg font-semibold text-white">Data Profiling Report</h3>
                <button
                  onClick={() => setReportModalUrl(null)}
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
    </div>
  );
};
