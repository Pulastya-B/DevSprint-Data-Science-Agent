import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Settings, Eye, EyeOff, Check, Loader2, ExternalLink, AlertTriangle } from 'lucide-react';
import { useAuth } from '../lib/AuthContext';
import { updateHuggingFaceToken, getHuggingFaceStatus, clearHfStatusCache } from '../lib/supabase';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// HuggingFace logo SVG component
const HuggingFaceLogo = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 120 120" fill="currentColor">
    <path d="M60 0C26.863 0 0 26.863 0 60s26.863 60 60 60 60-26.863 60-60S93.137 0 60 0zm0 10c27.614 0 50 22.386 50 50s-22.386 50-50 50S10 87.614 10 60 32.386 10 60 10z"/>
    <circle cx="40" cy="50" r="8"/>
    <circle cx="80" cy="50" r="8"/>
    <path d="M40 75c0 11.046 8.954 20 20 20s20-8.954 20-20H40z"/>
  </svg>
);

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<'huggingface' | 'account'>('huggingface');
  
  // HuggingFace settings
  const [hfToken, setHfToken] = useState('');
  const [hfUsername, setHfUsername] = useState('');
  const [showToken, setShowToken] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tokenMasked, setTokenMasked] = useState<string | null>(null);

  // Load current HuggingFace status on mount
  useEffect(() => {
    const loadHfStatus = async () => {
      if (user?.id) {
        // Clear cache to ensure fresh data when modal opens
        clearHfStatusCache();
        const status = await getHuggingFaceStatus(user.id);
        setIsConnected(status.connected);
        setHfUsername(status.username || '');
        setTokenMasked(status.tokenMasked || null);
      }
    };
    
    if (isOpen) {
      loadHfStatus();
    }
  }, [user, isOpen]);

  // Validate HuggingFace token
  const validateToken = async (token: string): Promise<{ valid: boolean; username?: string; error?: string }> => {
    try {
      const response = await fetch('https://huggingface.co/api/whoami-v2', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        return { valid: true, username: data.name };
      } else if (response.status === 401) {
        return { valid: false, error: 'Invalid token. Please check your token and try again.' };
      } else {
        return { valid: false, error: 'Could not validate token. Please try again.' };
      }
    } catch (err) {
      return { valid: false, error: 'Network error. Please check your connection.' };
    }
  };

  // Save HuggingFace settings
  const handleSaveHuggingFace = async () => {
    if (!user?.id) {
      setError('You must be logged in to save settings');
      return;
    }

    if (!hfToken.trim()) {
      setError('Please enter a HuggingFace token');
      return;
    }

    setIsSaving(true);
    setError(null);
    setSaveSuccess(false);

    try {
      // Validate the token first
      setIsValidating(true);
      const validation = await validateToken(hfToken);
      setIsValidating(false);

      if (!validation.valid) {
        setError(validation.error || 'Invalid token');
        setIsSaving(false);
        return;
      }

      // Save to Supabase
      console.log('[Settings] Saving HF token to Supabase for user:', user.id);
      const result = await updateHuggingFaceToken(user.id, hfToken, validation.username);
      console.log('[Settings] Save result:', result);
      
      if (result) {
        setIsConnected(true);
        setHfUsername(validation.username || '');
        setTokenMasked(`hf_****${hfToken.slice(-4)}`);
        setSaveSuccess(true);
        setHfToken(''); // Clear the input after saving
        
        // Hide success message after 3 seconds
        setTimeout(() => setSaveSuccess(false), 3000);
      } else {
        setError('Failed to save token. Please ensure your profile exists and try again.');
      }
    } catch (err: any) {
      console.error('[Settings] Save error:', err);
      setError(err?.message || 'An error occurred. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  // Disconnect HuggingFace
  const handleDisconnect = async () => {
    if (!user?.id) return;

    setIsSaving(true);
    try {
      const result = await updateHuggingFaceToken(user.id, '', '');
      if (result) {
        setIsConnected(false);
        setHfUsername('');
        setTokenMasked(null);
        setHfToken('');
      }
    } catch (err) {
      setError('Failed to disconnect. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="bg-[#0a0a0a] border border-white/10 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-white/5">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/5 rounded-xl flex items-center justify-center">
                <Settings className="w-5 h-5 text-white/60" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Settings</h2>
                <p className="text-xs text-white/40">Configure your integrations</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/5 transition-colors"
            >
              <X className="w-5 h-5 text-white/60" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-white/5">
            <button
              onClick={() => setActiveTab('huggingface')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                activeTab === 'huggingface'
                  ? 'text-yellow-400 border-b-2 border-yellow-400 bg-yellow-400/5'
                  : 'text-white/50 hover:text-white/70'
              }`}
            >
              <HuggingFaceLogo className="w-4 h-4" />
              HuggingFace
            </button>
            <button
              onClick={() => setActiveTab('account')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'account'
                  ? 'text-indigo-400 border-b-2 border-indigo-400 bg-indigo-400/5'
                  : 'text-white/50 hover:text-white/70'
              }`}
            >
              Account
            </button>
          </div>

          {/* Content */}
          <div className="p-6">
            {activeTab === 'huggingface' && (
              <div className="space-y-5">
                {/* Status Card */}
                <div className={`p-4 rounded-xl border ${
                  isConnected 
                    ? 'bg-green-500/10 border-green-500/30' 
                    : 'bg-yellow-500/10 border-yellow-500/30'
                }`}>
                  <div className="flex items-start gap-3">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                      isConnected ? 'bg-green-500/20' : 'bg-yellow-500/20'
                    }`}>
                      {isConnected ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <AlertTriangle className="w-4 h-4 text-yellow-400" />
                      )}
                    </div>
                    <div className="flex-1">
                      <h4 className={`text-sm font-semibold ${
                        isConnected ? 'text-green-300' : 'text-yellow-300'
                      }`}>
                        {isConnected ? 'Connected to HuggingFace' : 'Not Connected'}
                      </h4>
                      {isConnected ? (
                        <p className="text-xs text-white/50 mt-1">
                          Connected as <span className="text-white/80 font-medium">{hfUsername}</span>
                          <br />
                          Token: <span className="font-mono text-white/60">{tokenMasked}</span>
                        </p>
                      ) : (
                        <p className="text-xs text-white/50 mt-1">
                          Connect your HuggingFace account to save and export your assets
                        </p>
                      )}
                    </div>
                    {isConnected && (
                      <button
                        onClick={handleDisconnect}
                        disabled={isSaving}
                        className="text-xs text-red-400 hover:text-red-300 transition-colors"
                      >
                        Disconnect
                      </button>
                    )}
                  </div>
                </div>

                {/* Why Connect Section */}
                {!isConnected && (
                  <div className="p-4 bg-white/5 rounded-xl border border-white/10">
                    <h4 className="text-sm font-semibold text-white mb-2">🚀 Why connect?</h4>
                    <ul className="text-xs text-white/60 space-y-1.5">
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span><strong className="text-white/80">Permanent storage</strong> - Your datasets, models & plots saved forever</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span><strong className="text-white/80">One-click deploy</strong> - Turn models into APIs instantly</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span><strong className="text-white/80">Version control</strong> - Git-based versioning for free</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span><strong className="text-white/80">Your data</strong> - Everything stored in YOUR HuggingFace account</span>
                      </li>
                    </ul>
                  </div>
                )}

                {/* Token Input */}
                <div className="space-y-2">
                  <label className="text-sm text-white/70 flex items-center gap-2">
                    {isConnected ? 'Update Token' : 'Access Token'}
                    <span className="text-xs text-white/40">(Write permission required)</span>
                  </label>
                  <div className="relative">
                    <input
                      type={showToken ? 'text' : 'password'}
                      value={hfToken}
                      onChange={(e) => setHfToken(e.target.value)}
                      placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 pr-10 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-yellow-500/50 focus:ring-1 focus:ring-yellow-500/20 font-mono"
                    />
                    <button
                      type="button"
                      onClick={() => setShowToken(!showToken)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/60"
                    >
                      {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <p className="text-xs text-white/40">
                    Get your token from{' '}
                    <a
                      href="https://huggingface.co/settings/tokens"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-yellow-400 hover:text-yellow-300 inline-flex items-center gap-1"
                    >
                      huggingface.co/settings/tokens
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  </p>
                </div>

                {/* Error/Success Messages */}
                {error && (
                  <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
                    {error}
                  </div>
                )}
                {saveSuccess && (
                  <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg text-green-400 text-sm flex items-center gap-2">
                    <Check className="w-4 h-4" />
                    Successfully connected to HuggingFace!
                  </div>
                )}

                {/* Save Button */}
                <button
                  onClick={handleSaveHuggingFace}
                  disabled={isSaving || !hfToken.trim()}
                  className={`w-full py-3 rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2 ${
                    isSaving || !hfToken.trim()
                      ? 'bg-white/5 text-white/30 cursor-not-allowed'
                      : 'bg-yellow-500 text-black hover:bg-yellow-400'
                  }`}
                >
                  {isSaving ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      {isValidating ? 'Validating...' : 'Saving...'}
                    </>
                  ) : (
                    <>
                      <HuggingFaceLogo className="w-4 h-4" />
                      {isConnected ? 'Update Connection' : 'Connect HuggingFace'}
                    </>
                  )}
                </button>

                {/* Security Note */}
                <p className="text-xs text-white/30 text-center">
                  🔒 Your token is encrypted and stored securely. We only use it to save files to your account.
                </p>
              </div>
            )}

            {activeTab === 'account' && (
              <div className="space-y-4">
                <div className="p-4 bg-white/5 rounded-xl border border-white/10">
                  <h4 className="text-sm font-semibold text-white mb-2">Account Information</h4>
                  {user ? (
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-white/50">Email</span>
                        <span className="text-white/80">{user.email}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-white/50">User ID</span>
                        <span className="text-white/60 font-mono text-xs">{user.id.slice(0, 8)}...</span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-white/50">Not signed in</p>
                  )}
                </div>
                
                <div className="p-4 bg-white/5 rounded-xl border border-white/10">
                  <h4 className="text-sm font-semibold text-white mb-2">Data & Privacy</h4>
                  <p className="text-xs text-white/50">
                    Your chat history is stored locally in your browser. 
                    Connect HuggingFace to permanently save your generated assets.
                  </p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default SettingsModal;
