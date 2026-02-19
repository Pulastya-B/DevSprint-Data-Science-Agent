import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, Zap, BarChart3, CheckCircle2, XCircle, 
  Loader2, ChevronDown, ChevronUp, Lightbulb,
  Search, FlaskConical, FileText, Target, ArrowRight
} from 'lucide-react';
import { cn } from '../lib/utils';

// ─── Types ───────────────────────────────────────────────────

export interface PipelineStep {
  id: string;
  type: 'intent' | 'hypothesis' | 'reason' | 'act' | 'evaluate' | 'finding' | 'synthesize';
  status: 'pending' | 'active' | 'completed' | 'failed';
  title: string;
  subtitle?: string;
  detail?: string;       // Extended info (shown on expand)
  confidence?: number;   // 0-1
  timestamp?: Date;
  tool?: string;
  iteration?: number;
}

interface PipelineViewProps {
  steps: PipelineStep[];
  mode: string | null;           // "direct" | "investigative" | "exploratory" | null
  currentStep: string;           // Existing currentStep string from ChatInterface
  isActive: boolean;             // Whether analysis is running
  hypotheses?: string[];
  className?: string;
}

// ─── Icons per step type ─────────────────────────────────────

const stepIcons: Record<PipelineStep['type'], React.ElementType> = {
  intent: Target,
  hypothesis: Lightbulb,
  reason: Brain,
  act: Zap,
  evaluate: Search,
  finding: FlaskConical,
  synthesize: FileText,
};

const stepColors: Record<PipelineStep['type'], string> = {
  intent: 'text-violet-400 bg-violet-500/10 border-violet-500/20',
  hypothesis: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  reason: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
  act: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  evaluate: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  finding: 'text-pink-400 bg-pink-500/10 border-pink-500/20',
  synthesize: 'text-orange-400 bg-orange-500/10 border-orange-500/20',
};

const statusDotColors: Record<PipelineStep['status'], string> = {
  pending: 'bg-white/20',
  active: 'bg-emerald-500',
  completed: 'bg-emerald-500',
  failed: 'bg-red-500',
};

// ─── Confidence Bar ──────────────────────────────────────────

const ConfidenceBar: React.FC<{ value: number }> = ({ value }) => (
  <div className="flex items-center gap-2 mt-1">
    <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
      <motion.div
        className={cn(
          "h-full rounded-full",
          value >= 0.7 ? "bg-emerald-500" : value >= 0.4 ? "bg-amber-500" : "bg-red-400"
        )}
        initial={{ width: 0 }}
        animate={{ width: `${Math.round(value * 100)}%` }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      />
    </div>
    <span className="text-[10px] font-mono text-white/30 w-8 text-right">
      {Math.round(value * 100)}%
    </span>
  </div>
);

// ─── Mode Badge ──────────────────────────────────────────────

const ModeBadge: React.FC<{ mode: string }> = ({ mode }) => {
  const config: Record<string, { label: string; color: string; icon: React.ElementType }> = {
    direct: { label: 'Direct', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20', icon: Zap },
    investigative: { label: 'Investigative', color: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20', icon: Search },
    exploratory: { label: 'Exploratory', color: 'bg-violet-500/10 text-violet-400 border-violet-500/20', icon: FlaskConical },
  };
  const { label, color, icon: Icon } = config[mode] || config.direct;

  return (
    <span className={cn("inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] font-medium rounded-full border", color)}>
      <Icon className="w-3 h-3" />
      {label} Mode
    </span>
  );
};

// ─── Single Step Row ─────────────────────────────────────────

const StepRow: React.FC<{ step: PipelineStep; isLast: boolean }> = ({ step, isLast }) => {
  const [expanded, setExpanded] = React.useState(false);
  const Icon = stepIcons[step.type] || Zap;
  const colorClass = stepColors[step.type] || stepColors.act;
  const isActive = step.status === 'active';
  const isCompleted = step.status === 'completed';
  const isFailed = step.status === 'failed';

  return (
    <div className="relative">
      {/* Connector line */}
      {!isLast && (
        <div className={cn(
          "absolute left-4 top-10 w-px h-[calc(100%-16px)]",
          isCompleted ? "bg-emerald-500/30" : "bg-white/5"
        )} />
      )}

      <motion.div
        initial={{ opacity: 0, x: -12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
        className={cn(
          "relative flex items-start gap-3 p-2 rounded-lg cursor-pointer transition-colors",
          isActive && "bg-white/[0.03]",
          expanded && "bg-white/[0.02]"
        )}
        onClick={() => step.detail && setExpanded(!expanded)}
      >
        {/* Icon circle */}
        <div className={cn(
          "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border",
          colorClass,
          isActive && "animate-pulse"
        )}>
          {isActive ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : isCompleted ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          ) : isFailed ? (
            <XCircle className="w-4 h-4 text-red-400" />
          ) : (
            <Icon className="w-4 h-4" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={cn(
              "text-xs font-medium truncate",
              isActive ? "text-white" : isCompleted ? "text-white/70" : "text-white/40"
            )}>
              {step.title}
            </span>
            {step.iteration && (
              <span className="text-[10px] font-mono text-white/20 shrink-0">
                #{step.iteration}
              </span>
            )}
            {step.detail && (
              expanded 
                ? <ChevronUp className="w-3 h-3 text-white/20 shrink-0" /> 
                : <ChevronDown className="w-3 h-3 text-white/20 shrink-0" />
            )}
          </div>
          
          {step.subtitle && (
            <p className="text-[11px] text-white/30 mt-0.5 truncate">
              {step.subtitle}
            </p>
          )}
          
          {step.confidence !== undefined && step.confidence > 0 && (
            <ConfidenceBar value={step.confidence} />
          )}
        </div>

        {/* Status dot */}
        <div className={cn(
          "w-2 h-2 rounded-full shrink-0 mt-2",
          statusDotColors[step.status]
        )} />
      </motion.div>

      {/* Expanded detail */}
      <AnimatePresence>
        {expanded && step.detail && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="ml-11 mr-2 mb-2 p-2 rounded-lg bg-white/[0.02] border border-white/5">
              <p className="text-[11px] text-white/40 leading-relaxed whitespace-pre-wrap">
                {step.detail}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// ─── Hypotheses Panel ────────────────────────────────────────

const HypothesesPanel: React.FC<{ hypotheses: string[] }> = ({ hypotheses }) => {
  const [collapsed, setCollapsed] = React.useState(false);

  if (!hypotheses.length) return null;

  return (
    <div className="mb-3">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center gap-1.5 text-[10px] font-medium text-amber-400/70 hover:text-amber-400 transition-colors mb-1.5"
      >
        <Lightbulb className="w-3 h-3" />
        <span>{hypotheses.length} Hypotheses</span>
        {collapsed ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
      </button>
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="space-y-1 ml-4">
              {hypotheses.map((h, i) => (
                <div key={i} className="flex items-start gap-1.5">
                  <ArrowRight className="w-3 h-3 text-amber-500/30 mt-0.5 shrink-0" />
                  <span className="text-[11px] text-white/30">{h}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// ─── Main Pipeline View ──────────────────────────────────────

export const PipelineView: React.FC<PipelineViewProps> = ({
  steps,
  mode,
  currentStep,
  isActive,
  hypotheses = [],
  className
}) => {
  // If no steps yet and not in reasoning mode, show the simple fallback
  if (!steps.length && !mode) {
    return (
      <div className={cn("flex gap-4", className)}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-white/5 border border-white/10">
          <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
        </div>
        <div className="bg-white/[0.03] p-4 rounded-2xl border border-white/5">
          <div className="flex items-center gap-3">
            <div className="flex gap-1">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce" />
            </div>
            <span className="text-sm text-white/60">
              {currentStep || '🔧 Starting analysis...'}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Count completed steps
  const completedCount = steps.filter(s => s.status === 'completed').length;
  const totalCount = steps.length;
  const progressPct = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

  return (
    <div className={cn("flex gap-4", className)}>
      {/* Bot avatar */}
      <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-white/5 border border-white/10">
        <Brain className="w-4 h-4 text-cyan-400" />
      </div>

      {/* Pipeline card */}
      <div className="flex-1 bg-white/[0.03] p-4 rounded-2xl border border-white/5 max-w-lg">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-white/80">Reasoning Pipeline</span>
            {mode && <ModeBadge mode={mode} />}
          </div>
          {isActive && (
            <div className="flex items-center gap-1.5 text-[10px] text-emerald-400">
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>Running</span>
            </div>
          )}
        </div>

        {/* Progress bar */}
        <div className="h-1 bg-white/5 rounded-full overflow-hidden mb-3">
          <motion.div
            className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progressPct}%` }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          />
        </div>

        {/* Hypotheses (exploratory mode) */}
        {hypotheses.length > 0 && <HypothesesPanel hypotheses={hypotheses} />}

        {/* Steps timeline */}
        <div className="space-y-0.5 max-h-[320px] overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/5">
          {steps.map((step, i) => (
            <StepRow key={step.id} step={step} isLast={i === steps.length - 1} />
          ))}
        </div>

        {/* Footer summary */}
        {!isActive && completedCount > 0 && (
          <div className="mt-3 pt-2 border-t border-white/5 flex items-center justify-between">
            <span className="text-[10px] text-white/20">
              {completedCount} step{completedCount !== 1 ? 's' : ''} completed
            </span>
            <span className="text-[10px] text-white/20 font-mono">
              {steps.filter(s => s.type === 'finding').length} finding{steps.filter(s => s.type === 'finding').length !== 1 ? 's' : ''}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PipelineView;
