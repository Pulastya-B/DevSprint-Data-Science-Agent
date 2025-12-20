
import React from 'react';
import { motion } from 'framer-motion';
import { Database, Wrench, Cpu, Brain, LineChart, Server } from 'lucide-react';
import { cn } from '../lib/utils';

const capabilities = [
  {
    title: "Autonomous ML Pipelines",
    description: "End-to-end automation from profiling to deployment without manual coding.",
    icon: Database,
    color: "from-blue-500/20 to-cyan-500/20",
    hover: "hover:bg-blue-500/10 hover:border-blue-500/30 hover:shadow-[0_0_30px_-10px_rgba(59,130,246,0.2)]"
  },
  {
    title: "82+ Specialized Tools",
    description: "An extensive arsenal for cleaning, statistical testing, and predictive modeling.",
    icon: Wrench,
    color: "from-purple-500/20 to-pink-500/20",
    hover: "hover:bg-pink-500/10 hover:border-pink-500/30 hover:shadow-[0_0_30px_-10px_rgba(236,72,153,0.2)]"
  },
  {
    title: "Dual LLM Intelligence",
    description: "Orchestrated by Groq (for speed) and Gemini (for deep reasoning).",
    icon: Brain,
    color: "from-orange-500/20 to-amber-500/20",
    hover: "hover:bg-amber-500/10 hover:border-amber-500/30 hover:shadow-[0_0_30px_-10px_rgba(245,158,11,0.2)]"
  },
  {
    title: "Session Memory",
    description: "Maintains context across complex workflows, allowing for iterative refinement.",
    icon: Cpu,
    color: "from-emerald-500/20 to-teal-500/20",
    hover: "hover:bg-emerald-500/10 hover:border-emerald-500/30 hover:shadow-[0_0_30px_-10px_rgba(16,185,129,0.2)]"
  },
  {
    title: "Visual Insights",
    description: "Automatic generation of publication-quality charts and explainability reports.",
    icon: LineChart,
    color: "from-indigo-500/20 to-blue-500/20",
    hover: "hover:bg-indigo-500/10 hover:border-indigo-500/30 hover:shadow-[0_0_30px_-10px_rgba(99,102,241,0.2)]"
  },
  {
    title: "Cloud Run Ready",
    description: "Deploy your optimized models directly to production-grade cloud environments.",
    icon: Server,
    color: "from-rose-500/20 to-red-500/20",
    hover: "hover:bg-rose-500/10 hover:border-rose-500/30 hover:shadow-[0_0_30px_-10px_rgba(244,63,94,0.2)]"
  }
];

const KeyCapabilities = () => {
  return (
    <section id="features" className="py-24 bg-[#030303]">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-4 tracking-tight">Powerful Orchestration</h2>
          <p className="text-white/40 text-xl font-medium">Not just a chatbot, but a true system of intelligence.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {capabilities.map((cap, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              whileHover={{ scale: 1.02, y: -5 }}
              className={cn(
                "group p-8 rounded-2xl bg-white/[0.02] border border-white/[0.08] transition-all duration-300 cursor-default",
                cap.hover
              )}
            >
              <div className={cn(
                "w-12 h-12 rounded-lg bg-gradient-to-br flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300",
                cap.color
              )}>
                <cap.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3 tracking-tight">{cap.title}</h3>
              <p className="text-white/50 leading-relaxed font-medium">{cap.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default KeyCapabilities;
