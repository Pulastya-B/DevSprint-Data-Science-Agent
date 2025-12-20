
import React from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, Zap, ShieldCheck, Clock } from 'lucide-react';

const ProblemSolution = () => {
  return (
    <section className="py-24 relative bg-[#030303] overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl md:text-5xl font-extrabold text-white mb-6 tracking-tight">
              The Data Science <span className="text-rose-400">Bottleneck</span>
            </h2>
            <p className="text-white/60 text-lg mb-8 leading-relaxed font-medium">
              Modern data science is 80% manual labor. Cleaning messy datasets, engineering features, and tuning models takes weeks of repetitive effort. Mistakes are costly, and scaling insights is slow.
            </p>
            <ul className="space-y-4">
              {[
                { icon: AlertCircle, text: "Error-prone manual data preprocessing", color: "text-rose-400" },
                { icon: Clock, text: "Days spent on hyperparameter tuning", color: "text-rose-400" },
                { icon: AlertCircle, text: "Disconnected silos of code and insights", color: "text-rose-400" },
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-3 text-white/80 font-semibold">
                  <item.icon className={`w-5 h-5 ${item.color}`} />
                  <span>{item.text}</span>
                </li>
              ))}
            </ul>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="relative p-8 md:p-12 rounded-3xl bg-gradient-to-br from-indigo-500/10 via-white/5 to-rose-500/10 border border-white/10"
          >
            <div className="absolute -top-6 -right-6 w-32 h-32 bg-indigo-500/20 blur-3xl" />
            <h2 className="text-3xl md:text-5xl font-extrabold text-white mb-6 tracking-tight">
              The <span className="text-indigo-400">Autonomous</span> Solution
            </h2>
            <p className="text-white/60 text-lg mb-8 leading-relaxed font-medium">
              DATA SCIENCE AGENT automates the entire lifecycle. From raw CSV to production-ready models and interactive dashboards, our agent uses 82+ specialized tools to deliver precision at scale.
            </p>
            <ul className="space-y-4">
              {[
                { icon: Zap, text: "Instant feature engineering and selection", color: "text-indigo-400" },
                { icon: ShieldCheck, text: "Automated error recovery and re-training", color: "text-indigo-400" },
                { icon: Zap, text: "Explainable AI (XAI) reports by default", color: "text-indigo-400" },
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-3 text-white/80 font-semibold">
                  <item.icon className={`w-5 h-5 ${item.color}`} />
                  <span>{item.text}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default ProblemSolution;
