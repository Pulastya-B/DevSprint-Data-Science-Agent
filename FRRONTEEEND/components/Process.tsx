
import React from 'react';
import { motion } from 'framer-motion';

const steps = [
  {
    number: "01",
    title: "Ingest Data",
    description: "Upload your raw CSV, JSON, or Parquet files directly to the secure environment."
  },
  {
    number: "02",
    title: "Define Objective",
    description: "Describe what you want to achieve in natural language. 'Predict churn' or 'Find outliers'."
  },
  {
    number: "03",
    title: "Agent Execution",
    description: "The agent orchestrates tools to clean, transform, and model your data autonomously."
  },
  {
    number: "04",
    title: "Receive Assets",
    description: "Get fully trained models, performance metrics, and interactive explainable reports."
  }
];

const Process = () => {
  return (
    <section id="process" className="py-24 bg-[#030303] border-y border-white/5">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-4 tracking-tight">How it Works</h2>
          <p className="text-white/40 text-xl font-medium">From raw data to actionable intelligence in 4 steps.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
          {steps.map((step, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="relative"
            >
              <span className="text-7xl font-extrabold text-white/5 absolute -top-10 -left-4 select-none italic">
                {step.number}
              </span>
              <div className="relative z-10">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2 tracking-tight">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                  {step.title}
                </h3>
                <p className="text-white/40 leading-relaxed font-medium">
                  {step.description}
                </p>
              </div>
              {i < steps.length - 1 && (
                <div className="hidden lg:block absolute top-1/2 -right-6 w-12 h-[1px] bg-gradient-to-r from-white/10 to-transparent" />
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Process;
