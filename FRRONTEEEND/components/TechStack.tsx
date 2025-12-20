
import React from 'react';
import { motion } from 'framer-motion';

const techs = [
  "Python", "Polars", "Pandas", "Scikit-Learn", "XGBoost", "LightGBM", "Groq", "Gemini", "FastAPI", "Cloud Run", "Docker", "PyTorch"
];

const TechStack = () => {
  return (
    <section className="py-24 bg-[#030303]">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-12">
          <h3 className="text-xs font-bold uppercase tracking-[0.3em] text-white/30 italic">Built with the modern AI Stack</h3>
        </div>
        
        <div className="flex flex-wrap justify-center gap-4 md:gap-6 opacity-60">
          {techs.map((tech, i) => (
            <motion.div
              key={tech}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              className="px-5 py-2 rounded-lg border border-white/5 bg-white/[0.02] text-white/80 font-bold text-xs md:text-sm whitespace-nowrap tracking-wide uppercase"
            >
              {tech}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TechStack;
