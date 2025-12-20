
import React, { useRef, useId, useEffect } from 'react';
import { motion, animate, useMotionValue, AnimationPlaybackControls } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { Logo } from './Logo';

function mapRange(
    value: number,
    fromLow: number,
    fromHigh: number,
    toLow: number,
    toHigh: number
): number {
    if (fromLow === fromHigh) {
        return toLow;
    }
    const percentage = (value - fromLow) / (fromHigh - fromLow);
    return toLow + percentage * (toHigh - toLow);
}

const Footer = () => {
  const id = useId().replace(/:/g, "");
  const instanceId = `footer-shadow-${id}`;
  const feColorMatrixRef = useRef<SVGFEColorMatrixElement>(null);
  const hueRotateMotionValue = useMotionValue(0);
  const hueRotateAnimation = useRef<AnimationPlaybackControls | null>(null);

  // Configuration from ShadowSection
  const animationScale = 50;
  const animationSpeed = 15;
  const displacementScale = mapRange(animationScale, 1, 100, 20, 100);
  const animationDuration = mapRange(animationSpeed, 1, 100, 1000, 50);

  useEffect(() => {
    if (feColorMatrixRef.current) {
        hueRotateAnimation.current = animate(hueRotateMotionValue, 360, {
            duration: animationDuration / 25,
            repeat: Infinity,
            repeatType: "loop",
            ease: "linear",
            onUpdate: (value: number) => {
                if (feColorMatrixRef.current) {
                    feColorMatrixRef.current.setAttribute("values", String(value));
                }
            }
        });
        return () => hueRotateAnimation.current?.stop();
    }
  }, [animationDuration, hueRotateMotionValue]);

  return (
    <footer className="bg-[#030303] overflow-hidden">
      {/* High-Impact CTA with Atmospheric Shadow UI */}
      <section className="relative w-full py-32 md:py-48 flex items-center justify-center border-t border-white/5">
        <div
            className="absolute inset-0 pointer-events-none overflow-hidden"
            style={{
                filter: `url(#${instanceId}) blur(12px)`,
                opacity: 0.8
            }}
        >
            <svg style={{ position: "absolute", width: 0, height: 0 }}>
                <defs>
                    <filter id={instanceId}>
                        <feTurbulence
                            result="undulation"
                            numOctaves="2"
                            baseFrequency={`${mapRange(animationScale, 0, 100, 0.001, 0.0005)},${mapRange(animationScale, 0, 100, 0.004, 0.002)}`}
                            seed="0"
                            type="turbulence"
                        />
                        <feColorMatrix
                            ref={feColorMatrixRef}
                            in="undulation"
                            type="hueRotate"
                            values="180"
                        />
                        <feColorMatrix
                            in="dist"
                            result="circulation"
                            type="matrix"
                            values="4 0 0 0 1  4 0 0 0 1  4 0 0 0 1  1 0 0 0 0"
                        />
                        <feDisplacementMap
                            in="SourceGraphic"
                            in2="circulation"
                            scale={displacementScale}
                            result="dist"
                        />
                        <feDisplacementMap
                            in="dist"
                            in2="undulation"
                            scale={displacementScale}
                            result="output"
                        />
                    </filter>
                </defs>
            </svg>
            <div
                style={{
                    backgroundColor: 'rgba(99, 102, 241, 0.4)',
                    maskImage: `url('https://framerusercontent.com/images/ceBGguIpUU8luwByxuQz79t7To.png')`,
                    maskSize: "cover",
                    maskRepeat: "no-repeat",
                    maskPosition: "center",
                    width: "120%",
                    height: "120%",
                    position: 'absolute',
                    top: '-10%',
                    left: '-10%'
                }}
            />
        </div>

        {/* Noise overlay */}
        <div
            className="absolute inset-0 pointer-events-none opacity-[0.03]"
            style={{
                backgroundImage: `url("https://framerusercontent.com/images/g0QcWrxr87K0ufOxIUFBakwYA8.png")`,
                backgroundSize: '100px',
                backgroundRepeat: "repeat",
            }}
        />

        <div className="relative z-20 max-w-7xl mx-auto px-6 text-center">
            <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
            >
                <h2 className="text-4xl md:text-7xl font-extrabold text-white mb-8 tracking-tighter">
                    Ready to automate your workflow?
                </h2>
                <p className="text-white/50 text-xl md:text-2xl mb-12 max-w-2xl mx-auto font-medium leading-relaxed">
                    Build smarter ML workflows with AI autonomy. Join the next generation of data scientists.
                </p>
                <button className="group relative px-10 py-5 bg-white text-black font-extrabold rounded-2xl transition-all hover:scale-105 active:scale-95 shadow-[0_0_50px_-12px_rgba(255,255,255,0.5)] flex items-center gap-3 mx-auto">
                    Get Started Now
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
            </motion.div>
        </div>

        {/* Gradient fades to blend with rest of footer */}
        <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-[#030303] to-transparent z-10" />
        <div className="absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-[#030303] to-transparent z-10" />
      </section>

      {/* Main Footer Links */}
      <div className="max-w-7xl mx-auto px-6 pb-20">
        <div className="pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-4">
            <Logo className="w-8 h-8" />
            <span className="text-white font-extrabold tracking-tight uppercase">DATA SCIENCE AGENT</span>
          </div>
          <div className="text-white/30 text-[10px] sm:text-xs font-semibold uppercase tracking-wider">
            © 2025 Data Science Agent. Built for the autonomous future.
          </div>
          <div className="flex gap-8 text-white/40 text-sm font-bold italic">
            <a href="#" className="hover:text-white transition-colors">Twitter</a>
            <a href="#" className="hover:text-white transition-colors">GitHub</a>
            <a href="#" className="hover:text-white transition-colors">Docs</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
