
import React from 'react';
import { cn } from '../lib/utils';

interface LogoProps {
  className?: string;
  showText?: boolean;
}

export const Logo: React.FC<LogoProps> = ({ className, showText = false }) => {
  return (
    <div className={cn("flex flex-col items-center", className)}>
      <svg
        viewBox="0 0 120 120"
        className="w-full h-full"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="100%" stopColor="#6366f1" />
          </linearGradient>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Central Core */}
        <circle cx="60" cy="60" r="6" fill="url(#logoGradient)" filter="url(#glow)" />

        {/* Inner Circuit Ring */}
        <circle cx="60" cy="60" r="18" stroke="url(#logoGradient)" strokeWidth="1" strokeDasharray="2 4" opacity="0.4" />
        
        {/* Complex Neural Paths (Stylized) */}
        <g opacity="0.8">
          {[0, 45, 90, 135, 180, 225, 270, 315].map((angle) => (
            <g key={angle} transform={`rotate(${angle} 60 60)`}>
              <path
                d="M60 35 L60 30 M60 30 L55 25 M60 30 L65 25"
                stroke="url(#logoGradient)"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
              <circle cx="55" cy="25" r="1.5" fill="url(#logoGradient)" />
              <circle cx="65" cy="25" r="1.5" fill="url(#logoGradient)" />
            </g>
          ))}
        </g>

        {/* Middle Dashed Ring */}
        <circle cx="60" cy="60" r="32" stroke="url(#logoGradient)" strokeWidth="1.5" strokeDasharray="10 6" opacity="0.6" />

        {/* Outer Orbital with Squares */}
        <circle cx="60" cy="60" r="45" stroke="url(#logoGradient)" strokeWidth="0.5" opacity="0.3" />
        {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((angle) => (
          <rect
            key={angle}
            x="58"
            y="12"
            width="4"
            height="4"
            fill="url(#logoGradient)"
            transform={`rotate(${angle} 60 60)`}
            rx="1"
          />
        ))}

        {/* Connection Spokes */}
        {[0, 90, 180, 270].map((angle) => (
          <line
            key={angle}
            x1="60"
            y1="16"
            x2="60"
            y2="30"
            stroke="url(#logoGradient)"
            strokeWidth="1"
            opacity="0.5"
            transform={`rotate(${angle} 60 60)`}
          />
        ))}
      </svg>
      {showText && (
        <span className="mt-2 text-white font-extrabold tracking-widest text-[10px] sm:text-xs uppercase">
          DATA SCIENCE AGENT
        </span>
      )}
    </div>
  );
};
