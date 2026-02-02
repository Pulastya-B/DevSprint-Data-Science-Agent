import React, { useEffect, useRef, useState } from 'react';
import { Loader2, AlertCircle, Download, Maximize2, Minimize2 } from 'lucide-react';

interface PlotData {
  type: 'plotly' | 'chartjs';
  name: string;
  data: any;
  created_at: string;
}

interface PlotRendererProps {
  plotData?: PlotData;
  plotUrl?: string;  // Fallback for legacy HTML plots
  title: string;
  onClose?: () => void;
}

// Lazy load Plotly to reduce bundle size
const loadPlotly = (): Promise<any> => {
  return new Promise((resolve, reject) => {
    if ((window as any).Plotly) {
      resolve((window as any).Plotly);
      return;
    }
    
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
    script.async = true;
    script.onload = () => resolve((window as any).Plotly);
    script.onerror = () => reject(new Error('Failed to load Plotly'));
    document.head.appendChild(script);
  });
};

export const PlotRenderer: React.FC<PlotRendererProps> = ({ 
  plotData, 
  plotUrl,
  title,
  onClose 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (!plotData && !plotUrl) {
      setError('No plot data provided');
      setLoading(false);
      return;
    }

    const renderPlot = async () => {
      try {
        setLoading(true);
        setError(null);

        if (plotData && plotData.type === 'plotly') {
          // Render Plotly chart from JSON data
          const Plotly = await loadPlotly();
          
          if (containerRef.current) {
            // Extract data and layout from the plot data
            const { data, layout, config } = plotData.data;
            
            // Apply dark theme
            const darkLayout = {
              ...layout,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#ffffff' },
              xaxis: { 
                ...layout?.xaxis, 
                gridcolor: 'rgba(255,255,255,0.1)',
                linecolor: 'rgba(255,255,255,0.2)'
              },
              yaxis: { 
                ...layout?.yaxis, 
                gridcolor: 'rgba(255,255,255,0.1)',
                linecolor: 'rgba(255,255,255,0.2)'
              },
              margin: { t: 40, r: 20, b: 40, l: 60 }
            };

            const darkConfig = {
              ...config,
              responsive: true,
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['lasso2d', 'select2d']
            };

            Plotly.newPlot(containerRef.current, data, darkLayout, darkConfig);
          }
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error rendering plot:', err);
        setError(err instanceof Error ? err.message : 'Failed to render plot');
        setLoading(false);
      }
    };

    renderPlot();

    // Cleanup
    return () => {
      if (containerRef.current && (window as any).Plotly) {
        (window as any).Plotly.purge(containerRef.current);
      }
    };
  }, [plotData, plotUrl]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && (window as any).Plotly && plotData) {
        (window as any).Plotly.Plots.resize(containerRef.current);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [plotData]);

  const handleDownload = () => {
    if (containerRef.current && (window as any).Plotly) {
      (window as any).Plotly.downloadImage(containerRef.current, {
        format: 'png',
        width: 1200,
        height: 800,
        filename: title.replace(/\s+/g, '_')
      });
    }
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  // If we only have a URL (legacy HTML plot), use iframe
  if (!plotData && plotUrl) {
    return (
      <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-black' : 'w-full h-full'}`}>
        <div className="absolute top-2 right-2 flex gap-2 z-10">
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
        <iframe
          src={plotUrl}
          className="w-full h-full border-0"
          title={title}
          sandbox="allow-scripts allow-same-origin"
        />
      </div>
    );
  }

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-[#0a0a0a]' : 'w-full h-full'}`}>
      {/* Controls */}
      <div className="absolute top-2 right-2 flex gap-2 z-10">
        <button
          onClick={handleDownload}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
          title="Download as PNG"
        >
          <Download className="w-4 h-4" />
        </button>
        <button
          onClick={toggleFullscreen}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
          title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
        >
          {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="flex items-center gap-3 text-white/60">
            <Loader2 className="w-6 h-6 animate-spin" />
            <span>Loading visualization...</span>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex items-center gap-3 text-red-400">
            <AlertCircle className="w-6 h-6" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Plot container */}
      <div 
        ref={containerRef} 
        className="w-full h-full min-h-[400px]"
        style={{ visibility: loading ? 'hidden' : 'visible' }}
      />
    </div>
  );
};

export default PlotRenderer;
