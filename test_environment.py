#!/usr/bin/env python3
"""
Quick test script to verify all core imports work
"""

print("Testing core imports...")

try:
    print("  ✓ Python standard library")
    import sys
    import os
    
    print("  ✓ Data processing")
    import polars as pl
    import pandas as pd
    import numpy as np
    
    print("  ✓ Machine learning")
    import sklearn
    import xgboost
    import lightgbm
    
    print("  ✓ Visualization")
    import matplotlib
    import seaborn
    import plotly
    
    print("  ✓ LLM clients")
    import groq
    
    print("  ✓ Web framework")
    import gradio
    import fastapi
    
    print("\n✅ All core dependencies installed successfully!")
    print(f"\nPython version: {sys.version}")
    print(f"Gradio version: {gradio.__version__}")
    print(f"Polars version: {pl.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)

print("\n🎉 Environment setup complete! You can now run:")
print("   .venv/bin/python chat_ui.py")
