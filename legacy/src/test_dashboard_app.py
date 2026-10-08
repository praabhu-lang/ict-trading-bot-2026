# src/test_dashboard_app.py
import os
import sys
import pytest
from streamlit.testing.v1.app_test import AppTest

# Append root directory to path for robust imports across scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_dashboard_renders_cleanly():
    """Uses Streamlit's LocalScriptRunner to simulate a live UI interaction cycle."""
    script_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard.py"))
    at = AppTest.from_file(script_abs_path, default_timeout=15)
    
    # Test unauthenticated state (gate blocks main content rendering)
    at.run()
    assert len(at.title) == 0
    assert len(at.sidebar.text_input) >= 2

    # Authenticate via session state and test fully rendered dashboard
    at.session_state["authenticated"] = True
    at.run()
    
    # Assert Title exists and is visible
    assert len(at.title) > 0
    assert "Multi-Strategy Trading Dashboard" in at.title[0].value
    
    # Assert controls load successfully
    assert len(at.selectbox) > 0
    assert len(at.number_input) >= 2