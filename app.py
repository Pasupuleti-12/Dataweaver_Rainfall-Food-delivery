"""
Rainfall vs Food Delivery Demand Dashboard

An interactive web application that explores the relationship between daily rainfall 
and food delivery demand through data visualization and exploratory analysis.
"""

import logging
from dashboard_controller import DashboardController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main application entry point"""
    try:
        # Initialize and render the dashboard
        dashboard = DashboardController()
        dashboard.render_dashboard()
        
    except Exception as e:
        logging.error(f"Critical error in main application: {str(e)}")
        # Fallback error display if dashboard controller fails
        import streamlit as st
        st.error("A critical error occurred while starting the dashboard. Please refresh the page and try again.")

if __name__ == "__main__":
    main()