"""
Dashboard controller for the Rainfall vs Food Delivery Demand Dashboard

This module provides the DashboardController class for orchestrating the overall
dashboard flow and layout as specified in requirements 6.1, 6.3, and 6.4.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path
import io

from data_loader import DataLoader
from chart_generator import ChartGenerator
from insights_generator import InsightsGenerator


class ErrorHandler:
    """
    Utility class for creating user-friendly error messages and handling exceptions.
    
    Implements requirement 6.4 for user-friendly error messages without technical jargon.
    """
    
    @staticmethod
    def create_user_friendly_message(error: Exception, context: str = "") -> str:
        """
        Convert technical error messages into user-friendly ones.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            User-friendly error message string
        """
        error_str = str(error).lower()
        
        # File-related errors
        if "file not found" in error_str or "no such file" in error_str:
            return f"The file could not be found. Please check the file path and try uploading again."
        
        elif "permission denied" in error_str or "access denied" in error_str:
            return f"Unable to access the file. Please check file permissions and try again."
        
        elif "empty" in error_str and "file" in error_str:
            return f"The uploaded file appears to be empty. Please check your file and try again."
        
        # Data format errors
        elif "missing required columns" in error_str or "missing columns" in error_str:
            return f"Your file is missing required columns. Please ensure it has the correct column names and try again."
        
        elif "invalid" in error_str and "csv" in error_str:
            return f"The file format is not valid. Please upload a properly formatted CSV file."
        
        elif "date" in error_str and ("parse" in error_str or "format" in error_str):
            return f"Some dates in your file couldn't be understood. Please use a standard date format like YYYY-MM-DD."
        
        # Data validation errors
        elif "negative" in error_str or "invalid values" in error_str:
            return f"Some data values are invalid. Please check that all numbers are positive and try again."
        
        elif "correlation" in error_str:
            return f"Unable to calculate relationships between variables. This might be due to insufficient data."
        
        # Visualization errors
        elif "chart" in error_str or "plot" in error_str:
            return f"Unable to create visualization. Please check your data and try again."
        
        # Memory/performance errors
        elif "memory" in error_str:
            return f"The file is too large to process. Please try with a smaller dataset."
        
        elif "timeout" in error_str:
            return f"The operation took too long. Please try with a smaller dataset or check your internet connection."
        
        # Network errors
        elif "connection" in error_str or "network" in error_str:
            return f"Connection issue detected. Please check your internet connection and try again."
        
        # Generic fallback
        else:
            if context:
                return f"An issue occurred while {context}. Please try again or contact support if the problem persists."
            else:
                return f"An unexpected issue occurred. Please try again or contact support if the problem persists."
    
    @staticmethod
    def get_helpful_suggestion(error_type: str, data_type: str = "") -> str:
        """
        Provide helpful suggestions based on error type.
        
        Args:
            error_type: Type of error that occurred
            data_type: Type of data being processed (rainfall/delivery)
            
        Returns:
            Helpful suggestion string
        """
        suggestions = {
            'file_format': f"💡 **Tip:** Make sure your {data_type} file is a CSV with proper column headers.",
            'missing_columns': f"💡 **Tip:** {data_type.title()} files should have 'date' and {'precipitation_mm' if data_type == 'rainfall' else 'order_count'} columns.",
            'date_format': f"💡 **Tip:** Use date formats like 2023-01-15, 01/15/2023, or 15/01/2023.",
            'empty_data': f"💡 **Tip:** Check that your {data_type} file contains actual data rows, not just headers.",
            'invalid_values': f"💡 **Tip:** Make sure all {'precipitation' if data_type == 'rainfall' else 'order count'} values are positive numbers.",
            'no_overlap': f"💡 **Tip:** Ensure both datasets cover the same time period for relationship analysis."
        }
        
        return suggestions.get(error_type, "💡 **Tip:** Try refreshing the page and uploading your files again.")


class LoadingIndicator:
    """
    Utility class for managing loading indicators and status messages.
    
    Implements requirement 6.4 for loading indicators and status messages.
    """
    
    def __init__(self):
        self.placeholder = None
    
    def show_loading(self, message: str = "Processing...") -> None:
        """Show a loading indicator with message."""
        if not hasattr(st.session_state, 'loading_placeholder'):
            st.session_state.loading_placeholder = st.empty()
        
        st.session_state.loading_placeholder.info(f"⏳ {message}")
    
    def show_progress(self, progress: float, message: str = "Processing...") -> None:
        """Show a progress bar with message."""
        if not hasattr(st.session_state, 'progress_placeholder'):
            st.session_state.progress_placeholder = st.empty()
        
        with st.session_state.progress_placeholder.container():
            st.progress(progress)
            st.text(message)
    
    def clear_loading(self) -> None:
        """Clear loading indicators."""
        if hasattr(st.session_state, 'loading_placeholder'):
            st.session_state.loading_placeholder.empty()
        if hasattr(st.session_state, 'progress_placeholder'):
            st.session_state.progress_placeholder.empty()


class DashboardController:
    """
    Orchestrates the overall dashboard flow and layout for the rainfall vs delivery dashboard.
    
    This class implements the main dashboard interface as specified in requirements
    6.1, 6.3, and 6.4 for clean layout, logical organization, and user-friendly error handling.
    """
    
    def __init__(self):
        """Initialize the DashboardController with component instances."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.data_loader = DataLoader()
        self.chart_generator = ChartGenerator()
        self.insights_generator = InsightsGenerator()
        
        # Initialize utility components
        self.error_handler = ErrorHandler()
        self.loading_indicator = LoadingIndicator()
        
        # Initialize session state for data persistence
        if 'rainfall_data' not in st.session_state:
            st.session_state.rainfall_data = None
        if 'delivery_data' not in st.session_state:
            st.session_state.delivery_data = None
        if 'combined_data' not in st.session_state:
            st.session_state.combined_data = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'error_messages' not in st.session_state:
            st.session_state.error_messages = []
        if 'success_messages' not in st.session_state:
            st.session_state.success_messages = []
    
    def render_header(self) -> None:
        """
        Render the dashboard header with title and description.
        
        Implements requirement 6.1 for clean layout with clearly labeled sections
        and requirement 6.3 for logical organization with appropriate spacing.
        """
        try:
            # Main title with icon
            st.title("🌧️ Rainfall vs Food Delivery Demand Dashboard")
            
            # Description with proper spacing
            st.markdown("""
            This interactive dashboard explores the relationship between daily rainfall and food delivery demand 
            through data visualization and exploratory analysis. Upload your datasets or use sample data to begin.
            """)
            
            # Add visual separator
            st.markdown("---")
            
            self.logger.info("Dashboard header rendered successfully")
            
        except Exception as e:
            self.logger.error(f"Error rendering dashboard header: {str(e)}")
            st.error("Unable to load dashboard header. Please refresh the page.")
    
    def render_data_section(self) -> None:
        """
        Render the data loading section with status and quality metrics.
        
        Implements requirement 6.1 for clean layout and requirement 6.4 for
        user-friendly error messages without technical jargon.
        """
        try:
            st.subheader("📊 Data Management")
            
            # Create columns for data upload
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Rainfall Data**")
                rainfall_file = st.file_uploader(
                    "Upload rainfall CSV file",
                    type=['csv'],
                    key='rainfall_upload',
                    help="CSV file with 'date' and 'precipitation_mm' columns"
                )
                
                # Load sample rainfall data button
                if st.button("Use Sample Rainfall Data", key='sample_rainfall'):
                    self._load_sample_rainfall_data()
            
            with col2:
                st.markdown("**Delivery Data**")
                delivery_file = st.file_uploader(
                    "Upload delivery CSV file",
                    type=['csv'],
                    key='delivery_upload',
                    help="CSV file with 'date' and 'order_count' columns"
                )
                
                # Load sample delivery data button
                if st.button("Use Sample Delivery Data", key='sample_delivery'):
                    self._load_sample_delivery_data()
            
            # Process uploaded files
            if rainfall_file is not None:
                self._process_uploaded_file(rainfall_file, 'rainfall')
            
            if delivery_file is not None:
                self._process_uploaded_file(delivery_file, 'delivery')
            
            # Display data loading status
            self._display_data_status()
            
            # Display error messages if any
            self._display_error_messages()
            
            # Add spacing
            st.markdown("---")
            
        except Exception as e:
            self.logger.error(f"Error rendering data section: {str(e)}")
            st.error("Unable to load data management section. Please try refreshing the page.")
    
    def render_visualizations(self) -> None:
        """
        Render the visualization section with charts and plots.
        
        Implements requirement 6.1 for clean layout and requirement 6.3 for
        logical organization of multiple visualizations with appropriate spacing.
        """
        try:
            st.subheader("📈 Data Visualizations")
            
            if not st.session_state.data_loaded:
                st.info("📋 Upload datasets or use sample data to see visualizations.")
                return
            
            # Check if we have data to visualize
            rainfall_df = st.session_state.rainfall_data
            delivery_df = st.session_state.delivery_data
            combined_df = st.session_state.combined_data
            
            # Individual time series charts
            if rainfall_df is not None and not rainfall_df.empty:
                st.markdown("**Rainfall Over Time**")
                try:
                    rainfall_chart = self.chart_generator.create_time_series_chart(
                        rainfall_df, 
                        'precipitation_mm', 
                        'Daily Precipitation'
                    )
                    st.plotly_chart(rainfall_chart, use_container_width=True)
                except Exception as e:
                    self._display_chart_error("rainfall time series", str(e))
                    self._display_fallback_data_table(rainfall_df, "Rainfall Data")
            
            if delivery_df is not None and not delivery_df.empty:
                st.markdown("**Delivery Orders Over Time**")
                try:
                    delivery_chart = self.chart_generator.create_time_series_chart(
                        delivery_df, 
                        'order_count', 
                        'Daily Delivery Orders'
                    )
                    st.plotly_chart(delivery_chart, use_container_width=True)
                except Exception as e:
                    self._display_chart_error("delivery time series", str(e))
                    self._display_fallback_data_table(delivery_df, "Delivery Data")
            
            # Combined visualizations
            if combined_df is not None and not combined_df.empty:
                st.markdown("**Combined Analysis**")
                
                # Combined time series
                try:
                    combined_chart = self.chart_generator.create_combined_time_series(
                        combined_df,
                        "Rainfall and Delivery Data Over Time"
                    )
                    st.plotly_chart(combined_chart, use_container_width=True)
                except Exception as e:
                    self._display_chart_error("combined time series", str(e))
                    self._display_fallback_data_table(combined_df, "Combined Data")
                
                # Scatter plot for relationship analysis
                try:
                    scatter_chart = self.chart_generator.create_scatter_plot(
                        combined_df,
                        'precipitation_mm',
                        'order_count',
                        'Rainfall vs Delivery Orders Relationship'
                    )
                    st.plotly_chart(scatter_chart, use_container_width=True)
                except Exception as e:
                    self._display_chart_error("scatter plot", str(e))
                    # Show basic correlation as fallback
                    self._display_fallback_correlation(combined_df)
            
            # Add spacing
            st.markdown("---")
            
        except Exception as e:
            self.logger.error(f"Error rendering visualizations: {str(e)}")
            st.error("Unable to display visualizations. Please check your data and try again.")
    
    def render_insights(self) -> None:
        """
        Render the insights section with statistical summaries and observations.
        
        Implements requirement 6.1 for clean layout and requirement 6.3 for
        logical organization of insights display.
        """
        try:
            st.subheader("💡 Data Insights")
            
            if not st.session_state.data_loaded:
                st.info("📋 Upload datasets to see statistical insights and observations.")
                return
            
            rainfall_df = st.session_state.rainfall_data
            delivery_df = st.session_state.delivery_data
            combined_df = st.session_state.combined_data
            
            # Generate insights for individual datasets
            rainfall_insights = None
            delivery_insights = None
            
            if rainfall_df is not None and not rainfall_df.empty:
                try:
                    rainfall_insights = self.insights_generator.generate_summary_statistics(
                        rainfall_df, "Rainfall Data"
                    )
                except Exception as e:
                    self.logger.error(f"Error generating rainfall insights: {str(e)}")
            
            if delivery_df is not None and not delivery_df.empty:
                try:
                    delivery_insights = self.insights_generator.generate_summary_statistics(
                        delivery_df, "Delivery Data"
                    )
                except Exception as e:
                    self.logger.error(f"Error generating delivery insights: {str(e)}")
            
            # Display individual dataset insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Rainfall Data Summary**")
                if rainfall_insights:
                    self._display_dataset_insights(rainfall_insights)
                else:
                    st.info("No rainfall data available for analysis.")
            
            with col2:
                st.markdown("**Delivery Data Summary**")
                if delivery_insights:
                    self._display_dataset_insights(delivery_insights)
                else:
                    st.info("No delivery data available for analysis.")
            
            # Combined insights
            if combined_df is not None and not combined_df.empty and rainfall_insights and delivery_insights:
                st.markdown("**Combined Analysis Insights**")
                try:
                    combined_insights = self.insights_generator.generate_combined_insights(
                        combined_df, rainfall_insights, delivery_insights
                    )
                    self._display_combined_insights(combined_insights)
                except Exception as e:
                    self.logger.error(f"Error generating combined insights: {str(e)}")
                    st.warning("Unable to generate combined insights. Individual dataset summaries are shown above.")
            
        except Exception as e:
            self.logger.error(f"Error rendering insights section: {str(e)}")
            st.error("Unable to display insights. Please check your data and try again.")
    
    def _load_sample_rainfall_data(self) -> None:
        """Load sample rainfall data for demonstration purposes."""
        try:
            self.loading_indicator.show_loading("Loading sample rainfall data...")
            
            sample_file_path = Path("data/sample_rainfall.csv")
            if sample_file_path.exists():
                rainfall_df = self.data_loader.load_rainfall_data(str(sample_file_path))
                st.session_state.rainfall_data = rainfall_df
                self._add_success_message(f"Sample rainfall data loaded successfully: {len(rainfall_df)} records")
                self._update_combined_data()
            else:
                self._add_error_message("Sample rainfall data file not found. Please upload your own data.")
                st.info(self.error_handler.get_helpful_suggestion('file_format', 'rainfall'))
                
        except Exception as e:
            self.logger.error(f"Error loading sample rainfall data: {str(e)}")
            error_msg = self.error_handler.create_user_friendly_message(e, "loading sample rainfall data")
            self._add_error_message(error_msg)
        finally:
            self.loading_indicator.clear_loading()
    
    def _load_sample_delivery_data(self) -> None:
        """Load sample delivery data for demonstration purposes."""
        try:
            self.loading_indicator.show_loading("Loading sample delivery data...")
            
            sample_file_path = Path("data/sample_delivery.csv")
            if sample_file_path.exists():
                delivery_df = self.data_loader.load_delivery_data(str(sample_file_path))
                st.session_state.delivery_data = delivery_df
                self._add_success_message(f"Sample delivery data loaded successfully: {len(delivery_df)} records")
                self._update_combined_data()
            else:
                self._add_error_message("Sample delivery data file not found. Please upload your own data.")
                st.info(self.error_handler.get_helpful_suggestion('file_format', 'delivery'))
                
        except Exception as e:
            self.logger.error(f"Error loading sample delivery data: {str(e)}")
            error_msg = self.error_handler.create_user_friendly_message(e, "loading sample delivery data")
            self._add_error_message(error_msg)
        finally:
            self.loading_indicator.clear_loading()
    
    def _process_uploaded_file(self, uploaded_file, data_type: str) -> None:
        """
        Process an uploaded CSV file and load it into session state.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            data_type: Either 'rainfall' or 'delivery'
        """
        try:
            self.loading_indicator.show_loading(f"Processing {data_type} file...")
            
            # Validate file size (basic check)
            file_content = uploaded_file.read()
            if len(file_content) == 0:
                raise ValueError("The uploaded file is empty")
            
            # Load data based on type
            if data_type == 'rainfall':
                # Save to temporary CSV file for processing
                temp_path = f"temp_rainfall_{uploaded_file.name}"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(file_content.decode('utf-8'))
                
                rainfall_df = self.data_loader.load_rainfall_data(temp_path)
                st.session_state.rainfall_data = rainfall_df
                self._add_success_message(f"Rainfall data uploaded successfully: {len(rainfall_df)} records")
                
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
                
            elif data_type == 'delivery':
                # Save to temporary CSV file for processing
                temp_path = f"temp_delivery_{uploaded_file.name}"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(file_content.decode('utf-8'))
                
                delivery_df = self.data_loader.load_delivery_data(temp_path)
                st.session_state.delivery_data = delivery_df
                self._add_success_message(f"Delivery data uploaded successfully: {len(delivery_df)} records")
                
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
            
            # Update combined data
            self._update_combined_data()
            
        except UnicodeDecodeError:
            error_msg = f"Unable to read the {data_type} file. Please ensure it's a valid text file with UTF-8 encoding."
            self._add_error_message(error_msg)
            st.info(self.error_handler.get_helpful_suggestion('file_format', data_type))
            
        except Exception as e:
            self.logger.error(f"Error processing uploaded {data_type} file: {str(e)}")
            error_msg = self.error_handler.create_user_friendly_message(e, f"processing {data_type} file")
            self._add_error_message(error_msg)
            
            # Provide specific suggestions based on error type
            if "missing required columns" in str(e).lower():
                st.info(self.error_handler.get_helpful_suggestion('missing_columns', data_type))
            elif "date" in str(e).lower():
                st.info(self.error_handler.get_helpful_suggestion('date_format', data_type))
            elif "empty" in str(e).lower():
                st.info(self.error_handler.get_helpful_suggestion('empty_data', data_type))
        finally:
            self.loading_indicator.clear_loading()
    
    def _update_combined_data(self) -> None:
        """Update the combined dataset when both rainfall and delivery data are available."""
        try:
            rainfall_df = st.session_state.rainfall_data
            delivery_df = st.session_state.delivery_data
            
            if rainfall_df is not None and delivery_df is not None:
                combined_df = self.data_loader.join_datasets(rainfall_df, delivery_df)
                st.session_state.combined_data = combined_df
                st.session_state.data_loaded = True
                
                if not combined_df.empty:
                    st.info(f"📊 Combined dataset created with {len(combined_df)} matching dates")
                else:
                    st.warning("⚠️ No matching dates found between datasets")
            else:
                st.session_state.data_loaded = bool(rainfall_df is not None or delivery_df is not None)
                
        except Exception as e:
            self.logger.error(f"Error updating combined data: {str(e)}")
            self._add_error_message("Unable to combine datasets. Please check that both files have valid date columns.")
    
    def _display_data_status(self) -> None:
        """Display the current data loading status."""
        try:
            rainfall_df = st.session_state.rainfall_data
            delivery_df = st.session_state.delivery_data
            combined_df = st.session_state.combined_data
            
            st.markdown("**Data Status**")
            
            # Rainfall status
            if rainfall_df is not None:
                st.success(f"🌧️ Rainfall: {len(rainfall_df)} records loaded")
            else:
                st.info("🌧️ Rainfall: No data loaded")
            
            # Delivery status
            if delivery_df is not None:
                st.success(f"🚚 Delivery: {len(delivery_df)} records loaded")
            else:
                st.info("🚚 Delivery: No data loaded")
            
            # Combined status
            if combined_df is not None and not combined_df.empty:
                st.success(f"📊 Combined: {len(combined_df)} matching dates")
            elif rainfall_df is not None and delivery_df is not None:
                st.warning("📊 Combined: No matching dates found")
            else:
                st.info("📊 Combined: Waiting for both datasets")
                
        except Exception as e:
            self.logger.error(f"Error displaying data status: {str(e)}")
            st.warning("Unable to display data status.")
    
    def _display_error_messages(self) -> None:
        """Display any accumulated error and success messages."""
        # Display success messages
        if st.session_state.success_messages:
            for success_msg in st.session_state.success_messages:
                st.success(f"✅ {success_msg}")
        
        # Display error messages
        if st.session_state.error_messages:
            st.markdown("**Issues Found**")
            for error_msg in st.session_state.error_messages:
                st.error(f"❌ {error_msg}")
            
            # Provide general help
            with st.expander("Need Help?"):
                st.markdown("""
                **Common Solutions:**
                - Make sure your CSV files have the correct column names
                - Check that dates are in a standard format (YYYY-MM-DD)
                - Ensure all numeric values are positive
                - Verify that files are not empty
                
                **Required Columns:**
                - Rainfall files: `date`, `precipitation_mm`
                - Delivery files: `date`, `order_count`
                """)
        
        # Clear messages button (only show if there are messages)
        if st.session_state.error_messages or st.session_state.success_messages:
            if st.button("Clear All Messages", key='clear_messages'):
                st.session_state.error_messages = []
                st.session_state.success_messages = []
                st.rerun()
    
    def _display_chart_error(self, chart_type: str, error_details: str) -> None:
        """
        Display a user-friendly error message for chart generation failures.
        
        Implements requirement 6.4 for user-friendly error messages without technical jargon.
        """
        user_message = f"Unable to display {chart_type} chart. "
        
        # Provide helpful suggestions based on common error patterns
        if "empty" in error_details.lower():
            user_message += "This might be because there's no data available for this visualization."
        elif "column" in error_details.lower():
            user_message += "Please check that your data file has the required columns."
        elif "date" in error_details.lower():
            user_message += "There might be an issue with the date format in your data."
        else:
            user_message += "Please check your data and try again."
        
        st.warning(f"⚠️ {user_message}")
        st.info("📊 Showing data table as alternative below:")
    
    def _display_fallback_data_table(self, df: pd.DataFrame, title: str) -> None:
        """
        Display a fallback data table when charts fail to render.
        
        Implements requirement 6.4 for fallback displays for visualization errors.
        """
        try:
            st.markdown(f"**{title} (Table View)**")
            
            if df.empty:
                st.info("No data available to display.")
                return
            
            # Show basic statistics
            if len(df) > 0:
                st.write(f"📊 **Data Summary:** {len(df)} records")
                
                # Show date range if available
                if 'date' in df.columns:
                    date_col = df['date'].dropna()
                    if len(date_col) > 0:
                        st.write(f"📅 **Date Range:** {date_col.min()} to {date_col.max()}")
            
            # Display sample of data (first 10 rows)
            st.dataframe(df.head(10), use_container_width=True)
            
            if len(df) > 10:
                st.info(f"Showing first 10 of {len(df)} records. Full data is being used for analysis.")
                
        except Exception as e:
            self.logger.error(f"Error displaying fallback table: {str(e)}")
            st.error("Unable to display data table. Please check your data format.")
    
    def _display_fallback_correlation(self, df: pd.DataFrame) -> None:
        """
        Display basic correlation information when scatter plot fails.
        
        Implements requirement 6.4 for fallback displays for visualization errors.
        """
        try:
            if 'precipitation_mm' in df.columns and 'order_count' in df.columns:
                clean_df = df.dropna(subset=['precipitation_mm', 'order_count'])
                
                if len(clean_df) >= 2:
                    correlation = clean_df['precipitation_mm'].corr(clean_df['order_count'])
                    
                    if not pd.isna(correlation):
                        st.markdown("**Relationship Analysis (Text Summary)**")
                        
                        # Interpret correlation strength
                        abs_corr = abs(correlation)
                        if abs_corr < 0.1:
                            strength = "very weak"
                        elif abs_corr < 0.3:
                            strength = "weak"
                        elif abs_corr < 0.5:
                            strength = "moderate"
                        elif abs_corr < 0.7:
                            strength = "strong"
                        else:
                            strength = "very strong"
                        
                        direction = "positive" if correlation > 0 else "negative"
                        
                        st.info(f"📈 **Correlation:** {correlation:.3f} ({strength} {direction} relationship)")
                        st.write(f"📊 **Sample Size:** {len(clean_df)} data points")
                    else:
                        st.info("Unable to calculate correlation - insufficient variation in data.")
                else:
                    st.info("Insufficient data points for relationship analysis.")
            else:
                st.info("Required columns not available for relationship analysis.")
                
        except Exception as e:
            self.logger.error(f"Error displaying fallback correlation: {str(e)}")
            st.info("Unable to calculate relationship statistics.")
    
    def _display_dataset_insights(self, insights: Dict[str, Any]) -> None:
        """Display insights for an individual dataset."""
        try:
            # Data quality message
            data_quality = insights.get('data_quality', {})
            quality_message = data_quality.get('message', 'No quality information available.')
            
            if data_quality.get('completeness', 0) >= 90:
                st.success(quality_message)
            elif data_quality.get('completeness', 0) >= 70:
                st.info(quality_message)
            else:
                st.warning(quality_message)
            
            # Key statistics
            statistics = insights.get('statistics', {})
            if statistics:
                for col, stats in statistics.items():
                    if stats['count'] > 0:
                        col_name = col.replace('_', ' ').title()
                        st.metric(
                            f"{col_name} Average",
                            f"{stats['mean']:.1f}",
                            help=f"Range: {stats['min']:.1f} - {stats['max']:.1f}"
                        )
            
            # Observations
            observations = insights.get('observations', [])
            if observations:
                st.markdown("**Key Observations:**")
                for obs in observations[:3]:  # Limit to top 3 observations
                    st.write(f"• {obs}")
                    
        except Exception as e:
            self.logger.error(f"Error displaying dataset insights: {str(e)}")
            st.warning("Unable to display insights for this dataset.")
    
    def _display_combined_insights(self, insights: Dict[str, Any]) -> None:
        """Display insights for the combined analysis."""
        try:
            # Correlation analysis
            correlation_analysis = insights.get('correlation_analysis')
            if correlation_analysis and correlation_analysis.get('correlation') is not None:
                corr_value = correlation_analysis['correlation']
                interpretation = correlation_analysis.get('interpretation', 'No interpretation available')
                
                st.metric(
                    "Correlation Coefficient",
                    f"{corr_value:.3f}",
                    help=interpretation
                )
            
            # Relationship observations
            relationship_obs = insights.get('relationship_observations', [])
            if relationship_obs:
                st.markdown("**Relationship Patterns:**")
                for obs in relationship_obs:
                    st.write(f"• {obs}")
            
            # Data quality notes
            quality_notes = insights.get('data_quality_notes', [])
            if quality_notes:
                with st.expander("Data Quality Notes"):
                    for note in quality_notes:
                        st.write(f"• {note}")
            
            # Summary message
            summary_message = insights.get('summary_message', '')
            if summary_message:
                st.info(summary_message)
                
        except Exception as e:
            self.logger.error(f"Error displaying combined insights: {str(e)}")
            st.warning("Unable to display combined analysis insights.")
    
    def _create_user_friendly_error_message(self, error_details: str, data_type: str) -> str:
        """
        Create user-friendly error messages without technical jargon.
        
        Implements requirement 6.4 for user-friendly error messages.
        """
        error_lower = error_details.lower()
        
        if "file not found" in error_lower or "no such file" in error_lower:
            return f"The {data_type} file could not be found. Please check the file path and try again."
        
        elif "missing required columns" in error_lower or "missing columns" in error_lower:
            if data_type == 'rainfall':
                return "Your rainfall file is missing required columns. Please ensure it has 'date' and 'precipitation_mm' columns."
            else:
                return "Your delivery file is missing required columns. Please ensure it has 'date' and 'order_count' columns."
        
        elif "empty" in error_lower:
            return f"The {data_type} file appears to be empty or contains no valid data."
        
        elif "date" in error_lower and "parse" in error_lower:
            return f"Some dates in your {data_type} file couldn't be understood. Please use a standard date format like YYYY-MM-DD."
        
        elif "csv" in error_lower or "format" in error_lower:
            return f"There's an issue with the {data_type} file format. Please ensure it's a valid CSV file."
        
        elif "permission" in error_lower or "access" in error_lower:
            return f"Unable to access the {data_type} file. Please check file permissions and try again."
        
        else:
            return f"There was an issue loading your {data_type} data. Please check the file format and try again."
    
    def _add_error_message(self, message: str) -> None:
        """Add an error message to the session state for display."""
        if message not in st.session_state.error_messages:
            st.session_state.error_messages.append(message)
    
    def _add_success_message(self, message: str) -> None:
        """Add a success message to the session state for display."""
        if message not in st.session_state.success_messages:
            st.session_state.success_messages.append(message)
    
    def render_dashboard(self) -> None:
        """
        Render the complete dashboard with all sections.
        
        Main entry point for dashboard rendering that orchestrates all components
        with proper error handling and user feedback.
        """
        try:
            # Configure Streamlit page
            st.set_page_config(
                page_title="Rainfall vs Delivery Dashboard",
                page_icon="🌧️",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Render all sections in order
            self.render_header()
            self.render_data_section()
            self.render_visualizations()
            self.render_insights()
            
            # Footer
            st.markdown("---")
            st.markdown("*Dashboard built with Streamlit for exploratory data analysis*")
            
        except Exception as e:
            self.logger.error(f"Critical error rendering dashboard: {str(e)}")
            st.error("A critical error occurred while loading the dashboard. Please refresh the page and try again.")