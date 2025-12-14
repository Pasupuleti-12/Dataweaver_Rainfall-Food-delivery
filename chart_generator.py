"""
Chart generation components for the Rainfall vs Food Delivery Demand Dashboard

This module provides the ChartGenerator class for creating interactive visualizations
using Plotly as specified in the requirements for time series and scatter plot displays.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from datetime import date


class ChartGenerator:
    """
    Creates interactive visualizations using Plotly for rainfall and delivery data.
    
    This class implements chart generation functionality as specified in requirements
    1.1, 1.3, 2.1, 2.3, 4.1, and 4.2 for time series and relationship visualization.
    """
    
    def __init__(self):
        """Initialize the ChartGenerator with logging configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Default color scheme for visual differentiation
        self.colors = {
            'rainfall': '#1f77b4',  # Blue
            'delivery': '#ff7f0e',  # Orange
            'scatter': '#2ca02c',   # Green
            'correlation': '#d62728' # Red
        }
        
        # Default chart styling
        self.default_layout = {
            'font': {'size': 12},
            'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60},
            'hovermode': 'x unified',
            'showlegend': True
        }
    
    def create_time_series_chart(
        self, 
        df: pd.DataFrame, 
        column: str, 
        title: str,
        y_axis_title: Optional[str] = None,
        color: Optional[str] = None,
        show_data_quality: bool = True
    ) -> go.Figure:
        """
        Create a time series chart for rainfall or delivery data.
        
        Implements requirements 1.1, 1.3, 2.1, 2.3 for time series visualization
        with proper axis labeling, titles, and units. Also handles missing data
        gracefully as per requirements 1.2, 1.4, 2.2, 2.4.
        
        Args:
            df: DataFrame with 'date' column and the specified data column
            column: Name of the column to plot (e.g., 'precipitation_mm', 'order_count')
            title: Chart title
            y_axis_title: Y-axis label (auto-generated if None)
            color: Line color (uses default scheme if None)
            show_data_quality: Whether to show data quality information in title
            
        Returns:
            Plotly Figure object with the time series chart
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            # Validate input data
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            required_columns = ['date', column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Ensure data is sorted by date for proper time series display
            df_sorted = df.sort_values('date').copy()
            
            # Handle missing data and generate quality information
            original_count = len(df_sorted)
            df_clean = df_sorted.dropna(subset=[column]).copy()
            missing_count = original_count - len(df_clean)
            
            # Generate data quality message for user-friendly feedback
            quality_message = ""
            if missing_count > 0:
                quality_message = f" ({missing_count} missing values excluded)"
            elif len(df_clean) == 0:
                quality_message = " (No valid data available)"
            
            # Auto-generate y-axis title if not provided
            if y_axis_title is None:
                if column == 'precipitation_mm':
                    y_axis_title = 'Precipitation (mm)'
                elif column == 'order_count':
                    y_axis_title = 'Number of Orders'
                else:
                    y_axis_title = column.replace('_', ' ').title()
            
            # Select color based on data type if not specified
            if color is None:
                if 'precipitation' in column.lower() or 'rainfall' in column.lower():
                    color = self.colors['rainfall']
                elif 'order' in column.lower() or 'delivery' in column.lower():
                    color = self.colors['delivery']
                else:
                    color = self.colors['rainfall']  # Default
            
            # Create the time series chart
            fig = go.Figure()
            
            # Handle case where no valid data exists
            if len(df_clean) == 0:
                # Create empty chart with informative message
                fig.add_annotation(
                    text="No valid data available for visualization",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
            else:
                # Add data trace with proper handling of gaps
                fig.add_trace(go.Scatter(
                    x=df_clean['date'],
                    y=df_clean[column],
                    mode='lines+markers',
                    name=y_axis_title,
                    line=dict(color=color, width=2),
                    marker=dict(size=4, color=color),
                    connectgaps=False,  # Don't connect across missing data gaps
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                 f'<b>{y_axis_title}:</b> %{{y}}<br>' +
                                 '<extra></extra>'
                ))
            
            # Update layout with proper labels and formatting
            chart_title = title
            if show_data_quality and quality_message:
                chart_title += quality_message
                
            fig.update_layout(
                title={
                    'text': chart_title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': 'black'}
                },
                xaxis_title='Date',
                yaxis_title=y_axis_title,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    tickformat='%Y-%m-%d'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                **self.default_layout
            )
            
            # Ensure chronological ordering is maintained
            if len(df_clean) > 1:
                date_range = [df_clean['date'].min(), df_clean['date'].max()]
                fig.update_xaxes(range=date_range)
            elif len(df_sorted) > 1:
                # Use original data range even if no valid data for proper axis scaling
                date_range = [df_sorted['date'].min(), df_sorted['date'].max()]
                fig.update_xaxes(range=date_range)
            
            self.logger.info(f"Created time series chart for {column} with {len(df_clean)} valid data points ({missing_count} missing)")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating time series chart: {str(e)}")
            raise
    
    def create_scatter_plot(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str,
        title: Optional[str] = None,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        show_correlation: bool = True,
        show_data_quality: bool = True
    ) -> go.Figure:
        """
        Create a scatter plot for relationship visualization between two variables.
        
        Implements requirements 4.1 and 4.2 for scatter plot generation with
        visual differentiation and correlation information display. Handles
        missing data gracefully per requirements 1.2, 2.2.
        
        Args:
            df: DataFrame containing the data to plot
            x_col: Name of the x-axis column
            y_col: Name of the y-axis column  
            title: Chart title (auto-generated if None)
            x_axis_title: X-axis label (auto-generated if None)
            y_axis_title: Y-axis label (auto-generated if None)
            show_correlation: Whether to display correlation coefficient
            show_data_quality: Whether to show data quality information
            
        Returns:
            Plotly Figure object with the scatter plot
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            # Validate input data
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            required_columns = [x_col, y_col]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove any rows with NaN values in the plotting columns
            original_count = len(df)
            df_clean = df.dropna(subset=[x_col, y_col]).copy()
            missing_count = original_count - len(df_clean)
            
            # Generate data quality message
            quality_message = ""
            if missing_count > 0:
                quality_message = f" ({missing_count} incomplete records excluded)"
            
            if df_clean.empty:
                # Create empty chart with informative message instead of raising error
                fig = go.Figure()
                fig.add_annotation(
                    text="No complete data pairs available for scatter plot",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
                
                # Set basic layout for empty chart
                fig.update_layout(
                    title={
                        'text': title or "Scatter Plot - No Data Available",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16, 'color': 'black'}
                    },
                    **self.default_layout
                )
                return fig
            
            # Auto-generate titles if not provided
            if title is None:
                title = f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}"
            
            if x_axis_title is None:
                if 'precipitation' in x_col.lower():
                    x_axis_title = 'Precipitation (mm)'
                elif 'order' in x_col.lower():
                    x_axis_title = 'Number of Orders'
                else:
                    x_axis_title = x_col.replace('_', ' ').title()
            
            if y_axis_title is None:
                if 'precipitation' in y_col.lower():
                    y_axis_title = 'Precipitation (mm)'
                elif 'order' in y_col.lower():
                    y_axis_title = 'Number of Orders'
                else:
                    y_axis_title = y_col.replace('_', ' ').title()
            
            # Create the scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_clean[x_col],
                y=df_clean[y_col],
                mode='markers',
                name='Data Points',
                marker=dict(
                    size=8,
                    color=self.colors['scatter'],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertemplate=f'<b>{x_axis_title}:</b> %{{x}}<br>' +
                             f'<b>{y_axis_title}:</b> %{{y}}<br>' +
                             '<extra></extra>'
            ))
            
            # Calculate and display correlation if requested
            correlation_text = ""
            if show_correlation and len(df_clean) >= 2:
                try:
                    correlation = df_clean[x_col].corr(df_clean[y_col])
                    if not np.isnan(correlation):
                        correlation_text = f"<br>Correlation: {correlation:.3f}"
                        
                        # Add trend line if correlation is significant
                        if abs(correlation) > 0.1:  # Arbitrary threshold for showing trend line
                            z = np.polyfit(df_clean[x_col], df_clean[y_col], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(df_clean[x_col].min(), df_clean[x_col].max(), 100)
                            y_trend = p(x_trend)
                            
                            fig.add_trace(go.Scatter(
                                x=x_trend,
                                y=y_trend,
                                mode='lines',
                                name=f'Trend Line (r={correlation:.3f})',
                                line=dict(color=self.colors['correlation'], width=2, dash='dash'),
                                hoverinfo='skip'
                            ))
                except Exception as e:
                    self.logger.warning(f"Could not calculate correlation: {str(e)}")
            
            # Update layout with proper labels and formatting
            chart_title = title + correlation_text
            if show_data_quality and quality_message:
                chart_title += quality_message
                
            fig.update_layout(
                title={
                    'text': chart_title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': 'black'}
                },
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                **self.default_layout
            )
            
            self.logger.info(f"Created scatter plot with {len(df_clean)} data points")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {str(e)}")
            raise
    
    def create_combined_time_series(
        self,
        df: pd.DataFrame,
        title: str = "Rainfall and Delivery Data Over Time"
    ) -> go.Figure:
        """
        Create a combined time series chart showing both rainfall and delivery data.
        
        Implements visual differentiation requirement 4.2 by using distinct colors
        and separate y-axes for different data series.
        
        Args:
            df: DataFrame with 'date', 'precipitation_mm', and 'order_count' columns
            title: Chart title
            
        Returns:
            Plotly Figure object with dual y-axis time series
            
        Raises:
            ValueError: If required columns are missing
        """
        try:
            # Validate input data
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            required_columns = ['date', 'precipitation_mm', 'order_count']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Sort by date
            df_sorted = df.sort_values('date').copy()
            
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add rainfall data (primary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=df_sorted['date'],
                    y=df_sorted['precipitation_mm'],
                    mode='lines+markers',
                    name='Precipitation',
                    line=dict(color=self.colors['rainfall'], width=2),
                    marker=dict(size=4, color=self.colors['rainfall']),
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                 '<b>Precipitation:</b> %{y} mm<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=False,
            )
            
            # Add delivery data (secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=df_sorted['date'],
                    y=df_sorted['order_count'],
                    mode='lines+markers',
                    name='Delivery Orders',
                    line=dict(color=self.colors['delivery'], width=2),
                    marker=dict(size=4, color=self.colors['delivery']),
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                 '<b>Orders:</b> %{y}<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=True,
            )
            
            # Set x-axis title
            fig.update_xaxes(title_text="Date")
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=False)
            fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': 'black'}
                },
                **self.default_layout
            )
            
            self.logger.info(f"Created combined time series chart with {len(df_sorted)} data points")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating combined time series chart: {str(e)}")
            raise
    
    def calculate_correlation(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ) -> Dict[str, Any]:
        """
        Calculate correlation coefficient and related statistics.
        
        Implements requirement 4.3 for correlation coefficient calculation
        with proper handling of edge cases with insufficient data.
        
        Args:
            df: DataFrame containing the data
            x_col: Name of the first variable column
            y_col: Name of the second variable column
            
        Returns:
            Dictionary containing correlation statistics and metadata
        """
        try:
            # Validate input data
            if df.empty:
                return {
                    'correlation': None,
                    'p_value': None,
                    'sample_size': 0,
                    'is_significant': False,
                    'interpretation': 'No data available',
                    'error': 'Empty dataset'
                }
            
            required_columns = [x_col, y_col]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {
                    'correlation': None,
                    'p_value': None,
                    'sample_size': 0,
                    'is_significant': False,
                    'interpretation': 'Missing required columns',
                    'error': f'Missing columns: {missing_columns}'
                }
            
            # Remove rows with NaN values
            df_clean = df.dropna(subset=[x_col, y_col]).copy()
            sample_size = len(df_clean)
            
            # Handle insufficient data cases
            if sample_size < 2:
                return {
                    'correlation': None,
                    'p_value': None,
                    'sample_size': sample_size,
                    'is_significant': False,
                    'interpretation': 'Insufficient data for correlation analysis (need at least 2 data points)',
                    'error': 'Insufficient data'
                }
            
            # Calculate correlation
            correlation = df_clean[x_col].corr(df_clean[y_col])
            
            if np.isnan(correlation):
                return {
                    'correlation': None,
                    'p_value': None,
                    'sample_size': sample_size,
                    'is_significant': False,
                    'interpretation': 'Cannot calculate correlation (possibly constant values)',
                    'error': 'Correlation calculation failed'
                }
            
            # Calculate p-value using scipy if available, otherwise provide basic interpretation
            p_value = None
            try:
                from scipy.stats import pearsonr
                _, p_value = pearsonr(df_clean[x_col], df_clean[y_col])
            except ImportError:
                self.logger.warning("scipy not available for p-value calculation")
            
            # Determine significance (basic rule: |r| > 0.3 and n > 10)
            is_significant = abs(correlation) > 0.3 and sample_size > 10
            
            # Provide interpretation
            if abs(correlation) < 0.1:
                strength = "very weak"
            elif abs(correlation) < 0.3:
                strength = "weak"
            elif abs(correlation) < 0.5:
                strength = "moderate"
            elif abs(correlation) < 0.7:
                strength = "strong"
            else:
                strength = "very strong"
            
            direction = "positive" if correlation > 0 else "negative"
            interpretation = f"A {strength} {direction} correlation (r = {correlation:.3f})"
            
            if p_value is not None:
                interpretation += f" with p-value = {p_value:.3f}"
            
            return {
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': sample_size,
                'is_significant': is_significant,
                'interpretation': interpretation,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {str(e)}")
            return {
                'correlation': None,
                'p_value': None,
                'sample_size': 0,
                'is_significant': False,
                'interpretation': 'Error in correlation calculation',
                'error': str(e)
            }
    
    def create_summary_statistics_table(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate summary statistics table for specified columns.
        
        Implements requirement 5.1 for summary statistics display
        with descriptive statistics for both datasets.
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to analyze (all numeric columns if None)
            
        Returns:
            DataFrame with summary statistics
        """
        try:
            if df.empty:
                return pd.DataFrame()
            
            # Select columns to analyze
            if columns is None:
                # Auto-select numeric columns, excluding date
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                columns = [col for col in numeric_columns if col != 'date']
            
            if not columns:
                return pd.DataFrame()
            
            # Calculate summary statistics
            stats_dict = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    stats_dict[col] = {
                        'Count': 0,
                        'Mean': np.nan,
                        'Median': np.nan,
                        'Std Dev': np.nan,
                        'Min': np.nan,
                        'Max': np.nan,
                        'Q1': np.nan,
                        'Q3': np.nan
                    }
                else:
                    stats_dict[col] = {
                        'Count': len(col_data),
                        'Mean': col_data.mean(),
                        'Median': col_data.median(),
                        'Std Dev': col_data.std(),
                        'Min': col_data.min(),
                        'Max': col_data.max(),
                        'Q1': col_data.quantile(0.25),
                        'Q3': col_data.quantile(0.75)
                    }
            
            # Convert to DataFrame
            stats_df = pd.DataFrame(stats_dict).T
            
            # Round numeric values for display
            numeric_cols = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3']
            for col in numeric_cols:
                if col in stats_df.columns:
                    stats_df[col] = stats_df[col].round(2)
            
            self.logger.info(f"Generated summary statistics for {len(columns)} columns")
            return stats_df
            
        except Exception as e:
            self.logger.error(f"Error creating summary statistics table: {str(e)}")
            return pd.DataFrame()
    
    def generate_data_quality_report(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Dataset"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report for user-friendly display.
        
        Implements requirement 5.4 for data quality reporting with user-friendly
        messages about datasets with gaps and quality issues.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dictionary containing data quality information and user messages
        """
        try:
            if df.empty:
                return {
                    'dataset_name': dataset_name,
                    'total_records': 0,
                    'valid_records': 0,
                    'missing_data_summary': "Dataset is empty",
                    'date_coverage': "No data available",
                    'quality_issues': ["Dataset contains no records"],
                    'user_message': f"{dataset_name} is empty. Please check your data file.",
                    'quality_score': 0.0
                }
            
            report = {
                'dataset_name': dataset_name,
                'total_records': len(df),
                'valid_records': 0,
                'missing_data_summary': "",
                'date_coverage': "",
                'quality_issues': [],
                'user_message': "",
                'quality_score': 1.0
            }
            
            # Analyze missing data for each column
            missing_info = []
            for col in df.columns:
                if col == 'date':
                    continue
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    percentage = (missing_count / len(df)) * 100
                    missing_info.append(f"{col}: {missing_count} missing ({percentage:.1f}%)")
                    report['quality_issues'].append(f"Missing values in {col}")
            
            if missing_info:
                report['missing_data_summary'] = "; ".join(missing_info)
            else:
                report['missing_data_summary'] = "No missing values detected"
            
            # Analyze date coverage if date column exists
            if 'date' in df.columns:
                df_with_dates = df.dropna(subset=['date'])
                if len(df_with_dates) > 0:
                    start_date = df_with_dates['date'].min()
                    end_date = df_with_dates['date'].max()
                    total_days = (end_date - start_date).days + 1
                    actual_days = len(df_with_dates['date'].unique())
                    
                    report['date_coverage'] = f"{start_date} to {end_date} ({actual_days}/{total_days} days)"
                    
                    if actual_days < total_days:
                        missing_days = total_days - actual_days
                        report['quality_issues'].append(f"{missing_days} missing days in date range")
                else:
                    report['date_coverage'] = "No valid dates found"
                    report['quality_issues'].append("No valid dates in dataset")
            
            # Count valid records (non-null for all non-date columns)
            data_columns = [col for col in df.columns if col != 'date']
            if data_columns:
                report['valid_records'] = len(df.dropna(subset=data_columns))
            else:
                report['valid_records'] = len(df)
            
            # Calculate quality score
            if len(df) > 0:
                completeness = report['valid_records'] / len(df)
                report['quality_score'] = completeness
            
            # Generate user-friendly message
            if not report['quality_issues']:
                report['user_message'] = f"{dataset_name} looks good with {report['valid_records']} complete records."
            else:
                issue_count = len(report['quality_issues'])
                report['user_message'] = f"{dataset_name} has {issue_count} quality issue(s). {report['valid_records']} of {report['total_records']} records are complete."
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating data quality report: {str(e)}")
            return {
                'dataset_name': dataset_name,
                'total_records': 0,
                'valid_records': 0,
                'missing_data_summary': "Error analyzing data quality",
                'date_coverage': "Unknown",
                'quality_issues': [f"Analysis error: {str(e)}"],
                'user_message': f"Unable to analyze {dataset_name} quality due to an error.",
                'quality_score': 0.0
            }

    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """
        Create a correlation heatmap for multiple variables.
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to include (all numeric columns if None)
            title: Chart title
            
        Returns:
            Plotly Figure object with correlation heatmap
        """
        try:
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            # Select columns to analyze
            if columns is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                columns = [col for col in numeric_columns if col != 'date']
            
            if len(columns) < 2:
                raise ValueError("Need at least 2 numeric columns for correlation matrix")
            
            # Calculate correlation matrix
            corr_matrix = df[columns].corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': 'black'}
                },
                **self.default_layout
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            raise