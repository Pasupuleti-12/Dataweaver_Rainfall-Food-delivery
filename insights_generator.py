"""
Insights and observations generation for the Rainfall vs Food Delivery Demand Dashboard

This module provides the InsightsGenerator class for creating statistical summaries
and descriptive observations without causal language as specified in requirements 5.1-5.5.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import date


class InsightsGenerator:
    """
    Generates statistical summaries and descriptive insights for rainfall and delivery data.
    
    This class implements statistical analysis and reporting functionality as specified
    in requirements 5.1, 5.2, 5.3, 5.4, and 5.5 with careful attention to avoiding
    causal language and focusing on descriptive observations only.
    """
    
    def __init__(self):
        """Initialize the InsightsGenerator with logging configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Words and phrases to avoid (causal language)
        self.causal_words = {
            'causes', 'caused', 'causing', 'because', 'due to', 'results in',
            'leads to', 'triggers', 'influences', 'affects', 'impacts',
            'drives', 'determines', 'produces', 'creates', 'makes',
            'forces', 'brings about', 'generates', 'induces', 'provokes',
            'stems from', 'originates from', 'arises from', 'depends on',
            'is responsible for', 'accounts for', 'explains', 'predicts'
        }
        
        # Preferred descriptive language
        self.descriptive_phrases = [
            'is associated with', 'tends to occur alongside', 'is observed together with',
            'shows a pattern of', 'demonstrates a relationship with', 'exhibits',
            'displays', 'shows', 'indicates', 'suggests a correlation with',
            'appears to coincide with', 'is found in combination with'
        ]
    
    def generate_summary_statistics(
        self, 
        df: pd.DataFrame, 
        dataset_name: str = "Dataset"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for a dataset.
        
        Implements requirement 5.1 for summary statistics calculation for both datasets
        with descriptive statistics and data quality information.
        
        Args:
            df: DataFrame containing the data to analyze
            dataset_name: Name of the dataset for reporting purposes
            
        Returns:
            Dictionary containing summary statistics and metadata
        """
        try:
            if df.empty:
                return {
                    'dataset_name': dataset_name,
                    'record_count': 0,
                    'date_range': None,
                    'statistics': {},
                    'data_quality': {
                        'completeness': 0.0,
                        'issues': ['Dataset is empty'],
                        'message': f'{dataset_name} contains no records.'
                    },
                    'observations': [f'{dataset_name} is empty and cannot be analyzed.']
                }
            
            summary = {
                'dataset_name': dataset_name,
                'record_count': len(df),
                'date_range': None,
                'statistics': {},
                'data_quality': {},
                'observations': []
            }
            
            # Analyze date range if date column exists
            if 'date' in df.columns:
                date_data = df['date'].dropna()
                if len(date_data) > 0:
                    start_date = date_data.min()
                    end_date = date_data.max()
                    total_days = (end_date - start_date).days + 1
                    actual_days = len(date_data.unique())
                    
                    summary['date_range'] = {
                        'start': start_date,
                        'end': end_date,
                        'total_days': total_days,
                        'actual_days': actual_days,
                        'coverage_percentage': (actual_days / total_days) * 100 if total_days > 0 else 0
                    }
            
            # Calculate statistics for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'date' in numeric_columns:
                numeric_columns.remove('date')
            
            # Also include expected numeric columns even if they're all NaN
            expected_numeric_cols = ['precipitation_mm', 'order_count']
            for col in expected_numeric_cols:
                if col in df.columns and col not in numeric_columns:
                    numeric_columns.append(col)
            
            for col in numeric_columns:
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    summary['statistics'][col] = {
                        'count': 0,
                        'mean': None,
                        'median': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'q1': None,
                        'q3': None,
                        'missing_count': len(df) - len(col_data),
                        'missing_percentage': 100.0
                    }
                else:
                    missing_count = len(df) - len(col_data)
                    summary['statistics'][col] = {
                        'count': len(col_data),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()) if len(col_data) > 1 else 0.0,
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'q1': float(col_data.quantile(0.25)),
                        'q3': float(col_data.quantile(0.75)),
                        'missing_count': missing_count,
                        'missing_percentage': (missing_count / len(df)) * 100
                    }
            
            # Generate data quality assessment
            summary['data_quality'] = self._assess_data_quality(df, dataset_name)
            
            # Generate descriptive observations
            summary['observations'] = self._generate_descriptive_observations(df, summary['statistics'], dataset_name)
            
            self.logger.info(f"Generated summary statistics for {dataset_name} with {len(df)} records")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary statistics for {dataset_name}: {str(e)}")
            return {
                'dataset_name': dataset_name,
                'record_count': 0,
                'date_range': None,
                'statistics': {},
                'data_quality': {
                    'completeness': 0.0,
                    'issues': [f'Analysis error: {str(e)}'],
                    'message': f'Unable to analyze {dataset_name} - technical error encountered.'
                },
                'observations': [f'Analysis of {dataset_name} failed - technical issues encountered.']
            }
    
    def generate_combined_insights(
        self,
        combined_df: pd.DataFrame,
        rainfall_summary: Dict[str, Any],
        delivery_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate insights for combined rainfall and delivery data.
        
        Implements requirements 5.1, 5.2, 5.3 for combined analysis with correlation
        information and descriptive observations without causal implications.
        
        Args:
            combined_df: DataFrame with both rainfall and delivery data
            rainfall_summary: Summary statistics for rainfall data
            delivery_summary: Summary statistics for delivery data
            
        Returns:
            Dictionary containing combined insights and observations
        """
        try:
            if combined_df.empty:
                return {
                    'correlation_analysis': None,
                    'combined_statistics': {},
                    'relationship_observations': ['No combined data available for relationship analysis.'],
                    'data_quality_notes': ['Combined dataset is empty.'],
                    'summary_message': 'No data available for combined analysis.'
                }
            
            insights = {
                'correlation_analysis': None,
                'combined_statistics': {},
                'relationship_observations': [],
                'data_quality_notes': [],
                'summary_message': ''
            }
            
            # Calculate correlation if both columns exist
            if 'precipitation_mm' in combined_df.columns and 'order_count' in combined_df.columns:
                insights['correlation_analysis'] = self._calculate_correlation_insights(
                    combined_df, 'precipitation_mm', 'order_count'
                )
            
            # Generate combined statistics
            insights['combined_statistics'] = self.generate_summary_statistics(combined_df, "Combined Dataset")
            
            # Generate relationship observations (descriptive only)
            insights['relationship_observations'] = self._generate_relationship_observations(
                combined_df, rainfall_summary, delivery_summary
            )
            
            # Compile data quality notes
            insights['data_quality_notes'] = self._compile_data_quality_notes(
                combined_df, rainfall_summary, delivery_summary
            )
            
            # Generate summary message
            insights['summary_message'] = self._generate_summary_message(
                combined_df, insights['correlation_analysis']
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating combined insights: {str(e)}")
            return {
                'correlation_analysis': None,
                'combined_statistics': {},
                'relationship_observations': [f'Analysis failed - technical issues encountered: {str(e)}'],
                'data_quality_notes': ['Unable to assess data quality - analysis error occurred.'],
                'summary_message': 'Combined analysis could not be completed.'
            }
    
    def _assess_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Assess data quality and generate user-friendly quality report.
        
        Implements requirement 5.4 for data quality limitation reporting.
        """
        if df.empty:
            return {
                'completeness': 0.0,
                'issues': ['Dataset is empty'],
                'message': f'{dataset_name} contains no data.'
            }
        
        quality_issues = []
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                quality_issues.append(f'{col}: {missing_count} missing values ({percentage:.1f}%)')
        
        # Check for date gaps if date column exists
        if 'date' in df.columns:
            date_data = df['date'].dropna()
            if len(date_data) > 1:
                start_date = date_data.min()
                end_date = date_data.max()
                expected_days = (end_date - start_date).days + 1
                actual_days = len(date_data.unique())
                
                if actual_days < expected_days:
                    missing_days = expected_days - actual_days
                    quality_issues.append(f'Date gaps: {missing_days} missing days in the time range')
        
        # Generate quality message
        if not quality_issues:
            message = f'{dataset_name} shows excellent data quality with no missing values or gaps.'
        else:
            issue_count = len(quality_issues)
            if completeness >= 90:
                quality_level = 'very good'
            elif completeness >= 75:
                quality_level = 'good'
            elif completeness >= 50:
                quality_level = 'fair'
            else:
                quality_level = 'poor'
            
            message = f'{dataset_name} shows {quality_level} data quality ({completeness:.1f}% complete) with {issue_count} issue(s) identified.'
        
        return {
            'completeness': completeness,
            'issues': quality_issues,
            'message': message
        }
    
    def _generate_descriptive_observations(
        self, 
        df: pd.DataFrame, 
        statistics: Dict[str, Any], 
        dataset_name: str
    ) -> List[str]:
        """
        Generate descriptive observations about the data patterns.
        
        Implements requirement 5.3 for descriptive-only insights without causal language.
        """
        observations = []
        
        if df.empty or not statistics:
            observations.append(f'{dataset_name} contains no data for analysis.')
            return observations
        
        # Analyze each numeric column
        for col, stats in statistics.items():
            if stats['count'] == 0:
                observations.append(f'{col.replace("_", " ").title()} data is completely missing.')
                continue
            
            col_name = col.replace('_', ' ').title()
            
            # Basic descriptive statistics
            if col == 'precipitation_mm':
                unit = 'mm'
                observations.append(
                    f'Precipitation measurements range from {stats["min"]:.1f}{unit} to {stats["max"]:.1f}{unit}, '
                    f'with an average of {stats["mean"]:.1f}{unit}.'
                )
                
                # Describe distribution characteristics
                if stats['std'] > 0:
                    cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
                    if cv < 0.3:
                        variability = 'relatively consistent'
                    elif cv < 0.7:
                        variability = 'moderately variable'
                    else:
                        variability = 'highly variable'
                    
                    observations.append(f'Precipitation shows {variability} patterns across the time period.')
                
            elif col == 'order_count':
                observations.append(
                    f'Delivery orders range from {int(stats["min"])} to {int(stats["max"])}, '
                    f'with an average of {stats["mean"]:.0f} orders per day.'
                )
                
                # Describe order volume patterns
                if stats['std'] > 0:
                    cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
                    if cv < 0.2:
                        pattern = 'stable order volumes'
                    elif cv < 0.5:
                        pattern = 'moderate fluctuations in order volumes'
                    else:
                        pattern = 'significant variations in order volumes'
                    
                    observations.append(f'The delivery data shows {pattern} throughout the period.')
            
            # Note missing data if significant
            if stats['missing_percentage'] > 5:
                observations.append(
                    f'{col_name} has {stats["missing_count"]} missing values '
                    f'({stats["missing_percentage"]:.1f}% of records).'
                )
        
        # Add date range observation if available
        if 'date' in df.columns:
            date_data = df['date'].dropna()
            if len(date_data) > 0:
                start_date = date_data.min()
                end_date = date_data.max()
                days_span = (end_date - start_date).days + 1
                
                observations.append(
                    f'The dataset covers {days_span} days from {start_date} to {end_date}.'
                )
        
        return observations
    
    def _calculate_correlation_insights(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str
    ) -> Dict[str, Any]:
        """
        Calculate correlation and generate descriptive insights about the relationship.
        
        Implements requirement 5.2 to avoid causal language while describing correlations.
        """
        try:
            # Remove rows with missing values
            clean_df = df.dropna(subset=[x_col, y_col])
            
            if len(clean_df) < 2:
                return {
                    'correlation': None,
                    'sample_size': len(clean_df),
                    'interpretation': 'Insufficient data for correlation analysis.',
                    'descriptive_text': 'Not enough complete data pairs to analyze the relationship.',
                    'strength': 'unknown'
                }
            
            # Calculate correlation
            correlation = clean_df[x_col].corr(clean_df[y_col])
            
            if pd.isna(correlation):
                return {
                    'correlation': None,
                    'sample_size': len(clean_df),
                    'interpretation': 'Correlation cannot be calculated.',
                    'descriptive_text': 'The relationship between variables cannot be quantified.',
                    'strength': 'unknown'
                }
            
            # Determine correlation strength (avoiding causal language)
            abs_corr = abs(correlation)
            if abs_corr < 0.1:
                strength = 'very weak'
                relationship_desc = 'shows very little association'
            elif abs_corr < 0.3:
                strength = 'weak'
                relationship_desc = 'shows a weak association'
            elif abs_corr < 0.5:
                strength = 'moderate'
                relationship_desc = 'shows a moderate association'
            elif abs_corr < 0.7:
                strength = 'strong'
                relationship_desc = 'shows a strong association'
            else:
                strength = 'very strong'
                relationship_desc = 'shows a very strong association'
            
            # Determine direction
            direction = 'positive' if correlation > 0 else 'negative'
            
            # Generate descriptive text (avoiding causal language)
            if abs_corr < 0.1:
                descriptive_text = (
                    f'Rainfall and delivery orders appear to vary independently, '
                    f'with no clear pattern of association (r = {correlation:.3f}).'
                )
            else:
                descriptive_text = (
                    f'Rainfall and delivery orders {relationship_desc} '
                    f'(r = {correlation:.3f}). When one tends to be higher, '
                    f'the other tends to be {"higher" if correlation > 0 else "lower"} as well.'
                )
            
            return {
                'correlation': correlation,
                'sample_size': len(clean_df),
                'interpretation': f'A {strength} {direction} correlation',
                'descriptive_text': descriptive_text,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation insights: {str(e)}")
            return {
                'correlation': None,
                'sample_size': 0,
                'interpretation': 'Correlation analysis failed.',
                'descriptive_text': 'Unable to analyze the relationship between variables.',
                'strength': 'unknown'
            }
    
    def _generate_relationship_observations(
        self,
        combined_df: pd.DataFrame,
        rainfall_summary: Dict[str, Any],
        delivery_summary: Dict[str, Any]
    ) -> List[str]:
        """
        Generate observations about relationships between datasets.
        
        Implements requirement 5.3 for descriptive observations without causal implications.
        """
        observations = []
        
        if combined_df.empty:
            observations.append('No overlapping dates found between rainfall and delivery datasets.')
            return observations
        
        # Basic overlap information
        total_rainfall_days = rainfall_summary.get('record_count', 0)
        total_delivery_days = delivery_summary.get('record_count', 0)
        common_days = len(combined_df)
        
        observations.append(
            f'Analysis includes {common_days} days with both rainfall and delivery data, '
            f'from {total_rainfall_days} rainfall records and {total_delivery_days} delivery records.'
        )
        
        # Describe patterns in the combined data (avoiding causal language)
        if 'precipitation_mm' in combined_df.columns and 'order_count' in combined_df.columns:
            precip_data = combined_df['precipitation_mm'].dropna()
            order_data = combined_df['order_count'].dropna()
            
            if len(precip_data) > 0 and len(order_data) > 0:
                # Find days with high/low values
                high_rain_threshold = precip_data.quantile(0.75)
                low_rain_threshold = precip_data.quantile(0.25)
                high_orders_threshold = order_data.quantile(0.75)
                low_orders_threshold = order_data.quantile(0.25)
                
                high_rain_days = combined_df[combined_df['precipitation_mm'] >= high_rain_threshold]
                low_rain_days = combined_df[combined_df['precipitation_mm'] <= low_rain_threshold]
                
                if len(high_rain_days) > 0:
                    avg_orders_high_rain = high_rain_days['order_count'].mean()
                    observations.append(
                        f'On days with higher rainfall (≥{high_rain_threshold:.1f}mm), '
                        f'delivery orders averaged {avg_orders_high_rain:.0f} per day.'
                    )
                
                if len(low_rain_days) > 0:
                    avg_orders_low_rain = low_rain_days['order_count'].mean()
                    observations.append(
                        f'On days with lower rainfall (≤{low_rain_threshold:.1f}mm), '
                        f'delivery orders averaged {avg_orders_low_rain:.0f} per day.'
                    )
        
        return observations
    
    def _compile_data_quality_notes(
        self,
        combined_df: pd.DataFrame,
        rainfall_summary: Dict[str, Any],
        delivery_summary: Dict[str, Any]
    ) -> List[str]:
        """
        Compile data quality limitations for the combined analysis.
        
        Implements requirement 5.4 for data quality limitation reporting.
        """
        notes = []
        
        # Note limitations from individual datasets
        rainfall_quality = rainfall_summary.get('data_quality', {})
        delivery_quality = delivery_summary.get('data_quality', {})
        
        if rainfall_quality.get('issues'):
            notes.append(f"Rainfall data limitations: {'; '.join(rainfall_quality['issues'])}")
        
        if delivery_quality.get('issues'):
            notes.append(f"Delivery data limitations: {'; '.join(delivery_quality['issues'])}")
        
        # Note limitations specific to combined analysis
        if combined_df.empty:
            notes.append('No overlapping dates between datasets limits relationship analysis.')
        else:
            total_possible = max(
                rainfall_summary.get('record_count', 0),
                delivery_summary.get('record_count', 0)
            )
            actual_combined = len(combined_df)
            
            if actual_combined < total_possible * 0.5:
                notes.append(
                    f'Limited overlap ({actual_combined} of {total_possible} possible days) '
                    'may affect relationship analysis reliability.'
                )
        
        # Add general limitations note
        notes.append(
            'Analysis is descriptive only and does not establish causal relationships '
            'between rainfall and delivery demand.'
        )
        
        return notes
    
    def _generate_summary_message(
        self,
        combined_df: pd.DataFrame,
        correlation_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate a concise summary message for the overall analysis.
        
        Implements requirement 5.5 to ensure non-causal language in summary.
        """
        if combined_df.empty:
            return (
                'No overlapping data available for relationship analysis. '
                'Individual dataset summaries provide insights into rainfall and delivery patterns separately.'
            )
        
        days_analyzed = len(combined_df)
        
        if correlation_analysis and correlation_analysis.get('correlation') is not None:
            corr_strength = correlation_analysis.get('strength', 'unknown')
            
            if corr_strength in ['very weak', 'weak']:
                relationship_summary = 'show little association'
            elif corr_strength == 'moderate':
                relationship_summary = 'show a moderate association'
            else:
                relationship_summary = 'show a notable association'
            
            return (
                f'Analysis of {days_analyzed} days reveals that rainfall and delivery orders '
                f'{relationship_summary}. This descriptive analysis provides insights into '
                'patterns but does not establish causal relationships.'
            )
        else:
            return (
                f'Analysis covers {days_analyzed} days of combined data. '
                'Individual patterns in rainfall and delivery orders are described separately. '
                'Relationship analysis was not possible with the available data.'
            )
    
    def validate_non_causal_language(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate that text does not contain causal language.
        
        Implements requirement 5.2 and 5.5 for non-causal language enforcement.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, list_of_causal_words_found)
        """
        text_lower = text.lower()
        found_causal_words = []
        
        for causal_word in self.causal_words:
            if causal_word in text_lower:
                found_causal_words.append(causal_word)
        
        is_valid = len(found_causal_words) == 0
        return is_valid, found_causal_words
    
    def format_insights_for_display(self, insights: Dict[str, Any]) -> str:
        """
        Format insights dictionary into user-friendly display text.
        
        Args:
            insights: Dictionary containing analysis results
            
        Returns:
            Formatted string for display in the dashboard
        """
        try:
            if not insights:
                return "No insights available."
            
            sections = []
            
            # Add correlation analysis if available
            if insights.get('correlation_analysis'):
                corr_info = insights['correlation_analysis']
                sections.append("**Relationship Analysis**")
                sections.append(corr_info.get('descriptive_text', 'No relationship information available.'))
                sections.append("")
            
            # Add relationship observations
            if insights.get('relationship_observations'):
                sections.append("**Data Patterns**")
                for obs in insights['relationship_observations']:
                    sections.append(f"• {obs}")
                sections.append("")
            
            # Add data quality notes
            if insights.get('data_quality_notes'):
                sections.append("**Data Quality Notes**")
                for note in insights['data_quality_notes']:
                    sections.append(f"• {note}")
                sections.append("")
            
            # Add summary message
            if insights.get('summary_message'):
                sections.append("**Summary**")
                sections.append(insights['summary_message'])
            
            return "\n".join(sections)
            
        except Exception as e:
            self.logger.error(f"Error formatting insights for display: {str(e)}")
            return "Unable to format insights - technical issues encountered."