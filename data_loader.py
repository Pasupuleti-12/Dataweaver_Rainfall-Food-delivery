"""
Data loading and processing components for the Rainfall vs Food Delivery Demand Dashboard

This module provides the DataLoader class for handling CSV file loading, validation,
and dataset joining operations as specified in the requirements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict, Any
import logging
from pathlib import Path

from models import RainfallRecord, DeliveryRecord, CombinedDataPoint, create_combined_data_point


class DataLoader:
    """
    Handles loading, validation, and preprocessing of rainfall and delivery datasets.
    
    This class implements CSV file handling with robust date parsing, data validation,
    and error handling as specified in requirements 3.1, 3.3, and 3.5.
    """
    
    def __init__(self):
        """Initialize the DataLoader with logging configuration."""
        self.logger = logging.getLogger(__name__)
        
    def _parse_date(self, date_str: Any) -> Optional[date]:
        """
        Parse various date formats into a standardized date object.
        
        Supports common date formats including:
        - YYYY-MM-DD
        - MM/DD/YYYY  
        - DD/MM/YYYY
        - YYYY/MM/DD
        - ISO format with time components
        
        Args:
            date_str: Date string or date object to parse
            
        Returns:
            date object if parsing successful, None otherwise
        """
        if pd.isna(date_str):
            return None
            
        if isinstance(date_str, date):
            return date_str
            
        if isinstance(date_str, datetime):
            return date_str.date()
            
        if not isinstance(date_str, str):
            date_str = str(date_str)
            
        # Common date formats to try (ordered by preference to avoid ambiguity)
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%m/%d/%Y',
            '%m/%d/%Y %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt).date()
                return parsed_date
            except (ValueError, AttributeError):
                continue
                
        # Try pandas date parsing as fallback
        try:
            parsed_date = pd.to_datetime(date_str).date()
            return parsed_date
        except (ValueError, TypeError):
            self.logger.warning(f"Failed to parse date: {date_str}")
            return None
    
    def load_rainfall_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate rainfall data from CSV file.
        
        Expected CSV format:
        - date column: Date in various supported formats
        - precipitation_mm column: Daily precipitation in millimeters (>= 0)
        
        Args:
            file_path: Path to the rainfall CSV file
            
        Returns:
            DataFrame with validated rainfall data, invalid records excluded
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid or required columns are missing
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Rainfall data file not found: {file_path}")
            
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Check for required columns (case-insensitive)
            df.columns = df.columns.str.lower().str.strip()
            required_cols = ['date', 'precipitation_mm']
            
            # Try alternative column names
            column_mapping = {
                'precipitation': 'precipitation_mm',
                'rainfall': 'precipitation_mm',
                'rain': 'precipitation_mm',
                'precip': 'precipitation_mm'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in rainfall data: {missing_cols}")
            
            # Parse and normalize dates
            df['parsed_date'] = df['date'].apply(self._parse_date)
            
            # Filter out rows with unparseable dates
            valid_date_mask = df['parsed_date'].notna()
            invalid_date_count = (~valid_date_mask).sum()
            if invalid_date_count > 0:
                self.logger.warning(f"Excluded {invalid_date_count} records with invalid dates")
            
            df = df[valid_date_mask].copy()
            
            # Validate precipitation values
            df['precipitation_mm'] = pd.to_numeric(df['precipitation_mm'], errors='coerce')
            
            # Filter out invalid precipitation values (negative or NaN)
            valid_precip_mask = (df['precipitation_mm'] >= 0) & df['precipitation_mm'].notna()
            invalid_precip_count = (~valid_precip_mask).sum()
            if invalid_precip_count > 0:
                self.logger.warning(f"Excluded {invalid_precip_count} records with invalid precipitation values")
            
            df = df[valid_precip_mask].copy()
            
            # Create final cleaned dataset
            result_df = pd.DataFrame({
                'date': df['parsed_date'],
                'precipitation_mm': df['precipitation_mm']
            })
            
            # Remove duplicates, keeping the first occurrence
            result_df = result_df.drop_duplicates(subset=['date'], keep='first')
            
            # Sort by date
            result_df = result_df.sort_values('date').reset_index(drop=True)
            
            self.logger.info(f"Successfully loaded {len(result_df)} rainfall records from {file_path}")
            return result_df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty or invalid CSV file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading rainfall data from {file_path}: {str(e)}")
            raise
    
    def load_delivery_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate delivery data from CSV file.
        
        Expected CSV format:
        - date column: Date in various supported formats  
        - order_count column: Number of delivery orders (>= 0, integer)
        
        Args:
            file_path: Path to the delivery CSV file
            
        Returns:
            DataFrame with validated delivery data, invalid records excluded
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid or required columns are missing
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Delivery data file not found: {file_path}")
            
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Check for required columns (case-insensitive)
            df.columns = df.columns.str.lower().str.strip()
            required_cols = ['date', 'order_count']
            
            # Try alternative column names
            column_mapping = {
                'orders': 'order_count',
                'deliveries': 'order_count',
                'delivery_count': 'order_count',
                'count': 'order_count'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in delivery data: {missing_cols}")
            
            # Parse and normalize dates
            df['parsed_date'] = df['date'].apply(self._parse_date)
            
            # Filter out rows with unparseable dates
            valid_date_mask = df['parsed_date'].notna()
            invalid_date_count = (~valid_date_mask).sum()
            if invalid_date_count > 0:
                self.logger.warning(f"Excluded {invalid_date_count} records with invalid dates")
            
            df = df[valid_date_mask].copy()
            
            # Validate order count values
            df['order_count'] = pd.to_numeric(df['order_count'], errors='coerce')
            
            # Convert to integer and filter out invalid values (negative, NaN, or non-integer)
            # First filter out NaN values, then check if remaining values are integers
            non_nan_mask = df['order_count'].notna()
            non_negative_mask = df['order_count'] >= 0
            
            # For non-NaN values, check if they are integers (whole numbers)
            integer_mask = df['order_count'].fillna(0) == df['order_count'].fillna(0).astype(int)
            
            valid_orders_mask = non_nan_mask & non_negative_mask & integer_mask
            
            invalid_orders_count = (~valid_orders_mask).sum()
            if invalid_orders_count > 0:
                self.logger.warning(f"Excluded {invalid_orders_count} records with invalid order counts")
            
            df = df[valid_orders_mask].copy()
            df['order_count'] = df['order_count'].astype(int)
            
            # Create final cleaned dataset
            result_df = pd.DataFrame({
                'date': df['parsed_date'],
                'order_count': df['order_count']
            })
            
            # Remove duplicates, keeping the first occurrence
            result_df = result_df.drop_duplicates(subset=['date'], keep='first')
            
            # Sort by date
            result_df = result_df.sort_values('date').reset_index(drop=True)
            
            self.logger.info(f"Successfully loaded {len(result_df)} delivery records from {file_path}")
            return result_df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty or invalid CSV file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading delivery data from {file_path}: {str(e)}")
            raise
    
    def join_datasets(self, rainfall_df: pd.DataFrame, delivery_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join rainfall and delivery datasets on date, preserving data integrity.
        
        Performs an inner join to only include dates present in both datasets.
        Validates data integrity after joining and handles edge cases.
        
        Args:
            rainfall_df: DataFrame with rainfall data (date, precipitation_mm columns)
            delivery_df: DataFrame with delivery data (date, order_count columns)
            
        Returns:
            DataFrame with combined data for common dates
            
        Raises:
            ValueError: If input DataFrames are invalid or have no common dates
        """
        try:
            # Validate input DataFrames
            if rainfall_df.empty and delivery_df.empty:
                raise ValueError("Both datasets are empty")
            
            if rainfall_df.empty:
                self.logger.warning("Rainfall dataset is empty")
                return pd.DataFrame(columns=['date', 'precipitation_mm', 'order_count'])
            
            if delivery_df.empty:
                self.logger.warning("Delivery dataset is empty")
                return pd.DataFrame(columns=['date', 'precipitation_mm', 'order_count'])
            
            # Check required columns
            rainfall_required = ['date', 'precipitation_mm']
            delivery_required = ['date', 'order_count']
            
            missing_rainfall = [col for col in rainfall_required if col not in rainfall_df.columns]
            missing_delivery = [col for col in delivery_required if col not in delivery_df.columns]
            
            if missing_rainfall:
                raise ValueError(f"Missing columns in rainfall data: {missing_rainfall}")
            if missing_delivery:
                raise ValueError(f"Missing columns in delivery data: {missing_delivery}")
            
            # Perform inner join on date
            joined_df = pd.merge(
                rainfall_df[['date', 'precipitation_mm']], 
                delivery_df[['date', 'order_count']], 
                on='date', 
                how='inner'
            )
            
            if joined_df.empty:
                self.logger.warning("No common dates found between datasets")
                return pd.DataFrame(columns=['date', 'precipitation_mm', 'order_count'])
            
            # Validate data integrity after joining
            # Check for any NaN values that shouldn't exist
            if joined_df.isnull().any().any():
                self.logger.warning("Found unexpected NaN values after joining, removing affected rows")
                joined_df = joined_df.dropna()
            
            # Validate that all values are still within expected ranges
            invalid_precip = (joined_df['precipitation_mm'] < 0).sum()
            invalid_orders = (joined_df['order_count'] < 0).sum()
            
            if invalid_precip > 0:
                self.logger.warning(f"Found {invalid_precip} invalid precipitation values after joining")
                joined_df = joined_df[joined_df['precipitation_mm'] >= 0]
            
            if invalid_orders > 0:
                self.logger.warning(f"Found {invalid_orders} invalid order counts after joining")
                joined_df = joined_df[joined_df['order_count'] >= 0]
            
            # Ensure chronological order
            joined_df = joined_df.sort_values('date').reset_index(drop=True)
            
            # Log join statistics
            rainfall_dates = set(rainfall_df['date'])
            delivery_dates = set(delivery_df['date'])
            common_dates = rainfall_dates.intersection(delivery_dates)
            
            self.logger.info(f"Dataset join completed:")
            self.logger.info(f"  Rainfall records: {len(rainfall_df)}")
            self.logger.info(f"  Delivery records: {len(delivery_df)}")
            self.logger.info(f"  Common dates: {len(common_dates)}")
            self.logger.info(f"  Final joined records: {len(joined_df)}")
            
            return joined_df
            
        except Exception as e:
            self.logger.error(f"Error joining datasets: {str(e)}")
            raise
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a data quality report for a dataset with user-friendly messages.
        
        Implements requirement 5.4 for data quality reporting with clear
        information about datasets with gaps and quality issues.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing data quality metrics and user-friendly messages
        """
        if df.empty:
            return {
                'total_records': 0,
                'valid_records': 0,
                'date_range': None,
                'missing_values': {},
                'data_gaps': [],
                'quality_score': 0.0,
                'issues': ['Dataset is empty'],
                'user_message': 'The dataset is empty. Please check your data file.',
                'chronological_order': True
            }
        
        quality_report = {
            'total_records': len(df),
            'valid_records': len(df),
            'date_range': None,
            'missing_values': {},
            'data_gaps': [],
            'quality_score': 1.0,
            'issues': [],
            'user_message': '',
            'chronological_order': True
        }
        
        # Analyze date range if date column exists
        if 'date' in df.columns:
            df_with_dates = df.dropna(subset=['date'])
            if len(df_with_dates) > 0:
                quality_report['date_range'] = {
                    'start': df_with_dates['date'].min(),
                    'end': df_with_dates['date'].max(),
                    'span_days': (df_with_dates['date'].max() - df_with_dates['date'].min()).days + 1,
                    'actual_days': len(df_with_dates['date'].unique())
                }
                
                # Check chronological order
                df_sorted = df_with_dates.sort_values('date')
                is_chronological = df_with_dates['date'].equals(df_sorted['date'])
                quality_report['chronological_order'] = is_chronological
                
                if not is_chronological:
                    quality_report['issues'].append("Data is not in chronological order")
        
        # Check for missing values in each column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                quality_report['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': percentage
                }
                quality_report['issues'].append(f"Missing values in {col}: {missing_count} ({percentage:.1f}%)")
        
        # Check for date gaps (if we have date column and date range)
        if quality_report['date_range'] and len(df) > 1:
            span_days = quality_report['date_range']['span_days']
            actual_days = quality_report['date_range']['actual_days']
            
            if actual_days < span_days:
                missing_days = span_days - actual_days
                quality_report['data_gaps'] = list(range(min(missing_days, 10)))  # Placeholder for gap analysis
                quality_report['issues'].append(f"Date gaps found: {missing_days} missing days in the date range")
        
        # Count valid records (complete records with no missing values in data columns)
        data_columns = [col for col in df.columns if col != 'date']
        if data_columns:
            quality_report['valid_records'] = len(df.dropna(subset=data_columns))
        
        # Calculate overall quality score
        if quality_report['date_range']:
            # Score based on date completeness and data completeness
            date_completeness = quality_report['date_range']['actual_days'] / quality_report['date_range']['span_days']
            data_completeness = quality_report['valid_records'] / len(df)
            quality_report['quality_score'] = (date_completeness + data_completeness) / 2
        else:
            # Score based on data completeness only
            quality_report['quality_score'] = quality_report['valid_records'] / len(df)
        
        # Generate user-friendly message
        if not quality_report['issues']:
            quality_report['user_message'] = f"Data quality is excellent! {quality_report['valid_records']} complete records with no issues detected."
        else:
            issue_count = len(quality_report['issues'])
            quality_score_pct = quality_report['quality_score'] * 100
            
            if quality_score_pct >= 90:
                quality_level = "very good"
            elif quality_score_pct >= 70:
                quality_level = "good"
            elif quality_score_pct >= 50:
                quality_level = "fair"
            else:
                quality_level = "poor"
            
            quality_report['user_message'] = (
                f"Data quality is {quality_level} ({quality_score_pct:.1f}%). "
                f"Found {issue_count} issue(s) affecting {len(df) - quality_report['valid_records']} records. "
                f"Visualizations will use the {quality_report['valid_records']} complete records."
            )
        
        return quality_report
    
    def ensure_chronological_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure data is in chronological order as required by requirements 1.4, 2.4.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame sorted by date in chronological order
        """
        if df.empty or 'date' not in df.columns:
            return df
        
        # Sort by date and reset index
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        self.logger.info(f"Ensured chronological ordering for {len(df_sorted)} records")
        return df_sorted