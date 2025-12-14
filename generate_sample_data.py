#!/usr/bin/env python3
"""
Sample data generator for the Rainfall vs Food Delivery Demand Dashboard

This script generates comprehensive sample datasets for testing and demonstration
purposes, including various patterns, missing values, and edge cases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import random
import math
from pathlib import Path


class SampleDataGenerator:
    """
    Generates realistic sample datasets for rainfall and delivery data with various patterns.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
    
    def generate_rainfall_data(
        self, 
        start_date: date, 
        end_date: date, 
        seasonal_pattern: bool = True,
        missing_rate: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate realistic rainfall data with seasonal patterns and variability.
        
        Args:
            start_date: Start date for the data
            end_date: End date for the data
            seasonal_pattern: Whether to include seasonal rainfall patterns
            missing_rate: Proportion of missing values to introduce (0.0 to 1.0)
            
        Returns:
            DataFrame with date and precipitation_mm columns
        """
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        rainfall_data = []
        
        for current_date in date_range:
            # Base rainfall probability and intensity based on season
            if seasonal_pattern:
                # Higher rainfall in winter/spring, lower in summer/fall
                day_of_year = current_date.timetuple().tm_yday
                seasonal_factor = 0.7 + 0.6 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
            else:
                seasonal_factor = 1.0
            
            # Probability of rain on any given day (adjusted by season)
            rain_probability = 0.3 * seasonal_factor
            
            if np.random.random() < rain_probability:
                # Generate rainfall amount with realistic distribution
                # Most days have light rain, occasional heavy rain
                if np.random.random() < 0.7:
                    # Light rain (0.1 - 10mm)
                    precipitation = np.random.exponential(3.0) + 0.1
                elif np.random.random() < 0.9:
                    # Moderate rain (10 - 30mm)
                    precipitation = np.random.normal(20, 5)
                else:
                    # Heavy rain (30 - 80mm)
                    precipitation = np.random.normal(50, 15)
                
                # Ensure non-negative values
                precipitation = max(0.0, precipitation)
                
                # Round to 1 decimal place
                precipitation = round(precipitation, 1)
            else:
                # No rain
                precipitation = 0.0
            
            rainfall_data.append({
                'date': current_date.date(),
                'precipitation_mm': precipitation
            })
        
        df = pd.DataFrame(rainfall_data)
        
        # Introduce missing values randomly
        if missing_rate > 0:
            missing_indices = np.random.choice(
                len(df), 
                size=int(len(df) * missing_rate), 
                replace=False
            )
            df.loc[missing_indices, 'precipitation_mm'] = np.nan
        
        return df
    
    def generate_delivery_data(
        self, 
        start_date: date, 
        end_date: date, 
        seasonal_pattern: bool = True,
        weekly_pattern: bool = True,
        missing_rate: float = 0.03
    ) -> pd.DataFrame:
        """
        Generate realistic delivery order data with seasonal and weekly patterns.
        
        Args:
            start_date: Start date for the data
            end_date: End date for the data
            seasonal_pattern: Whether to include seasonal demand patterns
            weekly_pattern: Whether to include day-of-week patterns
            missing_rate: Proportion of missing values to introduce
            
        Returns:
            DataFrame with date and order_count columns
        """
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        delivery_data = []
        
        # Base order count (average daily orders)
        base_orders = 250
        
        for current_date in date_range:
            # Seasonal factor (higher demand in winter, lower in summer)
            if seasonal_pattern:
                day_of_year = current_date.timetuple().tm_yday
                seasonal_factor = 1.0 + 0.3 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
            else:
                seasonal_factor = 1.0
            
            # Weekly pattern (higher on weekends, lower mid-week)
            if weekly_pattern:
                day_of_week = current_date.weekday()  # 0 = Monday, 6 = Sunday
                if day_of_week in [5, 6]:  # Saturday, Sunday
                    weekly_factor = 1.4
                elif day_of_week in [0, 4]:  # Monday, Friday
                    weekly_factor = 1.2
                else:  # Tuesday, Wednesday, Thursday
                    weekly_factor = 0.9
            else:
                weekly_factor = 1.0
            
            # Calculate expected orders
            expected_orders = base_orders * seasonal_factor * weekly_factor
            
            # Add random variation (normal distribution with some skew)
            variation = np.random.normal(0, expected_orders * 0.15)
            actual_orders = expected_orders + variation
            
            # Ensure positive integer values
            actual_orders = max(0, int(round(actual_orders)))
            
            delivery_data.append({
                'date': current_date.date(),
                'order_count': actual_orders
            })
        
        df = pd.DataFrame(delivery_data)
        
        # Introduce missing values randomly
        if missing_rate > 0:
            missing_indices = np.random.choice(
                len(df), 
                size=int(len(df) * missing_rate), 
                replace=False
            )
            df.loc[missing_indices, 'order_count'] = np.nan
        
        return df
    
    def generate_edge_case_datasets(self):
        """Generate datasets with various edge cases for testing."""
        
        # 1. Dataset with many missing values
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        rainfall_missing = self.generate_rainfall_data(start_date, end_date, missing_rate=0.3)
        delivery_missing = self.generate_delivery_data(start_date, end_date, missing_rate=0.25)
        
        rainfall_missing.to_csv(self.data_dir / "rainfall_missing_values.csv", index=False)
        delivery_missing.to_csv(self.data_dir / "delivery_missing_values.csv", index=False)
        
        # 2. Dataset with extreme values
        rainfall_extreme = rainfall_missing.copy()
        delivery_extreme = delivery_missing.copy()
        
        # Add some extreme rainfall values
        extreme_indices = np.random.choice(len(rainfall_extreme), size=3, replace=False)
        rainfall_extreme.loc[extreme_indices, 'precipitation_mm'] = [150.5, 200.0, 89.3]
        
        # Add some extreme delivery values
        extreme_indices = np.random.choice(len(delivery_extreme), size=3, replace=False)
        delivery_extreme.loc[extreme_indices, 'order_count'] = [1500, 2000, 50]
        
        rainfall_extreme.to_csv(self.data_dir / "rainfall_extreme_values.csv", index=False)
        delivery_extreme.to_csv(self.data_dir / "delivery_extreme_values.csv", index=False)
        
        # 3. Single data point datasets
        single_rainfall = pd.DataFrame({
            'date': [date(2023, 6, 15)],
            'precipitation_mm': [25.4]
        })
        single_delivery = pd.DataFrame({
            'date': [date(2023, 6, 15)],
            'order_count': [320]
        })
        
        single_rainfall.to_csv(self.data_dir / "rainfall_single_point.csv", index=False)
        single_delivery.to_csv(self.data_dir / "delivery_single_point.csv", index=False)
        
        # 4. Empty datasets (headers only)
        empty_rainfall = pd.DataFrame(columns=['date', 'precipitation_mm'])
        empty_delivery = pd.DataFrame(columns=['date', 'order_count'])
        
        empty_rainfall.to_csv(self.data_dir / "rainfall_empty.csv", index=False)
        empty_delivery.to_csv(self.data_dir / "delivery_empty.csv", index=False)
        
        # 5. Non-overlapping date ranges
        rainfall_early = self.generate_rainfall_data(date(2022, 1, 1), date(2022, 6, 30))
        delivery_late = self.generate_delivery_data(date(2023, 7, 1), date(2023, 12, 31))
        
        rainfall_early.to_csv(self.data_dir / "rainfall_early_period.csv", index=False)
        delivery_late.to_csv(self.data_dir / "delivery_late_period.csv", index=False)
    
    def generate_different_date_formats(self):
        """Generate datasets with different date formats for validation testing."""
        
        # Base data
        start_date = date(2023, 3, 1)
        end_date = date(2023, 3, 15)
        
        rainfall_base = self.generate_rainfall_data(start_date, end_date, missing_rate=0.0)
        delivery_base = self.generate_delivery_data(start_date, end_date, missing_rate=0.0)
        
        # Format 1: MM/DD/YYYY
        rainfall_us = rainfall_base.copy()
        delivery_us = delivery_base.copy()
        rainfall_us['date'] = rainfall_us['date'].apply(lambda x: x.strftime('%m/%d/%Y'))
        delivery_us['date'] = delivery_us['date'].apply(lambda x: x.strftime('%m/%d/%Y'))
        
        rainfall_us.to_csv(self.data_dir / "rainfall_us_format.csv", index=False)
        delivery_us.to_csv(self.data_dir / "delivery_us_format.csv", index=False)
        
        # Format 2: DD/MM/YYYY
        rainfall_eu = rainfall_base.copy()
        delivery_eu = delivery_base.copy()
        rainfall_eu['date'] = rainfall_eu['date'].apply(lambda x: x.strftime('%d/%m/%Y'))
        delivery_eu['date'] = delivery_eu['date'].apply(lambda x: x.strftime('%d/%m/%Y'))
        
        rainfall_eu.to_csv(self.data_dir / "rainfall_eu_format.csv", index=False)
        delivery_eu.to_csv(self.data_dir / "delivery_eu_format.csv", index=False)
        
        # Format 3: YYYY/MM/DD
        rainfall_iso_slash = rainfall_base.copy()
        delivery_iso_slash = delivery_base.copy()
        rainfall_iso_slash['date'] = rainfall_iso_slash['date'].apply(lambda x: x.strftime('%Y/%m/%d'))
        delivery_iso_slash['date'] = delivery_iso_slash['date'].apply(lambda x: x.strftime('%Y/%m/%d'))
        
        rainfall_iso_slash.to_csv(self.data_dir / "rainfall_iso_slash_format.csv", index=False)
        delivery_iso_slash.to_csv(self.data_dir / "delivery_iso_slash_format.csv", index=False)
        
        # Format 4: With timestamps
        rainfall_timestamp = rainfall_base.copy()
        delivery_timestamp = delivery_base.copy()
        rainfall_timestamp['date'] = rainfall_timestamp['date'].apply(
            lambda x: f"{x.strftime('%Y-%m-%d')} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
        )
        delivery_timestamp['date'] = delivery_timestamp['date'].apply(
            lambda x: f"{x.strftime('%Y-%m-%d')} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
        )
        
        rainfall_timestamp.to_csv(self.data_dir / "rainfall_with_timestamps.csv", index=False)
        delivery_timestamp.to_csv(self.data_dir / "delivery_with_timestamps.csv", index=False)
    
    def generate_comprehensive_datasets(self):
        """Generate comprehensive datasets for main testing and demonstration."""
        
        # 1. Full year dataset with realistic patterns
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        
        rainfall_full = self.generate_rainfall_data(start_date, end_date, seasonal_pattern=True)
        delivery_full = self.generate_delivery_data(start_date, end_date, seasonal_pattern=True, weekly_pattern=True)
        
        rainfall_full.to_csv(self.data_dir / "sample_rainfall.csv", index=False)
        delivery_full.to_csv(self.data_dir / "sample_delivery.csv", index=False)
        
        # 2. Six-month dataset for medium-term analysis
        start_date = date(2023, 6, 1)
        end_date = date(2023, 11, 30)
        
        rainfall_medium = self.generate_rainfall_data(start_date, end_date, seasonal_pattern=True)
        delivery_medium = self.generate_delivery_data(start_date, end_date, seasonal_pattern=True, weekly_pattern=True)
        
        rainfall_medium.to_csv(self.data_dir / "rainfall_6months.csv", index=False)
        delivery_medium.to_csv(self.data_dir / "delivery_6months.csv", index=False)
        
        # 3. One-month high-quality dataset
        start_date = date(2023, 9, 1)
        end_date = date(2023, 9, 30)
        
        rainfall_month = self.generate_rainfall_data(start_date, end_date, missing_rate=0.0)
        delivery_month = self.generate_delivery_data(start_date, end_date, missing_rate=0.0)
        
        rainfall_month.to_csv(self.data_dir / "rainfall_september.csv", index=False)
        delivery_month.to_csv(self.data_dir / "delivery_september.csv", index=False)
        
        # 4. Two-week dataset for quick testing
        start_date = date(2023, 10, 1)
        end_date = date(2023, 10, 14)
        
        rainfall_short = self.generate_rainfall_data(start_date, end_date, missing_rate=0.0)
        delivery_short = self.generate_delivery_data(start_date, end_date, missing_rate=0.0)
        
        rainfall_short.to_csv(self.data_dir / "rainfall_2weeks.csv", index=False)
        delivery_short.to_csv(self.data_dir / "delivery_2weeks.csv", index=False)
    
    def generate_all_sample_data(self):
        """Generate all sample datasets for comprehensive testing."""
        print("Generating comprehensive sample datasets...")
        
        print("1. Creating main datasets with realistic patterns...")
        self.generate_comprehensive_datasets()
        
        print("2. Creating edge case datasets...")
        self.generate_edge_case_datasets()
        
        print("3. Creating datasets with different date formats...")
        self.generate_different_date_formats()
        
        print("4. Updating README with dataset descriptions...")
        self.update_readme()
        
        print("Sample data generation complete!")
    
    def update_readme(self):
        """Update the README file with descriptions of all generated datasets."""
        
        readme_content = """# Data Directory

This directory contains CSV files for the rainfall and delivery dashboard.

## Main Sample Datasets

### Full Year Data (Recommended for main testing)
- `sample_rainfall.csv` - Full year of rainfall data (2023) with seasonal patterns
- `sample_delivery.csv` - Full year of delivery data (2023) with seasonal and weekly patterns

### Additional Time Periods
- `rainfall_6months.csv` / `delivery_6months.csv` - 6 months of data (Jun-Nov 2023)
- `rainfall_september.csv` / `delivery_september.csv` - 1 month of high-quality data (Sep 2023)
- `rainfall_2weeks.csv` / `delivery_2weeks.csv` - 2 weeks of data for quick testing (Oct 1-14, 2023)

## Edge Case Testing Datasets

### Missing Values
- `rainfall_missing_values.csv` / `delivery_missing_values.csv` - Datasets with ~25-30% missing values

### Extreme Values
- `rainfall_extreme_values.csv` / `delivery_extreme_values.csv` - Datasets with outliers and extreme values

### Minimal Data
- `rainfall_single_point.csv` / `delivery_single_point.csv` - Single data point for edge case testing
- `rainfall_empty.csv` / `delivery_empty.csv` - Empty datasets (headers only)

### Non-overlapping Periods
- `rainfall_early_period.csv` - Early 2022 data (Jan-Jun)
- `delivery_late_period.csv` - Late 2023 data (Jul-Dec)

## Date Format Validation Datasets

### Different Date Formats (same data, different formats)
- `rainfall_us_format.csv` / `delivery_us_format.csv` - MM/DD/YYYY format
- `rainfall_eu_format.csv` / `delivery_eu_format.csv` - DD/MM/YYYY format  
- `rainfall_iso_slash_format.csv` / `delivery_iso_slash_format.csv` - YYYY/MM/DD format
- `rainfall_with_timestamps.csv` / `delivery_with_timestamps.csv` - YYYY-MM-DD HH:MM:SS format

## Expected File Structure

### Rainfall Files
- **Columns:** date, precipitation_mm
- **Date formats:** Various (YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, etc.)
- **Precipitation:** Non-negative values in millimeters (0.0 to ~200.0)

### Delivery Files  
- **Columns:** date, order_count
- **Date formats:** Various (YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, etc.)
- **Order counts:** Non-negative integers (typically 50 to 2000)

## Usage Notes

1. **Main testing:** Use `sample_rainfall.csv` and `sample_delivery.csv`
2. **Edge case testing:** Use the various edge case datasets to test error handling
3. **Date format testing:** Use the different format datasets to test date parsing
4. **Performance testing:** Use the full year datasets for larger data testing

All datasets are generated with realistic patterns including:
- Seasonal variations in both rainfall and delivery demand
- Weekly patterns in delivery data (higher weekend demand)
- Realistic value distributions and occasional missing data
- Various edge cases for robust testing
"""
        
        with open(self.data_dir / "README.md", 'w') as f:
            f.write(readme_content)


def main():
    """Main function to generate all sample data."""
    generator = SampleDataGenerator(seed=42)  # Fixed seed for reproducible data
    generator.generate_all_sample_data()
    
    # Print summary of generated files
    data_files = list(generator.data_dir.glob("*.csv"))
    print(f"\nGenerated {len(data_files)} CSV files:")
    for file_path in sorted(data_files):
        df = pd.read_csv(file_path)
        print(f"  {file_path.name}: {len(df)} records")


if __name__ == "__main__":
    main()