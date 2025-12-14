"""
Property-based tests for DataLoader functionality

Tests the data loading and processing components to ensure they properly handle
date normalization, dataset joining, and data validation as specified in requirements.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import pandas as pd
import numpy as np
from datetime import date, datetime
import tempfile
import os
from pathlib import Path

from data_loader import DataLoader
from models import RainfallRecord, DeliveryRecord


# Hypothesis strategies for generating test data
valid_dates = st.dates(min_value=date(2020, 1, 1), max_value=date(2025, 12, 31))
valid_precipitation = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
valid_order_counts = st.integers(min_value=0, max_value=100000)

# Date format strategies for testing normalization
# Using unambiguous formats to avoid parsing conflicts
date_formats = [
    '%Y-%m-%d',
    '%Y/%m/%d',
    '%Y-%m-%d %H:%M:%S'
]

date_format_strategy = st.sampled_from(date_formats)


def create_temp_csv(data: list, headers: list) -> str:
    """Helper function to create temporary CSV files for testing"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    
    # Write headers
    temp_file.write(','.join(headers) + '\n')
    
    # Write data rows
    for row in data:
        temp_file.write(','.join(str(item) for item in row) + '\n')
    
    temp_file.close()
    return temp_file.name


class TestDataLoader:
    """Test cases for DataLoader functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
        self.temp_files = []
    
    def teardown_method(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass
    
    def test_load_nonexistent_file_raises_error(self):
        """Loading non-existent files should raise FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            self.loader.load_rainfall_data("nonexistent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            self.loader.load_delivery_data("nonexistent_file.csv")
    
    def test_load_empty_file_raises_error(self):
        """Loading empty CSV files should raise ValueError"""
        empty_file = create_temp_csv([], [])
        self.temp_files.append(empty_file)
        
        with pytest.raises(ValueError):
            self.loader.load_rainfall_data(empty_file)
    
    def test_load_file_missing_columns_raises_error(self):
        """Loading CSV files without required columns should raise ValueError"""
        # Rainfall file missing precipitation column
        rainfall_file = create_temp_csv([['2023-01-01']], ['date'])
        self.temp_files.append(rainfall_file)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.loader.load_rainfall_data(rainfall_file)
        
        # Delivery file missing order_count column  
        delivery_file = create_temp_csv([['2023-01-01']], ['date'])
        self.temp_files.append(delivery_file)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.loader.load_delivery_data(delivery_file)
    
    @given(st.lists(
        st.tuples(valid_dates, valid_precipitation),
        min_size=1,
        max_size=20
    ))
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_load_valid_rainfall_data(self, rainfall_data):
        """Valid rainfall data should load successfully"""
        # Create CSV data
        csv_data = []
        for test_date, precip in rainfall_data:
            csv_data.append([test_date.strftime('%Y-%m-%d'), precip])
        
        temp_file = create_temp_csv(csv_data, ['date', 'precipitation_mm'])
        self.temp_files.append(temp_file)
        
        # Load data
        result_df = self.loader.load_rainfall_data(temp_file)
        
        # Verify results
        assert len(result_df) <= len(rainfall_data)  # May be fewer due to deduplication
        assert list(result_df.columns) == ['date', 'precipitation_mm']
        assert all(isinstance(d, date) for d in result_df['date'])
        assert all(p >= 0 for p in result_df['precipitation_mm'])
    
    @given(st.lists(
        st.tuples(valid_dates, valid_order_counts),
        min_size=1,
        max_size=20
    ))
    def test_load_valid_delivery_data(self, delivery_data):
        """Valid delivery data should load successfully"""
        # Create CSV data
        csv_data = []
        for test_date, orders in delivery_data:
            csv_data.append([test_date.strftime('%Y-%m-%d'), orders])
        
        temp_file = create_temp_csv(csv_data, ['date', 'order_count'])
        self.temp_files.append(temp_file)
        
        # Load data
        result_df = self.loader.load_delivery_data(temp_file)
        
        # Verify results
        assert len(result_df) <= len(delivery_data)  # May be fewer due to deduplication
        assert list(result_df.columns) == ['date', 'order_count']
        assert all(isinstance(d, date) for d in result_df['date'])
        assert all(isinstance(o, (int, np.integer)) for o in result_df['order_count'])
        assert all(o >= 0 for o in result_df['order_count'])


# **Feature: rainfall-delivery-dashboard, Property 7: Date normalization**
# **Validates: Requirements 3.3**
@given(
    st.lists(
        st.tuples(
            valid_dates,
            date_format_strategy,
            valid_precipitation
        ),
        min_size=1,
        max_size=5
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
def test_property_date_normalization(date_format_data):
    """
    Property 7: Date normalization
    For any datasets with different date formats, the system should normalize 
    all dates to a consistent internal format before processing.
    """
    loader = DataLoader()
    temp_files = []
    
    try:
        # Create rainfall data with mixed date formats
        rainfall_csv_data = []
        expected_dates = []
        
        for test_date, date_format, precip in date_format_data:
            formatted_date = test_date.strftime(date_format)
            rainfall_csv_data.append([formatted_date, precip])
            expected_dates.append(test_date)
        
        # Create temporary CSV file
        rainfall_file = create_temp_csv(rainfall_csv_data, ['date', 'precipitation_mm'])
        temp_files.append(rainfall_file)
        
        # Load the data
        result_df = loader.load_rainfall_data(rainfall_file)
        
        # Property: All dates should be normalized to consistent date objects
        assert all(isinstance(d, date) for d in result_df['date'])
        
        # Property: The normalized dates should match the original dates (accounting for deduplication)
        result_dates = set(result_df['date'])
        expected_dates_set = set(expected_dates)
        
        # All result dates should be from the expected set
        assert result_dates.issubset(expected_dates_set)
        
        # Property: Dates should be in chronological order
        date_list = result_df['date'].tolist()
        assert date_list == sorted(date_list)
        
        # Property: No duplicate dates should exist
        assert len(result_df['date']) == len(set(result_df['date']))
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass


class TestDatasetJoining:
    """Test cases for dataset joining functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
    
    def test_join_empty_datasets(self):
        """Joining empty datasets should raise ValueError"""
        empty_df = pd.DataFrame(columns=['date', 'precipitation_mm'])
        empty_df2 = pd.DataFrame(columns=['date', 'order_count'])
        
        with pytest.raises(ValueError, match="Both datasets are empty"):
            self.loader.join_datasets(empty_df, empty_df2)
    
    def test_join_datasets_no_common_dates(self):
        """Joining datasets with no common dates should return empty result"""
        rainfall_df = pd.DataFrame({
            'date': [date(2023, 1, 1), date(2023, 1, 2)],
            'precipitation_mm': [10.0, 15.0]
        })
        
        delivery_df = pd.DataFrame({
            'date': [date(2023, 1, 3), date(2023, 1, 4)],
            'order_count': [100, 150]
        })
        
        result = self.loader.join_datasets(rainfall_df, delivery_df)
        assert result.empty
    
    def test_join_datasets_with_common_dates(self):
        """Joining datasets with common dates should return merged data"""
        rainfall_df = pd.DataFrame({
            'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            'precipitation_mm': [10.0, 15.0, 5.0]
        })
        
        delivery_df = pd.DataFrame({
            'date': [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
            'order_count': [100, 150, 200]
        })
        
        result = self.loader.join_datasets(rainfall_df, delivery_df)
        
        # Should only have dates 2023-01-02 and 2023-01-03 (common dates)
        assert len(result) == 2
        assert set(result['date']) == {date(2023, 1, 2), date(2023, 1, 3)}
        
        # Check data integrity
        jan_2_row = result[result['date'] == date(2023, 1, 2)].iloc[0]
        assert jan_2_row['precipitation_mm'] == 15.0
        assert jan_2_row['order_count'] == 100
        
        jan_3_row = result[result['date'] == date(2023, 1, 3)].iloc[0]
        assert jan_3_row['precipitation_mm'] == 5.0
        assert jan_3_row['order_count'] == 150


# **Feature: rainfall-delivery-dashboard, Property 6: Dataset joining accuracy**
# **Validates: Requirements 3.1, 3.2, 3.4**
@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation),
        min_size=1,
        max_size=20
    ),
    st.lists(
        st.tuples(valid_dates, valid_order_counts),
        min_size=1,
        max_size=20
    )
)
def test_property_dataset_joining_accuracy(rainfall_data, delivery_data):
    """
    Property 6: Dataset joining accuracy
    For any two datasets with overlapping dates, the join operation should preserve 
    data integrity and only include dates present in both datasets.
    """
    loader = DataLoader()
    
    # Create DataFrames from the generated data
    rainfall_df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip}
        for test_date, precip in rainfall_data
    ])
    
    delivery_df = pd.DataFrame([
        {'date': test_date, 'order_count': orders}
        for test_date, orders in delivery_data
    ])
    
    # Remove duplicates (keep first occurrence)
    rainfall_df = rainfall_df.drop_duplicates(subset=['date'], keep='first')
    delivery_df = delivery_df.drop_duplicates(subset=['date'], keep='first')
    
    # Perform the join
    result_df = loader.join_datasets(rainfall_df, delivery_df)
    
    # Property: Only dates present in both datasets should be included
    rainfall_dates = set(rainfall_df['date'])
    delivery_dates = set(delivery_df['date'])
    expected_common_dates = rainfall_dates.intersection(delivery_dates)
    result_dates = set(result_df['date'])
    
    assert result_dates == expected_common_dates
    
    # Property: Data integrity should be preserved
    for _, row in result_df.iterrows():
        test_date = row['date']
        
        # Find corresponding rows in original datasets
        rainfall_row = rainfall_df[rainfall_df['date'] == test_date]
        delivery_row = delivery_df[delivery_df['date'] == test_date]
        
        assert len(rainfall_row) == 1, f"Should have exactly one rainfall record for {test_date}"
        assert len(delivery_row) == 1, f"Should have exactly one delivery record for {test_date}"
        
        # Verify data matches
        assert row['precipitation_mm'] == rainfall_row.iloc[0]['precipitation_mm']
        assert row['order_count'] == delivery_row.iloc[0]['order_count']
    
    # Property: Result should be sorted chronologically
    if len(result_df) > 1:
        dates_list = result_df['date'].tolist()
        assert dates_list == sorted(dates_list)
    
    # Property: All values should be valid
    assert all(result_df['precipitation_mm'] >= 0)
    assert all(result_df['order_count'] >= 0)
    assert all(isinstance(d, date) for d in result_df['date'])
    assert all(isinstance(p, (int, float, np.number)) for p in result_df['precipitation_mm'])
    assert all(isinstance(o, (int, np.integer)) for o in result_df['order_count'])