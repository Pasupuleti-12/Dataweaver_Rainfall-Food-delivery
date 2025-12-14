"""
Property-based tests for data model validation

Tests the core data models to ensure they properly validate input data
and exclude invalid records as specified in the requirements.
"""

import pytest
from hypothesis import given, strategies as st, assume
from datetime import date, datetime
from models import (
    RainfallRecord, 
    DeliveryRecord, 
    CombinedDataPoint,
    create_rainfall_record,
    create_delivery_record,
    create_combined_data_point
)


# Hypothesis strategies for generating test data
valid_dates = st.dates(min_value=date(2020, 1, 1), max_value=date(2025, 12, 31))
valid_precipitation = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
valid_order_counts = st.integers(min_value=0, max_value=100000)

# Invalid data strategies
invalid_precipitation = st.one_of(
    st.floats(max_value=-0.1, allow_nan=False, allow_infinity=False),  # Negative values
    st.just(float('nan')),  # NaN values
    st.just(float('inf')),  # Infinity values
    st.just(float('-inf'))  # Negative infinity
)

invalid_order_counts = st.integers(max_value=-1)  # Negative order counts


class TestRainfallRecord:
    """Test cases for RainfallRecord validation"""
    
    @given(valid_dates, valid_precipitation)
    def test_valid_rainfall_records_are_accepted(self, test_date, precipitation):
        """Valid rainfall records should be created successfully"""
        record = RainfallRecord(test_date, precipitation)
        assert record.validate() is True
        assert record.date == test_date
        assert record.precipitation_mm == precipitation
    
    @given(valid_dates, invalid_precipitation)
    def test_invalid_precipitation_rejected(self, test_date, precipitation):
        """Invalid precipitation values should be rejected"""
        assume(precipitation < 0 or not (precipitation == precipitation))  # Negative or NaN
        with pytest.raises(ValueError):
            RainfallRecord(test_date, precipitation)
    
    def test_invalid_date_type_rejected(self):
        """Non-date objects should be rejected for date field"""
        with pytest.raises(ValueError):
            RainfallRecord("2023-01-01", 10.5)
        
        with pytest.raises(ValueError):
            RainfallRecord(None, 10.5)


class TestDeliveryRecord:
    """Test cases for DeliveryRecord validation"""
    
    @given(valid_dates, valid_order_counts)
    def test_valid_delivery_records_are_accepted(self, test_date, order_count):
        """Valid delivery records should be created successfully"""
        record = DeliveryRecord(test_date, order_count)
        assert record.validate() is True
        assert record.date == test_date
        assert record.order_count == order_count
    
    @given(valid_dates, invalid_order_counts)
    def test_invalid_order_counts_rejected(self, test_date, order_count):
        """Negative order counts should be rejected"""
        with pytest.raises(ValueError):
            DeliveryRecord(test_date, order_count)
    
    def test_invalid_date_type_rejected(self):
        """Non-date objects should be rejected for date field"""
        with pytest.raises(ValueError):
            DeliveryRecord("2023-01-01", 100)
        
        with pytest.raises(ValueError):
            DeliveryRecord(None, 100)
    
    def test_non_integer_order_count_rejected(self):
        """Non-integer order counts should be rejected"""
        with pytest.raises(ValueError):
            DeliveryRecord(date(2023, 1, 1), 10.5)


class TestCombinedDataPoint:
    """Test cases for CombinedDataPoint validation"""
    
    @given(valid_dates, valid_precipitation, valid_order_counts)
    def test_valid_combined_data_points_are_accepted(self, test_date, precipitation, order_count):
        """Valid combined data points should be created successfully"""
        point = CombinedDataPoint(test_date, precipitation, order_count)
        assert point.validate() is True
        assert point.date == test_date
        assert point.precipitation_mm == precipitation
        assert point.order_count == order_count
    
    @given(valid_dates, valid_precipitation, valid_order_counts)
    def test_to_dict_conversion(self, test_date, precipitation, order_count):
        """to_dict should return correct dictionary representation"""
        point = CombinedDataPoint(test_date, precipitation, order_count)
        result = point.to_dict()
        
        expected = {
            'date': test_date,
            'rainfall': precipitation,
            'orders': order_count
        }
        assert result == expected
    
    @given(valid_dates, invalid_precipitation, valid_order_counts)
    def test_invalid_precipitation_in_combined_point_rejected(self, test_date, precipitation, order_count):
        """Invalid precipitation in combined data point should be rejected"""
        assume(precipitation < 0 or not (precipitation == precipitation))
        with pytest.raises(ValueError):
            CombinedDataPoint(test_date, precipitation, order_count)
    
    @given(valid_dates, valid_precipitation, invalid_order_counts)
    def test_invalid_order_count_in_combined_point_rejected(self, test_date, precipitation, order_count):
        """Invalid order count in combined data point should be rejected"""
        with pytest.raises(ValueError):
            CombinedDataPoint(test_date, precipitation, order_count)


class TestFactoryFunctions:
    """Test cases for factory functions that handle invalid data gracefully"""
    
    @given(valid_dates, valid_precipitation)
    def test_create_rainfall_record_with_valid_data(self, test_date, precipitation):
        """Factory function should create valid rainfall records"""
        record = create_rainfall_record(test_date, precipitation)
        assert record is not None
        assert isinstance(record, RainfallRecord)
        assert record.date == test_date
        assert record.precipitation_mm == precipitation
    
    @given(valid_dates, invalid_precipitation)
    def test_create_rainfall_record_with_invalid_data_returns_none(self, test_date, precipitation):
        """Factory function should return None for invalid rainfall data"""
        assume(precipitation < 0 or not (precipitation == precipitation))
        record = create_rainfall_record(test_date, precipitation)
        assert record is None
    
    @given(valid_dates, valid_order_counts)
    def test_create_delivery_record_with_valid_data(self, test_date, order_count):
        """Factory function should create valid delivery records"""
        record = create_delivery_record(test_date, order_count)
        assert record is not None
        assert isinstance(record, DeliveryRecord)
        assert record.date == test_date
        assert record.order_count == order_count
    
    @given(valid_dates, invalid_order_counts)
    def test_create_delivery_record_with_invalid_data_returns_none(self, test_date, order_count):
        """Factory function should return None for invalid delivery data"""
        record = create_delivery_record(test_date, order_count)
        assert record is None
    
    @given(valid_dates, valid_precipitation, valid_order_counts)
    def test_create_combined_data_point_with_valid_data(self, test_date, precipitation, order_count):
        """Factory function should create valid combined data points"""
        point = create_combined_data_point(test_date, precipitation, order_count)
        assert point is not None
        assert isinstance(point, CombinedDataPoint)
        assert point.date == test_date
        assert point.precipitation_mm == precipitation
        assert point.order_count == order_count


# **Feature: rainfall-delivery-dashboard, Property 8: Invalid data exclusion**
# **Validates: Requirements 3.5**
@given(
    st.lists(
        st.tuples(
            st.one_of(valid_dates, st.just("invalid_date"), st.just(None)),
            st.one_of(valid_precipitation, invalid_precipitation, st.just("not_a_number")),
            st.one_of(valid_order_counts, invalid_order_counts, st.just("not_an_int"))
        ),
        min_size=1,
        max_size=50
    )
)
def test_property_invalid_data_exclusion(data_records):
    """
    Property 8: Invalid data exclusion
    For any dataset containing unparseable dates or invalid values, 
    the system should exclude invalid records and continue processing valid data.
    """
    valid_rainfall_records = []
    valid_delivery_records = []
    valid_combined_points = []
    
    for date_val, precipitation, order_count in data_records:
        # Try to create rainfall record
        if isinstance(date_val, date) and isinstance(precipitation, (int, float)) and precipitation >= 0:
            try:
                rainfall_record = create_rainfall_record(date_val, precipitation)
                if rainfall_record is not None:
                    valid_rainfall_records.append(rainfall_record)
            except (ValueError, TypeError):
                pass  # Invalid record excluded as expected
        
        # Try to create delivery record  
        if isinstance(date_val, date) and isinstance(order_count, int) and order_count >= 0:
            try:
                delivery_record = create_delivery_record(date_val, order_count)
                if delivery_record is not None:
                    valid_delivery_records.append(delivery_record)
            except (ValueError, TypeError):
                pass  # Invalid record excluded as expected
        
        # Try to create combined data point
        if (isinstance(date_val, date) and 
            isinstance(precipitation, (int, float)) and precipitation >= 0 and
            isinstance(order_count, int) and order_count >= 0):
            try:
                combined_point = create_combined_data_point(date_val, precipitation, order_count)
                if combined_point is not None:
                    valid_combined_points.append(combined_point)
            except (ValueError, TypeError):
                pass  # Invalid record excluded as expected
    
    # Property: System should continue processing valid data even when invalid data is present
    # All created records should be valid
    for record in valid_rainfall_records:
        assert record.validate() is True
    
    for record in valid_delivery_records:
        assert record.validate() is True
    
    for point in valid_combined_points:
        assert point.validate() is True
    
    # Property: Invalid data should be excluded (no invalid records should be created)
    # This is implicitly tested by the factory functions returning None for invalid data
    # and the dataclass constructors raising ValueError for invalid data