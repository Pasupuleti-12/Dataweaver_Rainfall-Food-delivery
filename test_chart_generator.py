"""
Property-based tests for ChartGenerator functionality

Tests the chart generation components to ensure they properly create visualizations
with correct labeling, data integrity, and handle various input scenarios.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import pandas as pd
import numpy as np
from datetime import date, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from chart_generator import ChartGenerator


# Hypothesis strategies for generating test data
valid_dates = st.dates(min_value=date(2020, 1, 1), max_value=date(2025, 12, 31))
valid_precipitation = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
valid_order_counts = st.integers(min_value=0, max_value=100000)


def create_test_dataframe(data_tuples, columns):
    """Helper function to create test DataFrames"""
    return pd.DataFrame(data_tuples, columns=columns)


class TestChartGenerator:
    """Test cases for ChartGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = ChartGenerator()
    
    def test_empty_dataframe_raises_error(self):
        """Empty DataFrames should raise ValueError"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            self.generator.create_time_series_chart(empty_df, 'precipitation_mm', 'Test Chart')
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            self.generator.create_scatter_plot(empty_df, 'precipitation_mm', 'order_count')
    
    def test_missing_columns_raises_error(self):
        """Missing required columns should raise ValueError"""
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.generator.create_time_series_chart(df, 'precipitation_mm', 'Test Chart')
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.generator.create_scatter_plot(df, 'precipitation_mm', 'order_count')
    
    def test_time_series_chart_basic_functionality(self):
        """Basic time series chart creation should work"""
        df = pd.DataFrame({
            'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            'precipitation_mm': [10.5, 15.2, 8.7]
        })
        
        fig = self.generator.create_time_series_chart(df, 'precipitation_mm', 'Rainfall Chart')
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].mode == 'lines+markers'
        assert 'Rainfall Chart' in fig.layout.title.text
    
    def test_scatter_plot_basic_functionality(self):
        """Basic scatter plot creation should work"""
        df = pd.DataFrame({
            'precipitation_mm': [10.5, 15.2, 8.7],
            'order_count': [100, 150, 80]
        })
        
        fig = self.generator.create_scatter_plot(df, 'precipitation_mm', 'order_count')
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # May have trend line too
        assert fig.data[0].mode == 'markers'
    
    def test_correlation_calculation_basic(self):
        """Basic correlation calculation should work"""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]  # Perfect positive correlation
        })
        
        result = self.generator.calculate_correlation(df, 'x', 'y')
        
        assert result['correlation'] is not None
        assert abs(result['correlation'] - 1.0) < 0.001  # Should be close to 1.0
        assert result['sample_size'] == 5
        assert result['error'] is None
    
    def test_summary_statistics_basic(self):
        """Basic summary statistics generation should work"""
        df = pd.DataFrame({
            'precipitation_mm': [10.0, 20.0, 30.0],
            'order_count': [100, 200, 300]
        })
        
        stats_df = self.generator.create_summary_statistics_table(df)
        
        assert not stats_df.empty
        assert 'precipitation_mm' in stats_df.index
        assert 'order_count' in stats_df.index
        assert 'Mean' in stats_df.columns
        assert 'Count' in stats_df.columns


# **Feature: rainfall-delivery-dashboard, Property 1: Time series data visualization**
# **Validates: Requirements 1.1, 2.1**
@given(st.lists(
    st.tuples(valid_dates, valid_precipitation),
    min_size=1,
    max_size=10
))
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10, deadline=None)
def test_property_time_series_data_visualization(rainfall_data):
    """
    Property 1: Time series data visualization
    For any valid dataset (rainfall or delivery), when loaded into the dashboard, 
    the resulting time series visualization should contain all data points from the original dataset.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip}
        for test_date, precip in rainfall_data
    ])
    
    # Remove duplicates (keep first occurrence) to match expected behavior
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Create time series chart
    fig = generator.create_time_series_chart(df, 'precipitation_mm', 'Test Rainfall Chart')
    
    # Property: Chart should be a valid Plotly figure
    assert isinstance(fig, go.Figure)
    
    # Property: Chart should have at least one trace
    assert len(fig.data) >= 1
    
    # Property: The trace should contain all data points from the original dataset
    trace_data = fig.data[0]
    assert len(trace_data.x) == len(df)
    assert len(trace_data.y) == len(df)
    
    # Property: All original dates should be present in the chart
    chart_dates = set(trace_data.x)
    original_dates = set(df['date'])
    assert chart_dates == original_dates
    
    # Property: All original precipitation values should be present in the chart
    chart_precip = set(trace_data.y)
    original_precip = set(df['precipitation_mm'])
    assert chart_precip == original_precip
    
    # Property: Chart should have proper mode for time series (lines+markers)
    assert trace_data.mode == 'lines+markers'
    
    # Property: Chart should have a title
    assert fig.layout.title is not None
    assert fig.layout.title.text is not None
    assert len(fig.layout.title.text) > 0


# Test the same property for delivery data
@given(st.lists(
    st.tuples(valid_dates, valid_order_counts),
    min_size=1,
    max_size=10
))
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10, deadline=None)
def test_property_time_series_data_visualization_delivery(delivery_data):
    """
    Property 1: Time series data visualization (for delivery data)
    For any valid delivery dataset, when loaded into the dashboard, 
    the resulting time series visualization should contain all data points from the original dataset.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'order_count': orders}
        for test_date, orders in delivery_data
    ])
    
    # Remove duplicates (keep first occurrence) to match expected behavior
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Create time series chart
    fig = generator.create_time_series_chart(df, 'order_count', 'Test Delivery Chart')
    
    # Property: Chart should be a valid Plotly figure
    assert isinstance(fig, go.Figure)
    
    # Property: Chart should have at least one trace
    assert len(fig.data) >= 1
    
    # Property: The trace should contain all data points from the original dataset
    trace_data = fig.data[0]
    assert len(trace_data.x) == len(df)
    assert len(trace_data.y) == len(df)
    
    # Property: All original dates should be present in the chart
    chart_dates = set(trace_data.x)
    original_dates = set(df['date'])
    assert chart_dates == original_dates
    
    # Property: All original order counts should be present in the chart
    chart_orders = set(trace_data.y)
    original_orders = set(df['order_count'])
    assert chart_orders == original_orders
    
    # Property: Chart should have proper mode for time series (lines+markers)
    assert trace_data.mode == 'lines+markers'
    
    # Property: Chart should have a title
    assert fig.layout.title is not None
    assert fig.layout.title.text is not None
    assert len(fig.layout.title.text) > 0


# **Feature: rainfall-delivery-dashboard, Property 3: Chart labeling completeness**
# **Validates: Requirements 1.3, 2.3**
@given(st.lists(
    st.tuples(valid_dates, valid_precipitation),
    min_size=1,
    max_size=10
))
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10, deadline=None)
def test_property_chart_labeling_completeness_rainfall(rainfall_data):
    """
    Property 3: Chart labeling completeness
    For any generated chart, the visualization should include proper axis labels, 
    units, and titles appropriate for the data type.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip}
        for test_date, precip in rainfall_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Create time series chart
    chart_title = 'Test Rainfall Chart'
    fig = generator.create_time_series_chart(df, 'precipitation_mm', chart_title)
    
    # Property: Chart should have a proper title
    assert fig.layout.title is not None
    assert fig.layout.title.text is not None
    assert len(fig.layout.title.text.strip()) > 0
    assert chart_title in fig.layout.title.text
    
    # Property: Chart should have proper x-axis label (Date)
    assert fig.layout.xaxis is not None
    assert fig.layout.xaxis.title is not None
    assert fig.layout.xaxis.title.text is not None
    assert 'Date' in fig.layout.xaxis.title.text
    
    # Property: Chart should have proper y-axis label with units for precipitation
    assert fig.layout.yaxis is not None
    assert fig.layout.yaxis.title is not None
    assert fig.layout.yaxis.title.text is not None
    y_axis_title = fig.layout.yaxis.title.text
    # Should contain precipitation-related terms and units
    assert any(term in y_axis_title.lower() for term in ['precipitation', 'rainfall', 'rain'])
    assert 'mm' in y_axis_title.lower()  # Units should be present
    
    # Property: Chart should have a legend or trace name
    assert len(fig.data) > 0
    trace = fig.data[0]
    assert trace.name is not None
    assert len(trace.name.strip()) > 0


@given(st.lists(
    st.tuples(valid_dates, valid_order_counts),
    min_size=1,
    max_size=10
))
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10, deadline=None)
def test_property_chart_labeling_completeness_delivery(delivery_data):
    """
    Property 3: Chart labeling completeness (for delivery data)
    For any generated delivery chart, the visualization should include proper axis labels, 
    units, and titles appropriate for the data type.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'order_count': orders}
        for test_date, orders in delivery_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Create time series chart
    chart_title = 'Test Delivery Chart'
    fig = generator.create_time_series_chart(df, 'order_count', chart_title)
    
    # Property: Chart should have a proper title
    assert fig.layout.title is not None
    assert fig.layout.title.text is not None
    assert len(fig.layout.title.text.strip()) > 0
    assert chart_title in fig.layout.title.text
    
    # Property: Chart should have proper x-axis label (Date)
    assert fig.layout.xaxis is not None
    assert fig.layout.xaxis.title is not None
    assert fig.layout.xaxis.title.text is not None
    assert 'Date' in fig.layout.xaxis.title.text
    
    # Property: Chart should have proper y-axis label for order counts
    assert fig.layout.yaxis is not None
    assert fig.layout.yaxis.title is not None
    assert fig.layout.yaxis.title.text is not None
    y_axis_title = fig.layout.yaxis.title.text
    # Should contain order-related terms
    assert any(term in y_axis_title.lower() for term in ['order', 'delivery', 'count'])
    
    # Property: Chart should have a legend or trace name
    assert len(fig.data) > 0
    trace = fig.data[0]
    assert trace.name is not None
    assert len(trace.name.strip()) > 0


@given(
    st.lists(st.tuples(valid_precipitation, valid_order_counts), min_size=2, max_size=10)
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10, deadline=None)
def test_property_chart_labeling_completeness_scatter(scatter_data):
    """
    Property 3: Chart labeling completeness (for scatter plots)
    For any generated scatter plot, the visualization should include proper axis labels, 
    units, and titles appropriate for the data type.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'precipitation_mm': precip, 'order_count': orders}
        for precip, orders in scatter_data
    ])
    
    # Create scatter plot
    fig = generator.create_scatter_plot(df, 'precipitation_mm', 'order_count')
    
    # Property: Chart should have a proper title
    assert fig.layout.title is not None
    assert fig.layout.title.text is not None
    assert len(fig.layout.title.text.strip()) > 0
    
    # Property: Chart should have proper x-axis label with units
    assert fig.layout.xaxis is not None
    assert fig.layout.xaxis.title is not None
    assert fig.layout.xaxis.title.text is not None
    x_axis_title = fig.layout.xaxis.title.text
    assert any(term in x_axis_title.lower() for term in ['precipitation', 'rainfall'])
    assert 'mm' in x_axis_title.lower()
    
    # Property: Chart should have proper y-axis label
    assert fig.layout.yaxis is not None
    assert fig.layout.yaxis.title is not None
    assert fig.layout.yaxis.title.text is not None
    y_axis_title = fig.layout.yaxis.title.text
    assert any(term in y_axis_title.lower() for term in ['order', 'delivery'])
    
    # Property: Chart should have at least one trace with a name
    assert len(fig.data) > 0
    trace = fig.data[0]
    assert trace.name is not None
    assert len(trace.name.strip()) > 0

# **Feature: rainfall-delivery-dashboard, Property 2: Missing data robustness**
# **Validates: Requirements 1.2, 2.2**
@given(
    st.lists(
        st.tuples(
            valid_dates,
            st.one_of(valid_precipitation, st.none())  # Allow None values for missing data
        ),
        min_size=1,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_missing_data_robustness_rainfall(rainfall_data_with_missing):
    """
    Property 2: Missing data robustness
    For any dataset containing missing values, the dashboard should render 
    visualizations without errors or crashes.
    """
    generator = ChartGenerator()
    
    # Create DataFrame with potential missing values
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip}
        for test_date, precip in rainfall_data_with_missing
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Property: Creating time series chart should not raise an exception
    try:
        fig = generator.create_time_series_chart(df, 'precipitation_mm', 'Test Rainfall Chart')
        chart_created_successfully = True
    except Exception as e:
        # Should not crash - this would violate the robustness property
        chart_created_successfully = False
        pytest.fail(f"Chart creation failed with missing data: {str(e)}")
    
    # Property: Chart should be created successfully
    assert chart_created_successfully
    assert isinstance(fig, go.Figure)
    
    # Property: Chart should have proper structure even with missing data
    assert fig.layout is not None
    assert fig.layout.title is not None
    
    # Property: If there are any valid data points, they should be displayed
    valid_data = df.dropna(subset=['precipitation_mm'])
    if len(valid_data) > 0:
        # Should have at least one trace with data
        assert len(fig.data) >= 1
        trace = fig.data[0]
        
        # All displayed values should be from the valid data
        if hasattr(trace, 'y') and len(trace.y) > 0:
            trace_values = set(trace.y)
            valid_values = set(valid_data['precipitation_mm'])
            assert trace_values.issubset(valid_values)
            
            # Should maintain chronological order
            trace_dates = list(trace.x)
            assert trace_dates == sorted(trace_dates)
    else:
        # If no valid data, chart should still be created (may be empty or have message)
        # This tests graceful handling of completely missing data
        assert isinstance(fig, go.Figure)


@given(
    st.lists(
        st.tuples(
            valid_dates,
            st.one_of(valid_order_counts, st.none())  # Allow None values for missing data
        ),
        min_size=1,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_missing_data_robustness_delivery(delivery_data_with_missing):
    """
    Property 2: Missing data robustness (for delivery data)
    For any delivery dataset containing missing values, the dashboard should render 
    visualizations without errors or crashes.
    """
    generator = ChartGenerator()
    
    # Create DataFrame with potential missing values
    df = pd.DataFrame([
        {'date': test_date, 'order_count': orders}
        for test_date, orders in delivery_data_with_missing
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Property: Creating time series chart should not raise an exception
    try:
        fig = generator.create_time_series_chart(df, 'order_count', 'Test Delivery Chart')
        chart_created_successfully = True
    except Exception as e:
        # Should not crash - this would violate the robustness property
        chart_created_successfully = False
        pytest.fail(f"Chart creation failed with missing data: {str(e)}")
    
    # Property: Chart should be created successfully
    assert chart_created_successfully
    assert isinstance(fig, go.Figure)
    
    # Property: Chart should have proper structure even with missing data
    assert fig.layout is not None
    assert fig.layout.title is not None
    
    # Property: If there are any valid data points, they should be displayed
    valid_data = df.dropna(subset=['order_count'])
    if len(valid_data) > 0:
        # Should have at least one trace with data
        assert len(fig.data) >= 1
        trace = fig.data[0]
        
        # All displayed values should be from the valid data
        if hasattr(trace, 'y') and len(trace.y) > 0:
            trace_values = set(trace.y)
            valid_values = set(valid_data['order_count'])
            assert trace_values.issubset(valid_values)
            
            # Should maintain chronological order
            trace_dates = list(trace.x)
            assert trace_dates == sorted(trace_dates)
    else:
        # If no valid data, chart should still be created (may be empty or have message)
        assert isinstance(fig, go.Figure)


@given(
    st.lists(
        st.tuples(
            st.one_of(valid_precipitation, st.none()),  # Allow None for missing precipitation
            st.one_of(valid_order_counts, st.none())    # Allow None for missing orders
        ),
        min_size=2,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_missing_data_robustness_scatter(scatter_data_with_missing):
    """
    Property 2: Missing data robustness (for scatter plots)
    For any combined dataset containing missing values, the dashboard should render 
    scatter plots without errors or crashes.
    """
    generator = ChartGenerator()
    
    # Create DataFrame with potential missing values
    df = pd.DataFrame([
        {'precipitation_mm': precip, 'order_count': orders}
        for precip, orders in scatter_data_with_missing
    ])
    
    # Property: Creating scatter plot should not raise an exception
    try:
        fig = generator.create_scatter_plot(df, 'precipitation_mm', 'order_count')
        chart_created_successfully = True
    except Exception as e:
        # Should not crash - this would violate the robustness property
        chart_created_successfully = False
        pytest.fail(f"Scatter plot creation failed with missing data: {str(e)}")
    
    # Property: Chart should be created successfully
    assert chart_created_successfully
    assert isinstance(fig, go.Figure)
    
    # Property: Chart should have proper structure even with missing data
    assert fig.layout is not None
    assert fig.layout.title is not None
    
    # Property: Only complete data pairs should be plotted
    valid_data = df.dropna(subset=['precipitation_mm', 'order_count'])
    
    if len(valid_data) > 0:
        # Should have at least one trace with data
        assert len(fig.data) >= 1
        scatter_trace = fig.data[0]
        
        # All plotted points should be from valid (complete) data pairs
        if hasattr(scatter_trace, 'x') and hasattr(scatter_trace, 'y'):
            if len(scatter_trace.x) > 0 and len(scatter_trace.y) > 0:
                plotted_points = set(zip(scatter_trace.x, scatter_trace.y))
                valid_points = set(zip(valid_data['precipitation_mm'], valid_data['order_count']))
                assert plotted_points.issubset(valid_points)
    else:
        # If no complete pairs, chart should still be created (may show "no data" message)
        assert isinstance(fig, go.Figure)


# **Feature: rainfall-delivery-dashboard, Property 4: Chronological data ordering**
# **Validates: Requirements 1.4, 2.4**
@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation),
        min_size=2,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_chronological_data_ordering_rainfall(rainfall_data):
    """
    Property 4: Chronological data ordering
    For any time series dataset, displayed data points should maintain chronological 
    order regardless of input order.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip}
        for test_date, precip in rainfall_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data points
    assume(len(df) >= 2)
    
    # Deliberately shuffle the data to test ordering property
    df_shuffled = df.sample(frac=1.0).reset_index(drop=True)
    
    # Create time series chart with shuffled data
    fig = generator.create_time_series_chart(df_shuffled, 'precipitation_mm', 'Test Rainfall Chart')
    
    # Property: Chart should be created successfully
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    
    # Property: Data points should be in chronological order regardless of input order
    trace = fig.data[0]
    if hasattr(trace, 'x') and len(trace.x) > 0:
        trace_dates = list(trace.x)
        
        # Property: Dates should be in chronological (ascending) order
        sorted_dates = sorted(trace_dates)
        assert trace_dates == sorted_dates, f"Dates not in chronological order: {trace_dates} vs {sorted_dates}"
        
        # Property: All dates should be unique (no duplicates in display)
        assert len(trace_dates) == len(set(trace_dates)), "Duplicate dates found in chart"
        
        # Property: Corresponding y-values should be properly aligned with sorted dates
        if hasattr(trace, 'y') and len(trace.y) == len(trace.x):
            # Create mapping from original data
            original_mapping = dict(zip(df['date'], df['precipitation_mm']))
            
            # Verify each displayed point matches the original data
            for chart_date, chart_value in zip(trace.x, trace.y):
                assert chart_date in original_mapping, f"Chart date {chart_date} not in original data"
                assert original_mapping[chart_date] == chart_value, f"Value mismatch for date {chart_date}"


@given(
    st.lists(
        st.tuples(valid_dates, valid_order_counts),
        min_size=2,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_chronological_data_ordering_delivery(delivery_data):
    """
    Property 4: Chronological data ordering (for delivery data)
    For any delivery time series dataset, displayed data points should maintain 
    chronological order regardless of input order.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'order_count': orders}
        for test_date, orders in delivery_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data points
    assume(len(df) >= 2)
    
    # Deliberately shuffle the data to test ordering property
    df_shuffled = df.sample(frac=1.0).reset_index(drop=True)
    
    # Create time series chart with shuffled data
    fig = generator.create_time_series_chart(df_shuffled, 'order_count', 'Test Delivery Chart')
    
    # Property: Chart should be created successfully
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    
    # Property: Data points should be in chronological order regardless of input order
    trace = fig.data[0]
    if hasattr(trace, 'x') and len(trace.x) > 0:
        trace_dates = list(trace.x)
        
        # Property: Dates should be in chronological (ascending) order
        sorted_dates = sorted(trace_dates)
        assert trace_dates == sorted_dates, f"Dates not in chronological order: {trace_dates} vs {sorted_dates}"
        
        # Property: All dates should be unique (no duplicates in display)
        assert len(trace_dates) == len(set(trace_dates)), "Duplicate dates found in chart"
        
        # Property: Corresponding y-values should be properly aligned with sorted dates
        if hasattr(trace, 'y') and len(trace.y) == len(trace.x):
            # Create mapping from original data
            original_mapping = dict(zip(df['date'], df['order_count']))
            
            # Verify each displayed point matches the original data
            for chart_date, chart_value in zip(trace.x, trace.y):
                assert chart_date in original_mapping, f"Chart date {chart_date} not in original data"
                assert original_mapping[chart_date] == chart_value, f"Value mismatch for date {chart_date}"


# Test chronological ordering in data loading as well
def test_property_chronological_data_ordering_data_loader():
    """
    Property 4: Chronological data ordering (in data loading)
    The DataLoader should ensure chronological ordering is maintained.
    """
    from data_loader import DataLoader
    import tempfile
    import os
    
    loader = DataLoader()
    
    # Create test data in non-chronological order
    test_data = [
        ('2023-01-03', 15.5),
        ('2023-01-01', 10.0),
        ('2023-01-05', 20.2),
        ('2023-01-02', 12.3),
        ('2023-01-04', 18.7)
    ]
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('date,precipitation_mm\n')
        for date_str, precip in test_data:
            f.write(f'{date_str},{precip}\n')
        temp_file = f.name
    
    try:
        # Load the data
        df = loader.load_rainfall_data(temp_file)
        
        # Property: Data should be in chronological order
        if len(df) > 1:
            dates = list(df['date'])
            sorted_dates = sorted(dates)
            assert dates == sorted_dates, f"DataLoader did not maintain chronological order: {dates} vs {sorted_dates}"
        
        # Property: All original data should be present
        assert len(df) == len(test_data)
        
        # Property: Data integrity should be maintained
        for i, (expected_date_str, expected_precip) in enumerate(sorted(test_data, key=lambda x: x[0])):
            expected_date = pd.to_datetime(expected_date_str).date()
            assert df.iloc[i]['date'] == expected_date
            assert df.iloc[i]['precipitation_mm'] == expected_precip
            
    finally:
        # Clean up temporary file
        os.unlink(temp_file)


# **Feature: rainfall-delivery-dashboard, Property 9: Scatter plot generation**
# **Validates: Requirements 4.1**
@given(
    st.lists(st.tuples(valid_precipitation, valid_order_counts), min_size=2, max_size=20)
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10, deadline=None)
def test_property_scatter_plot_generation(scatter_data):
    """
    Property 9: Scatter plot generation
    For any combined dataset, the system should generate scatter plots with correct 
    x-y coordinate mappings between rainfall and delivery variables.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'precipitation_mm': precip, 'order_count': orders}
        for precip, orders in scatter_data
    ])
    
    # Remove any potential duplicates
    df = df.drop_duplicates()
    
    # Skip if we don't have enough unique data points
    assume(len(df) >= 2)
    
    # Create scatter plot
    fig = generator.create_scatter_plot(df, 'precipitation_mm', 'order_count')
    
    # Property: Chart should be a valid Plotly figure
    assert isinstance(fig, go.Figure)
    
    # Property: Chart should have at least one trace (data points, possibly trend line)
    assert len(fig.data) >= 1
    
    # Property: First trace should be scatter plot with markers
    scatter_trace = fig.data[0]
    assert scatter_trace.mode == 'markers'
    
    # Property: Scatter plot should have correct x-y coordinate mappings
    # All x values should correspond to precipitation data
    scatter_x_values = list(scatter_trace.x)
    scatter_y_values = list(scatter_trace.y)
    
    # Should have same number of x and y points
    assert len(scatter_x_values) == len(scatter_y_values)
    
    # All x values should be from the precipitation column
    df_precip_values = set(df['precipitation_mm'])
    scatter_x_set = set(scatter_x_values)
    assert scatter_x_set.issubset(df_precip_values)
    
    # All y values should be from the order count column
    df_order_values = set(df['order_count'])
    scatter_y_set = set(scatter_y_values)
    assert scatter_y_set.issubset(df_order_values)
    
    # Property: Each data point in the scatter plot should correspond to a row in the original data
    scatter_points = set(zip(scatter_x_values, scatter_y_values))
    original_points = set(zip(df['precipitation_mm'], df['order_count']))
    assert scatter_points.issubset(original_points)
    
    # Property: Chart should have proper axis assignments
    # X-axis should be for precipitation (rainfall variable)
    assert fig.layout.xaxis is not None
    assert fig.layout.xaxis.title is not None
    x_title = fig.layout.xaxis.title.text.lower()
    assert any(term in x_title for term in ['precipitation', 'rainfall', 'rain'])
    
    # Y-axis should be for delivery (delivery variable)
    assert fig.layout.yaxis is not None
    assert fig.layout.yaxis.title is not None
    y_title = fig.layout.yaxis.title.text.lower()
    assert any(term in y_title for term in ['order', 'delivery', 'count'])
    
    # Property: All data values should be non-negative (as per our data model constraints)
    assert all(x >= 0 for x in scatter_x_values)
    assert all(y >= 0 for y in scatter_y_values)
    
    # Property: Chart should have a title
    assert fig.layout.title is not None
    assert fig.layout.title.text is not None
    assert len(fig.layout.title.text.strip()) > 0


# **Feature: rainfall-delivery-dashboard, Property 5: Date formatting consistency**
# **Validates: Requirements 1.5, 2.5**
@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation),
        min_size=2,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=15, deadline=None)
def test_property_date_formatting_consistency_rainfall(rainfall_data):
    """
    Property 5: Date formatting consistency
    For any dataset spanning multiple time periods, all date displays should use 
    consistent formatting throughout the interface.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data with varied date ranges
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip}
        for test_date, precip in rainfall_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data points
    assume(len(df) >= 2)
    
    # Ensure we have dates spanning multiple periods (days, months, or years)
    date_range = df['date'].max() - df['date'].min()
    assume(date_range.days >= 1)  # At least span more than one day
    
    # Create time series chart
    fig = generator.create_time_series_chart(df, 'precipitation_mm', 'Test Rainfall Chart')
    
    # Property: Chart should be created successfully
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    
    # Property: All dates in the chart should use consistent formatting
    trace = fig.data[0]
    if hasattr(trace, 'x') and len(trace.x) > 1:
        chart_dates = list(trace.x)
        
        # Property: All dates should be date objects (consistent type)
        assert all(isinstance(d, date) for d in chart_dates), "All chart dates should be date objects"
        
        # Property: Dates should be in consistent chronological order
        sorted_dates = sorted(chart_dates)
        assert chart_dates == sorted_dates, "Dates should be consistently ordered chronologically"
        
        # Property: Date axis formatting should be consistent
        x_axis = fig.layout.xaxis
        assert x_axis is not None
        assert x_axis.title is not None
        assert x_axis.title.text == 'Date', "X-axis should consistently be labeled 'Date'"
        
        # Property: If tick format is specified, it should be consistent
        if hasattr(x_axis, 'tickformat') and x_axis.tickformat:
            # Should use a standard date format
            assert '%' in x_axis.tickformat, "Date tick format should use standard format specifiers"
        
        # Property: Date range should encompass all data points consistently
        if len(chart_dates) > 1:
            chart_min_date = min(chart_dates)
            chart_max_date = max(chart_dates)
            data_min_date = df['date'].min()
            data_max_date = df['date'].max()
            
            assert chart_min_date == data_min_date, "Chart should consistently show minimum date from data"
            assert chart_max_date == data_max_date, "Chart should consistently show maximum date from data"


@given(
    st.lists(
        st.tuples(valid_dates, valid_order_counts),
        min_size=2,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=15, deadline=None)
def test_property_date_formatting_consistency_delivery(delivery_data):
    """
    Property 5: Date formatting consistency (for delivery data)
    For any delivery dataset spanning multiple time periods, all date displays should use 
    consistent formatting throughout the interface.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'order_count': orders}
        for test_date, orders in delivery_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data points
    assume(len(df) >= 2)
    
    # Ensure we have dates spanning multiple periods
    date_range = df['date'].max() - df['date'].min()
    assume(date_range.days >= 1)
    
    # Create time series chart
    fig = generator.create_time_series_chart(df, 'order_count', 'Test Delivery Chart')
    
    # Property: Chart should be created successfully
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    
    # Property: All dates in the chart should use consistent formatting
    trace = fig.data[0]
    if hasattr(trace, 'x') and len(trace.x) > 1:
        chart_dates = list(trace.x)
        
        # Property: All dates should be date objects (consistent type)
        assert all(isinstance(d, date) for d in chart_dates), "All chart dates should be date objects"
        
        # Property: Dates should be in consistent chronological order
        sorted_dates = sorted(chart_dates)
        assert chart_dates == sorted_dates, "Dates should be consistently ordered chronologically"
        
        # Property: Date axis formatting should be consistent with rainfall charts
        x_axis = fig.layout.xaxis
        assert x_axis is not None
        assert x_axis.title is not None
        assert x_axis.title.text == 'Date', "X-axis should consistently be labeled 'Date'"
        
        # Property: Chart layout should use consistent date formatting approach
        # This ensures consistency across different chart types
        assert fig.layout.xaxis.title.text == 'Date', "Date axis should be consistently labeled across chart types"


@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation, valid_order_counts),
        min_size=2,
        max_size=10
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10, deadline=None)
def test_property_date_formatting_consistency_combined(combined_data):
    """
    Property 5: Date formatting consistency (for combined charts)
    For any combined dataset, date formatting should be consistent across all chart types.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in combined_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data points
    assume(len(df) >= 2)
    
    # Create different chart types to test consistency
    rainfall_chart = generator.create_time_series_chart(df, 'precipitation_mm', 'Rainfall Chart')
    delivery_chart = generator.create_time_series_chart(df, 'order_count', 'Delivery Chart')
    combined_chart = generator.create_combined_time_series(df, 'Combined Chart')
    
    charts = [
        ('rainfall', rainfall_chart),
        ('delivery', delivery_chart),
        ('combined', combined_chart)
    ]
    
    # Property: All charts should use consistent date formatting
    x_axis_titles = []
    date_formats = []
    
    for chart_name, chart in charts:
        assert isinstance(chart, go.Figure), f"{chart_name} chart should be a valid Figure"
        
        # Collect x-axis formatting information
        x_axis = chart.layout.xaxis
        assert x_axis is not None, f"{chart_name} chart should have x-axis"
        assert x_axis.title is not None, f"{chart_name} chart should have x-axis title"
        
        x_axis_titles.append(x_axis.title.text)
        
        # Check for consistent tick formatting if specified
        if hasattr(x_axis, 'tickformat') and x_axis.tickformat:
            date_formats.append(x_axis.tickformat)
    
    # Property: All charts should have the same x-axis title
    unique_titles = set(x_axis_titles)
    assert len(unique_titles) == 1, f"All charts should have consistent x-axis titles, found: {unique_titles}"
    assert 'Date' in list(unique_titles)[0], "X-axis title should consistently reference dates"
    
    # Property: If date formats are specified, they should be consistent
    if date_formats:
        unique_formats = set(date_formats)
        assert len(unique_formats) <= 1, f"Date formats should be consistent across charts, found: {unique_formats}"
    
    # Property: All charts should display the same date range for the same data
    for chart_name, chart in charts:
        if len(chart.data) > 0:
            trace = chart.data[0]
            if hasattr(trace, 'x') and len(trace.x) > 0:
                chart_dates = list(trace.x)
                
                # Should include all dates from the original data
                original_dates = set(df['date'])
                chart_dates_set = set(chart_dates)
                
                # All chart dates should be from the original data
                assert chart_dates_set.issubset(original_dates), f"{chart_name} chart dates should be from original data"
                
                # Dates should be in chronological order
                sorted_chart_dates = sorted(chart_dates)
                assert chart_dates == sorted_chart_dates, f"{chart_name} chart dates should be chronologically ordered"


def test_property_date_formatting_consistency_cross_component():
    """
    Property 5: Date formatting consistency (cross-component)
    Date formatting should be consistent between ChartGenerator and DataLoader components.
    """
    from data_loader import DataLoader
    import tempfile
    import os
    
    generator = ChartGenerator()
    loader = DataLoader()
    
    # Create test data with various date formats
    test_data = [
        ('2023-01-01', 10.5),
        ('2023-01-02', 15.2),
        ('2023-01-03', 8.7),
        ('2023-01-04', 12.1),
        ('2023-01-05', 20.3)
    ]
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('date,precipitation_mm\n')
        for date_str, precip in test_data:
            f.write(f'{date_str},{precip}\n')
        temp_file = f.name
    
    try:
        # Load data through DataLoader
        df = loader.load_rainfall_data(temp_file)
        
        # Create chart through ChartGenerator
        fig = generator.create_time_series_chart(df, 'precipitation_mm', 'Test Chart')
        
        # Property: DataLoader and ChartGenerator should use consistent date types
        loader_dates = list(df['date'])
        assert all(isinstance(d, date) for d in loader_dates), "DataLoader should produce date objects"
        
        chart_dates = list(fig.data[0].x)
        assert all(isinstance(d, date) for d in chart_dates), "ChartGenerator should use date objects"
        
        # Property: Date values should be identical between components
        loader_dates_set = set(loader_dates)
        chart_dates_set = set(chart_dates)
        assert loader_dates_set == chart_dates_set, "DataLoader and ChartGenerator should use identical date values"
        
        # Property: Both components should maintain chronological ordering
        assert loader_dates == sorted(loader_dates), "DataLoader should maintain chronological order"
        assert chart_dates == sorted(chart_dates), "ChartGenerator should maintain chronological order"
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)


# **Feature: rainfall-delivery-dashboard, Property 10: Visual differentiation**
# **Validates: Requirements 4.2**
@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation, valid_order_counts),
        min_size=2,
        max_size=10
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=15, deadline=None)
def test_property_visual_differentiation_combined_charts(combined_data):
    """
    Property 10: Visual differentiation
    For any multi-series chart, different data series should have visually distinct 
    colors, markers, or styling.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in combined_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data points
    assume(len(df) >= 2)
    
    # Create combined time series chart (multi-series)
    fig = generator.create_combined_time_series(df, 'Combined Chart Test')
    
    # Property: Chart should be created successfully
    assert isinstance(fig, go.Figure)
    
    # Property: Chart should have multiple traces (series)
    assert len(fig.data) >= 2, "Combined chart should have at least 2 data series"
    
    # Collect visual properties from all traces
    trace_colors = []
    trace_names = []
    trace_markers = []
    trace_line_styles = []
    
    for i, trace in enumerate(fig.data):
        # Property: Each trace should have a distinct name
        assert trace.name is not None, f"Trace {i} should have a name"
        assert len(trace.name.strip()) > 0, f"Trace {i} should have a non-empty name"
        trace_names.append(trace.name)
        
        # Property: Each trace should have color information
        if hasattr(trace, 'line') and trace.line:
            if hasattr(trace.line, 'color') and trace.line.color:
                trace_colors.append(trace.line.color)
        
        if hasattr(trace, 'marker') and trace.marker:
            if hasattr(trace.marker, 'color') and trace.marker.color:
                trace_colors.append(trace.marker.color)
            
            # Collect marker properties for differentiation
            marker_info = {}
            if hasattr(trace.marker, 'size'):
                marker_info['size'] = trace.marker.size
            if hasattr(trace.marker, 'symbol'):
                marker_info['symbol'] = trace.marker.symbol
            trace_markers.append(marker_info)
        
        # Collect line style information
        if hasattr(trace, 'line') and trace.line:
            line_info = {}
            if hasattr(trace.line, 'width'):
                line_info['width'] = trace.line.width
            if hasattr(trace.line, 'dash'):
                line_info['dash'] = trace.line.dash
            trace_line_styles.append(line_info)
    
    # Property: All trace names should be unique (distinct identification)
    unique_names = set(trace_names)
    assert len(unique_names) == len(trace_names), f"All trace names should be unique, found duplicates in: {trace_names}"
    
    # Property: Traces should use different colors for visual differentiation
    if len(trace_colors) >= 2:
        unique_colors = set(trace_colors)
        assert len(unique_colors) >= 2, f"Multiple traces should use different colors, found: {trace_colors}"
    
    # Property: Trace names should be descriptive and different
    expected_terms = ['precipitation', 'rainfall', 'delivery', 'order']
    name_terms_found = []
    
    for name in trace_names:
        name_lower = name.lower()
        for term in expected_terms:
            if term in name_lower:
                name_terms_found.append(term)
                break
    
    # Should have found terms for different data types
    assert len(set(name_terms_found)) >= 2, f"Trace names should reference different data types, found terms: {name_terms_found}"
    
    # Property: Chart should have proper legend to distinguish series
    assert fig.layout.showlegend is True, "Multi-series chart should show legend for differentiation"
    
    # Property: Different y-axes should be used for different data types (if applicable)
    # This is a key visual differentiation for combined charts
    y_axis_titles = []
    
    # Check primary y-axis
    if fig.layout.yaxis and fig.layout.yaxis.title:
        y_axis_titles.append(fig.layout.yaxis.title.text)
    
    # Check secondary y-axis (if exists)
    if hasattr(fig.layout, 'yaxis2') and fig.layout.yaxis2 and fig.layout.yaxis2.title:
        y_axis_titles.append(fig.layout.yaxis2.title.text)
    
    # Property: If multiple y-axes exist, they should have different titles
    if len(y_axis_titles) >= 2:
        unique_y_titles = set(y_axis_titles)
        assert len(unique_y_titles) == len(y_axis_titles), f"Multiple y-axes should have different titles: {y_axis_titles}"
        
        # Should reference different measurement types
        title_terms = []
        for title in y_axis_titles:
            title_lower = title.lower()
            if any(term in title_lower for term in ['precipitation', 'rainfall', 'mm']):
                title_terms.append('precipitation')
            elif any(term in title_lower for term in ['order', 'delivery', 'count']):
                title_terms.append('orders')
        
        assert len(set(title_terms)) >= 2, f"Y-axis titles should reference different measurement types: {y_axis_titles}"


@given(
    st.lists(
        st.tuples(valid_precipitation, valid_order_counts),
        min_size=3,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=15, deadline=None)
def test_property_visual_differentiation_scatter_plot(scatter_data):
    """
    Property 10: Visual differentiation (for scatter plots with trend lines)
    For scatter plots with multiple visual elements, different elements should be 
    visually distinct.
    """
    generator = ChartGenerator()
    
    # Create DataFrame with some correlation to ensure trend line appears
    df = pd.DataFrame([
        {'precipitation_mm': precip, 'order_count': orders}
        for precip, orders in scatter_data
    ])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Skip if we don't have enough data points
    assume(len(df) >= 3)
    
    # Create scatter plot
    fig = generator.create_scatter_plot(df, 'precipitation_mm', 'order_count')
    
    # Property: Chart should be created successfully
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1, "Scatter plot should have at least one trace"
    
    # Property: If multiple traces exist (data points + trend line), they should be visually distinct
    if len(fig.data) >= 2:
        trace_modes = []
        trace_colors = []
        trace_names = []
        
        for trace in fig.data:
            # Collect visual differentiation properties
            if hasattr(trace, 'mode'):
                trace_modes.append(trace.mode)
            
            if hasattr(trace, 'name') and trace.name:
                trace_names.append(trace.name)
            
            # Collect color information
            if hasattr(trace, 'marker') and trace.marker and hasattr(trace.marker, 'color'):
                trace_colors.append(trace.marker.color)
            elif hasattr(trace, 'line') and trace.line and hasattr(trace.line, 'color'):
                trace_colors.append(trace.line.color)
        
        # Property: Different traces should use different modes for visual distinction
        if len(trace_modes) >= 2:
            # Should have different modes (e.g., 'markers' vs 'lines')
            unique_modes = set(trace_modes)
            assert len(unique_modes) >= 2, f"Multiple traces should use different modes for distinction: {trace_modes}"
        
        # Property: Different traces should have different names
        if len(trace_names) >= 2:
            unique_names = set(trace_names)
            assert len(unique_names) == len(trace_names), f"All trace names should be unique: {trace_names}"
        
        # Property: Different traces should use different colors
        if len(trace_colors) >= 2:
            unique_colors = set(trace_colors)
            assert len(unique_colors) >= 2, f"Multiple traces should use different colors: {trace_colors}"
    
    # Property: Main scatter trace should use markers mode
    main_trace = fig.data[0]
    assert main_trace.mode == 'markers', "Main scatter trace should use markers mode"
    
    # Property: Scatter points should have visible styling
    assert hasattr(main_trace, 'marker'), "Scatter trace should have marker styling"
    assert main_trace.marker is not None, "Scatter trace marker should be defined"
    
    if hasattr(main_trace.marker, 'size'):
        assert main_trace.marker.size > 0, "Scatter markers should have positive size"
    
    # Property: Chart should have distinct axis labels for visual clarity
    assert fig.layout.xaxis is not None, "Chart should have x-axis"
    assert fig.layout.yaxis is not None, "Chart should have y-axis"
    
    x_title = fig.layout.xaxis.title.text if fig.layout.xaxis.title else ""
    y_title = fig.layout.yaxis.title.text if fig.layout.yaxis.title else ""
    
    assert x_title != y_title, f"X and Y axis titles should be different for visual distinction: '{x_title}' vs '{y_title}'"


def test_property_visual_differentiation_individual_charts():
    """
    Property 10: Visual differentiation (individual chart consistency)
    Individual charts should use consistent but distinct visual styling.
    """
    generator = ChartGenerator()
    
    # Create test data
    test_data = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        'precipitation_mm': [10.5, 15.2, 8.7],
        'order_count': [100, 150, 80]
    })
    
    # Create individual charts
    rainfall_chart = generator.create_time_series_chart(test_data, 'precipitation_mm', 'Rainfall Chart')
    delivery_chart = generator.create_time_series_chart(test_data, 'order_count', 'Delivery Chart')
    
    charts = [
        ('rainfall', rainfall_chart),
        ('delivery', delivery_chart)
    ]
    
    # Property: Each chart should have distinct visual identity
    chart_colors = []
    chart_titles = []
    y_axis_titles = []
    
    for chart_name, chart in charts:
        assert isinstance(chart, go.Figure), f"{chart_name} should be a valid chart"
        assert len(chart.data) >= 1, f"{chart_name} should have data traces"
        
        # Collect visual properties
        trace = chart.data[0]
        
        # Chart titles should be different
        if chart.layout.title and chart.layout.title.text:
            chart_titles.append(chart.layout.title.text)
        
        # Y-axis titles should be different
        if chart.layout.yaxis and chart.layout.yaxis.title and chart.layout.yaxis.title.text:
            y_axis_titles.append(chart.layout.yaxis.title.text)
        
        # Colors should be different
        if hasattr(trace, 'line') and trace.line and hasattr(trace.line, 'color'):
            chart_colors.append(trace.line.color)
        elif hasattr(trace, 'marker') and trace.marker and hasattr(trace.marker, 'color'):
            chart_colors.append(trace.marker.color)
        
        # Property: Each chart should have proper visual styling
        assert trace.mode == 'lines+markers', f"{chart_name} should use lines+markers mode"
        
        if hasattr(trace, 'name') and trace.name:
            assert len(trace.name.strip()) > 0, f"{chart_name} trace should have a meaningful name"
    
    # Property: Charts should have different titles
    if len(chart_titles) >= 2:
        unique_titles = set(chart_titles)
        assert len(unique_titles) == len(chart_titles), f"Chart titles should be unique: {chart_titles}"
    
    # Property: Charts should have different y-axis labels
    if len(y_axis_titles) >= 2:
        unique_y_titles = set(y_axis_titles)
        assert len(unique_y_titles) == len(y_axis_titles), f"Y-axis titles should be unique: {y_axis_titles}"
    
    # Property: Charts should use different colors for visual distinction
    if len(chart_colors) >= 2:
        unique_colors = set(chart_colors)
        assert len(unique_colors) >= 2, f"Charts should use different colors: {chart_colors}"


def test_property_visual_differentiation_color_scheme():
    """
    Property 10: Visual differentiation (color scheme consistency)
    The color scheme should provide sufficient visual differentiation while maintaining consistency.
    """
    generator = ChartGenerator()
    
    # Property: ChartGenerator should have a defined color scheme
    assert hasattr(generator, 'colors'), "ChartGenerator should have a colors attribute"
    assert isinstance(generator.colors, dict), "Colors should be a dictionary"
    
    # Property: Color scheme should include different colors for different data types
    expected_color_keys = ['rainfall', 'delivery', 'scatter', 'correlation']
    for key in expected_color_keys:
        assert key in generator.colors, f"Color scheme should include '{key}' color"
        assert isinstance(generator.colors[key], str), f"Color for '{key}' should be a string"
        assert len(generator.colors[key]) > 0, f"Color for '{key}' should not be empty"
    
    # Property: All colors should be different for visual differentiation
    color_values = list(generator.colors.values())
    unique_colors = set(color_values)
    assert len(unique_colors) == len(color_values), f"All colors should be unique: {color_values}"
    
    # Property: Colors should be valid color specifications
    for color_name, color_value in generator.colors.items():
        # Should be hex color or named color
        if color_value.startswith('#'):
            assert len(color_value) == 7, f"Hex color for '{color_name}' should be 7 characters: {color_value}"
            # Check if it's valid hex
            try:
                int(color_value[1:], 16)
            except ValueError:
                pytest.fail(f"Invalid hex color for '{color_name}': {color_value}")
        else:
            # Should be a reasonable color name/specification
            assert len(color_value) >= 3, f"Color specification for '{color_name}' too short: {color_value}"


# **Feature: rainfall-delivery-dashboard, Property 11: Correlation calculation accuracy**
# **Validates: Requirements 4.3**
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
        ),
        min_size=3,
        max_size=20
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=25, deadline=None)
def test_property_correlation_calculation_accuracy(data_pairs):
    """
    Property 11: Correlation calculation accuracy
    For any dataset with sufficient data points, calculated correlation coefficients 
    should match standard statistical formulas within acceptable precision.
    """
    generator = ChartGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'precipitation_mm': x, 'order_count': y}
        for x, y in data_pairs
    ])
    
    # Remove duplicates and ensure we have enough data
    df = df.drop_duplicates()
    assume(len(df) >= 3)
    
    # Calculate correlation using ChartGenerator
    correlation_result = generator.calculate_correlation(df, 'precipitation_mm', 'order_count')
    
    # Property: Correlation calculation should succeed with sufficient data
    assert correlation_result is not None, "Correlation calculation should return a result"
    assert isinstance(correlation_result, dict), "Correlation result should be a dictionary"
    assert 'correlation' in correlation_result, "Result should contain correlation value"
    assert 'sample_size' in correlation_result, "Result should contain sample size"
    assert 'error' in correlation_result, "Result should contain error field"
    
    # Property: Sample size should match input data
    assert correlation_result['sample_size'] == len(df), f"Sample size should match data length: {correlation_result['sample_size']} vs {len(df)}"
    
    # Property: If no error occurred, correlation should be calculated
    if correlation_result['error'] is None:
        calculated_corr = correlation_result['correlation']
        assert calculated_corr is not None, "Correlation should not be None when no error occurred"
        
        # Property: Correlation should be within valid range [-1, 1]
        assert -1.0 <= calculated_corr <= 1.0, f"Correlation should be between -1 and 1: {calculated_corr}"
        
        # Property: Correlation should match pandas calculation (ground truth)
        expected_corr = df['precipitation_mm'].corr(df['order_count'])
        
        if not pd.isna(expected_corr):
            # Should match within floating point precision
            assert abs(calculated_corr - expected_corr) < 1e-10, \
                f"Calculated correlation {calculated_corr} should match pandas correlation {expected_corr}"
        else:
            # If pandas returns NaN, our calculation should handle it appropriately
            assert correlation_result['error'] is not None or calculated_corr is None, \
                "Should handle NaN correlation appropriately"
    
    # Property: Error handling should be appropriate
    if correlation_result['error'] is not None:
        # Should have a meaningful error message
        assert isinstance(correlation_result['error'], str), "Error should be a string"
        assert len(correlation_result['error']) > 0, "Error message should not be empty"
        
        # Correlation should be None when there's an error
        assert correlation_result['correlation'] is None, "Correlation should be None when error occurred"


def test_property_correlation_calculation_accuracy_known_values():
    """
    Property 11: Correlation calculation accuracy (with known correlation values)
    Test correlation calculation with datasets that have known correlation values.
    """
    generator = ChartGenerator()
    
    # Test case 1: Perfect positive correlation
    perfect_positive_df = pd.DataFrame({
        'precipitation_mm': [1.0, 2.0, 3.0, 4.0, 5.0],
        'order_count': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    result = generator.calculate_correlation(perfect_positive_df, 'precipitation_mm', 'order_count')
    
    # Property: Perfect positive correlation should be 1.0
    assert result['error'] is None, "Perfect correlation calculation should not error"
    assert result['correlation'] is not None, "Perfect correlation should be calculated"
    assert abs(result['correlation'] - 1.0) < 1e-10, f"Perfect positive correlation should be 1.0, got {result['correlation']}"
    
    # Test case 2: Perfect negative correlation
    perfect_negative_df = pd.DataFrame({
        'precipitation_mm': [1.0, 2.0, 3.0, 4.0, 5.0],
        'order_count': [50.0, 40.0, 30.0, 20.0, 10.0]
    })
    
    result = generator.calculate_correlation(perfect_negative_df, 'precipitation_mm', 'order_count')
    
    # Property: Perfect negative correlation should be -1.0
    assert result['error'] is None, "Perfect negative correlation calculation should not error"
    assert result['correlation'] is not None, "Perfect negative correlation should be calculated"
    assert abs(result['correlation'] - (-1.0)) < 1e-10, f"Perfect negative correlation should be -1.0, got {result['correlation']}"
    
    # Test case 3: No correlation (random data)
    no_correlation_df = pd.DataFrame({
        'precipitation_mm': [1.0, 2.0, 3.0, 4.0, 5.0],
        'order_count': [25.0, 10.0, 40.0, 15.0, 30.0]
    })
    
    result = generator.calculate_correlation(no_correlation_df, 'precipitation_mm', 'order_count')
    
    # Property: Should calculate some correlation value (not necessarily 0)
    assert result['error'] is None, "No correlation calculation should not error"
    assert result['correlation'] is not None, "Should calculate correlation even for uncorrelated data"
    assert -1.0 <= result['correlation'] <= 1.0, "Correlation should be in valid range"
    
    # Verify against pandas calculation
    expected = no_correlation_df['precipitation_mm'].corr(no_correlation_df['order_count'])
    assert abs(result['correlation'] - expected) < 1e-10, "Should match pandas calculation"
    
    # Test case 4: Constant values (should handle gracefully)
    constant_df = pd.DataFrame({
        'precipitation_mm': [5.0, 5.0, 5.0, 5.0, 5.0],
        'order_count': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    result = generator.calculate_correlation(constant_df, 'precipitation_mm', 'order_count')
    
    # Property: Should handle constant values appropriately
    # (correlation is undefined when one variable is constant)
    if result['error'] is not None:
        assert 'constant' in result['error'].lower() or 'cannot' in result['error'].lower(), \
            "Should provide appropriate error message for constant values"
    else:
        # If it doesn't error, correlation should be NaN or None
        assert result['correlation'] is None or pd.isna(result['correlation']), \
            "Correlation with constant values should be None or NaN"


@given(
    st.integers(min_value=2, max_value=100)
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_correlation_calculation_accuracy_sample_sizes(sample_size):
    """
    Property 11: Correlation calculation accuracy (various sample sizes)
    Correlation calculation should work correctly for different sample sizes.
    """
    generator = ChartGenerator()
    
    # Generate data with known correlation structure
    np.random.seed(42)  # For reproducible results
    x = np.random.normal(0, 1, sample_size)
    # Create y with some correlation to x
    correlation_strength = 0.7
    y = correlation_strength * x + np.sqrt(1 - correlation_strength**2) * np.random.normal(0, 1, sample_size)
    
    # Ensure positive values as per our data model
    x = np.abs(x) * 10  # Scale to reasonable precipitation values
    y = np.abs(y) * 100  # Scale to reasonable order count values
    
    df = pd.DataFrame({
        'precipitation_mm': x,
        'order_count': y
    })
    
    # Calculate correlation
    result = generator.calculate_correlation(df, 'precipitation_mm', 'order_count')
    
    # Property: Should handle various sample sizes appropriately
    assert result is not None, f"Should handle sample size {sample_size}"
    assert result['sample_size'] == sample_size, f"Sample size should be recorded correctly"
    
    if sample_size >= 2:
        # Property: Should be able to calculate correlation with sufficient data
        if result['error'] is None:
            assert result['correlation'] is not None, f"Should calculate correlation for sample size {sample_size}"
            assert -1.0 <= result['correlation'] <= 1.0, "Correlation should be in valid range"
            
            # Property: Should match pandas calculation
            expected_corr = df['precipitation_mm'].corr(df['order_count'])
            if not pd.isna(expected_corr):
                assert abs(result['correlation'] - expected_corr) < 1e-10, \
                    f"Should match pandas for sample size {sample_size}"
    
    # Property: Interpretation should be provided
    assert 'interpretation' in result, "Should provide interpretation"
    assert isinstance(result['interpretation'], str), "Interpretation should be a string"
    assert len(result['interpretation']) > 0, "Interpretation should not be empty"


def test_property_correlation_calculation_accuracy_edge_cases():
    """
    Property 11: Correlation calculation accuracy (edge cases)
    Test correlation calculation with edge cases and boundary conditions.
    """
    generator = ChartGenerator()
    
    # Test case 1: Empty DataFrame
    empty_df = pd.DataFrame({'precipitation_mm': [], 'order_count': []})
    result = generator.calculate_correlation(empty_df, 'precipitation_mm', 'order_count')
    
    # Property: Should handle empty data gracefully
    assert result is not None, "Should return result for empty data"
    assert result['sample_size'] == 0, "Sample size should be 0 for empty data"
    assert result['correlation'] is None, "Correlation should be None for empty data"
    assert result['error'] is not None, "Should have error message for empty data"
    
    # Test case 2: Single data point
    single_df = pd.DataFrame({
        'precipitation_mm': [10.0],
        'order_count': [100.0]
    })
    result = generator.calculate_correlation(single_df, 'precipitation_mm', 'order_count')
    
    # Property: Should handle single data point appropriately
    assert result['sample_size'] == 1, "Sample size should be 1"
    assert result['correlation'] is None, "Correlation should be None for single point"
    assert result['error'] is not None, "Should have error for insufficient data"
    assert 'insufficient' in result['error'].lower(), "Error should mention insufficient data"
    
    # Test case 3: Missing columns
    missing_col_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    result = generator.calculate_correlation(missing_col_df, 'precipitation_mm', 'order_count')
    
    # Property: Should handle missing columns gracefully
    assert result['correlation'] is None, "Correlation should be None for missing columns"
    assert result['error'] is not None, "Should have error for missing columns"
    assert 'missing' in result['error'].lower(), "Error should mention missing columns"
    
    # Test case 4: Data with NaN values
    nan_df = pd.DataFrame({
        'precipitation_mm': [1.0, 2.0, np.nan, 4.0, 5.0],
        'order_count': [10.0, np.nan, 30.0, 40.0, 50.0]
    })
    result = generator.calculate_correlation(nan_df, 'precipitation_mm', 'order_count')
    
    # Property: Should handle NaN values by excluding them
    if result['error'] is None:
        # Should only count complete pairs
        complete_pairs = nan_df.dropna(subset=['precipitation_mm', 'order_count'])
        expected_sample_size = len(complete_pairs)
        assert result['sample_size'] == expected_sample_size, \
            f"Should count only complete pairs: expected {expected_sample_size}, got {result['sample_size']}"
        
        if expected_sample_size >= 2:
            # Should calculate correlation on complete pairs only
            expected_corr = complete_pairs['precipitation_mm'].corr(complete_pairs['order_count'])
            if not pd.isna(expected_corr):
                assert abs(result['correlation'] - expected_corr) < 1e-10, \
                    "Should calculate correlation on complete pairs only"
    
    # Test case 5: Very small values (numerical precision test)
    small_values_df = pd.DataFrame({
        'precipitation_mm': [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
        'order_count': [1e-8, 2e-8, 3e-8, 4e-8, 5e-8]
    })
    result = generator.calculate_correlation(small_values_df, 'precipitation_mm', 'order_count')
    
    # Property: Should handle very small values correctly
    if result['error'] is None:
        assert result['correlation'] is not None, "Should handle small values"
        # Should be perfect correlation (1.0) since it's a linear relationship
        assert abs(result['correlation'] - 1.0) < 1e-6, "Should maintain precision with small values"


def test_property_correlation_calculation_accuracy_statistical_properties():
    """
    Property 11: Correlation calculation accuracy (statistical properties)
    Test that correlation calculations maintain proper statistical properties.
    """
    generator = ChartGenerator()
    
    # Property: Correlation should be symmetric (corr(X,Y) = corr(Y,X))
    test_df = pd.DataFrame({
        'precipitation_mm': [1.0, 2.0, 3.0, 4.0, 5.0],
        'order_count': [2.0, 4.0, 1.0, 5.0, 3.0]
    })
    
    result_xy = generator.calculate_correlation(test_df, 'precipitation_mm', 'order_count')
    result_yx = generator.calculate_correlation(test_df, 'order_count', 'precipitation_mm')
    
    # Property: Correlation should be symmetric
    if result_xy['error'] is None and result_yx['error'] is None:
        assert abs(result_xy['correlation'] - result_yx['correlation']) < 1e-10, \
            f"Correlation should be symmetric: {result_xy['correlation']} vs {result_yx['correlation']}"
    
    # Property: Correlation should be scale-invariant
    scaled_df = test_df.copy()
    scaled_df['precipitation_mm'] = scaled_df['precipitation_mm'] * 100
    scaled_df['order_count'] = scaled_df['order_count'] * 0.01
    
    result_original = generator.calculate_correlation(test_df, 'precipitation_mm', 'order_count')
    result_scaled = generator.calculate_correlation(scaled_df, 'precipitation_mm', 'order_count')
    
    # Property: Scaling should not affect correlation
    if result_original['error'] is None and result_scaled['error'] is None:
        assert abs(result_original['correlation'] - result_scaled['correlation']) < 1e-10, \
            f"Correlation should be scale-invariant: {result_original['correlation']} vs {result_scaled['correlation']}"
    
    # Property: Adding constants should not affect correlation
    shifted_df = test_df.copy()
    shifted_df['precipitation_mm'] = shifted_df['precipitation_mm'] + 1000
    shifted_df['order_count'] = shifted_df['order_count'] + 5000
    
    result_shifted = generator.calculate_correlation(shifted_df, 'precipitation_mm', 'order_count')
    
    # Property: Translation should not affect correlation
    if result_original['error'] is None and result_shifted['error'] is None:
        assert abs(result_original['correlation'] - result_shifted['correlation']) < 1e-10, \
            f"Correlation should be translation-invariant: {result_original['correlation']} vs {result_shifted['correlation']}"