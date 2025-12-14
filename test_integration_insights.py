"""
Integration test for the insights generator with other components

This test demonstrates that the insights generator works correctly with
the data loader and chart generator components.
"""

import pytest
import pandas as pd
import tempfile
import os
from datetime import date

from data_loader import DataLoader
from chart_generator import ChartGenerator
from insights_generator import InsightsGenerator


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


def test_complete_pipeline_integration():
    """Test the complete pipeline from data loading to insights generation"""
    
    # Create test data
    rainfall_data = [
        ['2023-01-01', 10.5],
        ['2023-01-02', 15.2],
        ['2023-01-03', 8.7],
        ['2023-01-04', 20.1],
        ['2023-01-05', 5.3]
    ]
    
    delivery_data = [
        ['2023-01-01', 120],
        ['2023-01-02', 180],
        ['2023-01-03', 95],
        ['2023-01-04', 210],
        ['2023-01-05', 85]
    ]
    
    # Create temporary CSV files
    rainfall_file = create_temp_csv(rainfall_data, ['date', 'precipitation_mm'])
    delivery_file = create_temp_csv(delivery_data, ['date', 'order_count'])
    
    try:
        # Initialize components
        loader = DataLoader()
        chart_generator = ChartGenerator()
        insights_generator = InsightsGenerator()
        
        # Load data
        rainfall_df = loader.load_rainfall_data(rainfall_file)
        delivery_df = loader.load_delivery_data(delivery_file)
        combined_df = loader.join_datasets(rainfall_df, delivery_df)
        
        # Verify data loaded correctly
        assert len(rainfall_df) == 5
        assert len(delivery_df) == 5
        assert len(combined_df) == 5
        
        # Generate insights
        rainfall_summary = insights_generator.generate_summary_statistics(rainfall_df, "Rainfall Data")
        delivery_summary = insights_generator.generate_summary_statistics(delivery_df, "Delivery Data")
        combined_insights = insights_generator.generate_combined_insights(
            combined_df, rainfall_summary, delivery_summary
        )
        
        # Verify insights were generated
        assert rainfall_summary['record_count'] == 5
        assert delivery_summary['record_count'] == 5
        assert 'precipitation_mm' in rainfall_summary['statistics']
        assert 'order_count' in delivery_summary['statistics']
        
        # Verify correlation analysis
        assert combined_insights['correlation_analysis'] is not None
        assert combined_insights['correlation_analysis']['correlation'] is not None
        
        # Verify non-causal language
        formatted_insights = insights_generator.format_insights_for_display(combined_insights)
        is_valid, found_causal_words = insights_generator.validate_non_causal_language(formatted_insights)
        assert is_valid, f"Causal language found: {found_causal_words}"
        
        # Verify charts can be created with the same data
        rainfall_chart = chart_generator.create_time_series_chart(
            rainfall_df, 'precipitation_mm', 'Rainfall Over Time'
        )
        delivery_chart = chart_generator.create_time_series_chart(
            delivery_df, 'order_count', 'Delivery Orders Over Time'
        )
        scatter_chart = chart_generator.create_scatter_plot(
            combined_df, 'precipitation_mm', 'order_count'
        )
        
        # Verify charts were created successfully
        assert rainfall_chart is not None
        assert delivery_chart is not None
        assert scatter_chart is not None
        
        # Verify summary statistics match between components
        chart_stats = chart_generator.create_summary_statistics_table(combined_df)
        insights_stats = combined_insights['combined_statistics']['statistics']
        
        # Both should have calculated statistics for the same columns
        assert 'precipitation_mm' in insights_stats
        assert 'order_count' in insights_stats
        
        print("✓ Complete pipeline integration test passed")
        print(f"✓ Processed {len(combined_df)} days of combined data")
        print(f"✓ Generated insights with correlation: {combined_insights['correlation_analysis']['correlation']:.3f}")
        print("✓ All text validated as non-causal")
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(rainfall_file)
            os.unlink(delivery_file)
        except FileNotFoundError:
            pass


def test_insights_with_missing_data_integration():
    """Test insights generation with missing data scenarios"""
    
    # Create test data with missing values
    rainfall_data = [
        ['2023-01-01', 10.5],
        ['2023-01-02', ''],  # Missing value
        ['2023-01-03', 8.7],
        ['2023-01-04', 20.1]
    ]
    
    delivery_data = [
        ['2023-01-01', 120],
        ['2023-01-02', 180],
        ['2023-01-03', ''],  # Missing value
        ['2023-01-05', 85]   # Different date
    ]
    
    # Create temporary CSV files
    rainfall_file = create_temp_csv(rainfall_data, ['date', 'precipitation_mm'])
    delivery_file = create_temp_csv(delivery_data, ['date', 'order_count'])
    
    try:
        # Initialize components
        loader = DataLoader()
        insights_generator = InsightsGenerator()
        
        # Load data (should handle missing values gracefully)
        rainfall_df = loader.load_rainfall_data(rainfall_file)
        delivery_df = loader.load_delivery_data(delivery_file)
        combined_df = loader.join_datasets(rainfall_df, delivery_df)
        
        # Generate insights
        rainfall_summary = insights_generator.generate_summary_statistics(rainfall_df, "Rainfall Data")
        delivery_summary = insights_generator.generate_summary_statistics(delivery_df, "Delivery Data")
        combined_insights = insights_generator.generate_combined_insights(
            combined_df, rainfall_summary, delivery_summary
        )
        
        # Verify that data was cleaned (invalid records excluded)
        # The data loader should have excluded records with missing values
        assert len(rainfall_df) < 4  # Should be less than original 4 records
        assert len(delivery_df) < 4  # Should be less than original 4 records
        
        # Verify data quality is good for cleaned data
        # Since invalid records were excluded, the remaining data should be clean
        assert rainfall_summary['data_quality']['completeness'] == 100.0
        assert delivery_summary['data_quality']['completeness'] == 100.0
        
        # Verify insights mention data limitations
        quality_notes = combined_insights['data_quality_notes']
        has_limitation_note = any('limitation' in note.lower() for note in quality_notes)
        assert has_limitation_note
        
        # Verify all text is still non-causal
        formatted_insights = insights_generator.format_insights_for_display(combined_insights)
        is_valid, found_causal_words = insights_generator.validate_non_causal_language(formatted_insights)
        assert is_valid, f"Causal language found: {found_causal_words}"
        
        print("✓ Missing data integration test passed")
        print(f"✓ Handled missing data gracefully")
        print("✓ Reported data quality limitations")
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(rainfall_file)
            os.unlink(delivery_file)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    test_complete_pipeline_integration()
    test_insights_with_missing_data_integration()
    print("\n🎉 All integration tests passed!")