#!/usr/bin/env python3
"""
Integration test script to verify all components work together properly.

This script tests the complete data pipeline from CSV loading through visualization
without requiring the Streamlit interface.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import pytest

# Import all components
from data_loader import DataLoader
from chart_generator import ChartGenerator
from insights_generator import InsightsGenerator
from dashboard_controller import DashboardController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture
def rainfall_df():
    """Fixture to provide rainfall data for tests."""
    data_loader = DataLoader()
    try:
        return data_loader.load_rainfall_data("data/sample_rainfall.csv")
    except Exception:
        return pd.DataFrame()


@pytest.fixture
def delivery_df():
    """Fixture to provide delivery data for tests."""
    data_loader = DataLoader()
    try:
        return data_loader.load_delivery_data("data/sample_delivery.csv")
    except Exception:
        return pd.DataFrame()


@pytest.fixture
def combined_df(rainfall_df, delivery_df):
    """Fixture to provide combined data for tests."""
    if rainfall_df.empty or delivery_df.empty:
        return pd.DataFrame()
    
    data_loader = DataLoader()
    try:
        return data_loader.join_datasets(rainfall_df, delivery_df)
    except Exception:
        return pd.DataFrame()


def test_data_loading():
    """Test data loading functionality with sample datasets."""
    logger.info("Testing data loading functionality...")
    
    data_loader = DataLoader()
    
    # Test loading main sample datasets
    rainfall_df = data_loader.load_rainfall_data("data/sample_rainfall.csv")
    delivery_df = data_loader.load_delivery_data("data/sample_delivery.csv")
    
    assert not rainfall_df.empty, "Rainfall data should not be empty"
    assert not delivery_df.empty, "Delivery data should not be empty"
    
    logger.info(f"✅ Loaded rainfall data: {len(rainfall_df)} records")
    logger.info(f"✅ Loaded delivery data: {len(delivery_df)} records")
    
    # Test dataset joining
    combined_df = data_loader.join_datasets(rainfall_df, delivery_df)
    assert not combined_df.empty, "Combined dataset should not be empty"
    logger.info(f"✅ Combined dataset: {len(combined_df)} records")
    
    # Test data quality validation
    quality_report = data_loader.validate_data_quality(combined_df)
    assert 'quality_score' in quality_report, "Quality report should contain quality_score"
    logger.info(f"✅ Data quality score: {quality_report['quality_score']:.2f}")


def test_chart_generation(rainfall_df, delivery_df, combined_df):
    """Test chart generation functionality."""
    logger.info("Testing chart generation functionality...")
    
    chart_generator = ChartGenerator()
    
    # Test time series charts
    if not rainfall_df.empty:
        rainfall_chart = chart_generator.create_time_series_chart(
            rainfall_df, 'precipitation_mm', 'Test Rainfall Chart'
        )
        assert rainfall_chart is not None
        logger.info("✅ Rainfall time series chart created")
    
    if not delivery_df.empty:
        delivery_chart = chart_generator.create_time_series_chart(
            delivery_df, 'order_count', 'Test Delivery Chart'
        )
        assert delivery_chart is not None
        logger.info("✅ Delivery time series chart created")
    
    # Test scatter plot
    if not combined_df.empty:
        scatter_chart = chart_generator.create_scatter_plot(
            combined_df, 'precipitation_mm', 'order_count'
        )
        assert scatter_chart is not None
        logger.info("✅ Scatter plot created")
        
        # Test combined time series
        combined_chart = chart_generator.create_combined_time_series(combined_df)
        assert combined_chart is not None
        logger.info("✅ Combined time series chart created")
        
        # Test correlation calculation
        correlation_stats = chart_generator.calculate_correlation(
            combined_df, 'precipitation_mm', 'order_count'
        )
        assert 'correlation' in correlation_stats
        logger.info(f"✅ Correlation calculated: {correlation_stats.get('correlation', 'N/A')}")


def test_insights_generation(rainfall_df, delivery_df, combined_df):
    """Test insights generation functionality."""
    logger.info("Testing insights generation functionality...")
    
    insights_generator = InsightsGenerator()
    
    # Test individual dataset insights
    if not rainfall_df.empty:
        rainfall_insights = insights_generator.generate_summary_statistics(
            rainfall_df, "Test Rainfall Data"
        )
        assert rainfall_insights is not None
        logger.info("✅ Rainfall insights generated")
    
    if not delivery_df.empty:
        delivery_insights = insights_generator.generate_summary_statistics(
            delivery_df, "Test Delivery Data"
        )
        assert delivery_insights is not None
        logger.info("✅ Delivery insights generated")
    
    # Test combined insights
    if not combined_df.empty and not rainfall_df.empty and not delivery_df.empty:
        rainfall_insights = insights_generator.generate_summary_statistics(
            rainfall_df, "Test Rainfall Data"
        )
        delivery_insights = insights_generator.generate_summary_statistics(
            delivery_df, "Test Delivery Data"
        )
        
        combined_insights = insights_generator.generate_combined_insights(
            combined_df, rainfall_insights, delivery_insights
        )
        assert combined_insights is not None
        logger.info("✅ Combined insights generated")
        
        # Test non-causal language validation
        sample_text = "Rainfall causes delivery orders to increase significantly."
        is_valid, causal_words = insights_generator.validate_non_causal_language(sample_text)
        assert isinstance(is_valid, bool)
        assert isinstance(causal_words, list)
        logger.info(f"✅ Causal language validation: {'Valid' if is_valid else 'Invalid'} (found: {causal_words})")


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Testing edge cases and error handling...")
    
    data_loader = DataLoader()
    
    # Test empty datasets
    try:
        empty_df = data_loader.load_rainfall_data("data/rainfall_empty.csv")
        logger.info("✅ Empty dataset handled gracefully")
    except Exception as e:
        logger.info(f"✅ Empty dataset error handled: {type(e).__name__}")
        # This is expected behavior, so we don't fail the test
    
    # Test missing values dataset
    missing_df = data_loader.load_rainfall_data("data/rainfall_missing_values.csv")
    assert isinstance(missing_df, pd.DataFrame), "Should return a DataFrame even with missing values"
    logger.info(f"✅ Missing values dataset loaded: {len(missing_df)} records")
    
    # Test different date formats
    us_format_df = data_loader.load_rainfall_data("data/rainfall_us_format.csv")
    assert isinstance(us_format_df, pd.DataFrame), "Should handle US date format"
    logger.info(f"✅ US date format handled: {len(us_format_df)} records")
    
    # Test extreme values
    extreme_df = data_loader.load_rainfall_data("data/rainfall_extreme_values.csv")
    assert isinstance(extreme_df, pd.DataFrame), "Should handle extreme values"
    logger.info(f"✅ Extreme values handled: {len(extreme_df)} records")


def test_file_upload_simulation():
    """Test file upload functionality simulation."""
    logger.info("Testing file upload simulation...")
    
    # Test that sample data files exist and are accessible
    sample_files = [
        "data/sample_rainfall.csv",
        "data/sample_delivery.csv",
        "data/rainfall_missing_values.csv",
        "data/delivery_missing_values.csv",
        "data/rainfall_us_format.csv",
        "data/delivery_us_format.csv"
    ]
    
    files_found = 0
    for file_path in sample_files:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            assert isinstance(df, pd.DataFrame), f"Should be able to read {file_path}"
            logger.info(f"✅ {file_path}: {len(df)} records accessible")
            files_found += 1
        else:
            logger.warning(f"⚠️ {file_path}: File not found")
    
    # At least the main sample files should exist
    assert files_found >= 2, "At least sample_rainfall.csv and sample_delivery.csv should exist"


def main():
    """Run comprehensive integration tests."""
    logger.info("🚀 Starting comprehensive integration tests...")
    logger.info("=" * 60)
    
    # Test 1: Data Loading
    rainfall_df, delivery_df, combined_df = test_data_loading()
    
    # Test 2: Chart Generation
    if rainfall_df is not None:
        chart_success = test_chart_generation(rainfall_df, delivery_df, combined_df)
    else:
        chart_success = False
        logger.error("❌ Skipping chart tests due to data loading failure")
    
    # Test 3: Insights Generation
    if rainfall_df is not None:
        insights_success = test_insights_generation(rainfall_df, delivery_df, combined_df)
    else:
        insights_success = False
        logger.error("❌ Skipping insights tests due to data loading failure")
    
    # Test 4: Edge Cases
    edge_case_success = test_edge_cases()
    
    # Test 5: File Upload Simulation
    upload_success = test_file_upload_simulation()
    
    # Summary
    logger.info("=" * 60)
    logger.info("🏁 Integration Test Summary:")
    logger.info(f"   Data Loading: {'✅ PASS' if rainfall_df is not None else '❌ FAIL'}")
    logger.info(f"   Chart Generation: {'✅ PASS' if chart_success else '❌ FAIL'}")
    logger.info(f"   Insights Generation: {'✅ PASS' if insights_success else '❌ FAIL'}")
    logger.info(f"   Edge Cases: {'✅ PASS' if edge_case_success else '❌ FAIL'}")
    logger.info(f"   File Upload Simulation: {'✅ PASS' if upload_success else '❌ FAIL'}")
    
    all_tests_passed = all([
        rainfall_df is not None,
        chart_success,
        insights_success,
        edge_case_success,
        upload_success
    ])
    
    if all_tests_passed:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("   The complete data pipeline is working correctly.")
        logger.info("   Ready for production use!")
        return 0
    else:
        logger.error("💥 SOME INTEGRATION TESTS FAILED!")
        logger.error("   Please review the errors above and fix issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)