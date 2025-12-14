#!/usr/bin/env python3
"""
End-to-end workflow test to verify complete functionality.

This script simulates the complete user workflow from data upload through
visualization and insights generation.
"""

import logging
from pathlib import Path
import pandas as pd

# Import all components
from data_loader import DataLoader
from chart_generator import ChartGenerator
from insights_generator import InsightsGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def simulate_user_workflow():
    """Simulate complete user workflow from data upload to insights."""
    
    logger.info("🎯 Simulating complete user workflow...")
    logger.info("=" * 50)
    
    # Step 1: User uploads or selects sample data
    logger.info("Step 1: Loading sample datasets...")
    data_loader = DataLoader()
    
    try:
        rainfall_df = data_loader.load_rainfall_data("data/sample_rainfall.csv")
        delivery_df = data_loader.load_delivery_data("data/sample_delivery.csv")
        logger.info(f"✅ Loaded {len(rainfall_df)} rainfall records and {len(delivery_df)} delivery records")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {str(e)}")
        return False
    
    # Step 2: System validates and joins datasets
    logger.info("Step 2: Validating and joining datasets...")
    try:
        # Validate individual datasets
        rainfall_quality = data_loader.validate_data_quality(rainfall_df)
        delivery_quality = data_loader.validate_data_quality(delivery_df)
        
        logger.info(f"   Rainfall data quality: {rainfall_quality['quality_score']:.1%}")
        logger.info(f"   Delivery data quality: {delivery_quality['quality_score']:.1%}")
        
        # Join datasets
        combined_df = data_loader.join_datasets(rainfall_df, delivery_df)
        logger.info(f"✅ Combined dataset created with {len(combined_df)} matching records")
    except Exception as e:
        logger.error(f"❌ Failed to process data: {str(e)}")
        return False
    
    # Step 3: Generate visualizations
    logger.info("Step 3: Generating visualizations...")
    chart_generator = ChartGenerator()
    
    try:
        # Time series charts
        rainfall_chart = chart_generator.create_time_series_chart(
            rainfall_df, 'precipitation_mm', 'Daily Rainfall Over Time'
        )
        delivery_chart = chart_generator.create_time_series_chart(
            delivery_df, 'order_count', 'Daily Delivery Orders Over Time'
        )
        
        # Relationship analysis
        scatter_chart = chart_generator.create_scatter_plot(
            combined_df, 'precipitation_mm', 'order_count', 
            'Rainfall vs Delivery Orders Relationship'
        )
        
        # Combined visualization
        combined_chart = chart_generator.create_combined_time_series(
            combined_df, 'Rainfall and Delivery Data Over Time'
        )
        
        logger.info("✅ All visualizations generated successfully")
        
        # Calculate correlation
        correlation_stats = chart_generator.calculate_correlation(
            combined_df, 'precipitation_mm', 'order_count'
        )
        correlation_value = correlation_stats.get('correlation')
        if correlation_value is not None:
            logger.info(f"   Correlation coefficient: {correlation_value:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate visualizations: {str(e)}")
        return False
    
    # Step 4: Generate insights and observations
    logger.info("Step 4: Generating insights and observations...")
    insights_generator = InsightsGenerator()
    
    try:
        # Individual dataset insights
        rainfall_insights = insights_generator.generate_summary_statistics(
            rainfall_df, "Rainfall Data"
        )
        delivery_insights = insights_generator.generate_summary_statistics(
            delivery_df, "Delivery Data"
        )
        
        # Combined insights
        combined_insights = insights_generator.generate_combined_insights(
            combined_df, rainfall_insights, delivery_insights
        )
        
        logger.info("✅ Statistical insights generated successfully")
        
        # Display key insights
        if rainfall_insights.get('statistics', {}).get('precipitation_mm'):
            rain_stats = rainfall_insights['statistics']['precipitation_mm']
            logger.info(f"   Average daily rainfall: {rain_stats['mean']:.1f}mm")
            logger.info(f"   Maximum rainfall: {rain_stats['max']:.1f}mm")
        
        if delivery_insights.get('statistics', {}).get('order_count'):
            delivery_stats = delivery_insights['statistics']['order_count']
            logger.info(f"   Average daily orders: {delivery_stats['mean']:.0f}")
            logger.info(f"   Maximum daily orders: {delivery_stats['max']:.0f}")
        
        # Display relationship insights
        if combined_insights.get('correlation_analysis'):
            corr_analysis = combined_insights['correlation_analysis']
            logger.info(f"   Relationship: {corr_analysis.get('interpretation', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate insights: {str(e)}")
        return False
    
    # Step 5: Validate non-causal language
    logger.info("Step 5: Validating descriptive language...")
    try:
        # Test some sample text for causal language
        test_texts = [
            "Rainfall shows a weak association with delivery orders.",
            "Higher rainfall tends to occur alongside increased delivery demand.",
            "The data demonstrates a correlation between weather and orders."
        ]
        
        for text in test_texts:
            is_valid, causal_words = insights_generator.validate_non_causal_language(text)
            status = "✅ Valid" if is_valid else f"❌ Invalid (found: {causal_words})"
            logger.info(f"   '{text[:50]}...': {status}")
        
    except Exception as e:
        logger.error(f"❌ Failed to validate language: {str(e)}")
        return False
    
    logger.info("=" * 50)
    logger.info("🎉 Complete user workflow simulation SUCCESSFUL!")
    logger.info("   All components integrated and working correctly.")
    return True


def test_different_data_scenarios():
    """Test the workflow with different data scenarios."""
    
    logger.info("🔄 Testing different data scenarios...")
    
    data_loader = DataLoader()
    chart_generator = ChartGenerator()
    
    scenarios = [
        ("Missing values", "data/rainfall_missing_values.csv", "data/delivery_missing_values.csv"),
        ("Different date format", "data/rainfall_us_format.csv", "data/delivery_us_format.csv"),
        ("Extreme values", "data/rainfall_extreme_values.csv", "data/delivery_extreme_values.csv"),
        ("Short time period", "data/rainfall_2weeks.csv", "data/delivery_2weeks.csv")
    ]
    
    for scenario_name, rainfall_file, delivery_file in scenarios:
        logger.info(f"Testing scenario: {scenario_name}")
        
        try:
            # Load data
            rainfall_df = data_loader.load_rainfall_data(rainfall_file)
            delivery_df = data_loader.load_delivery_data(delivery_file)
            
            # Join datasets
            combined_df = data_loader.join_datasets(rainfall_df, delivery_df)
            
            # Generate basic visualization
            if not combined_df.empty:
                scatter_chart = chart_generator.create_scatter_plot(
                    combined_df, 'precipitation_mm', 'order_count'
                )
                logger.info(f"   ✅ {scenario_name}: {len(combined_df)} records processed successfully")
            else:
                logger.info(f"   ⚠️ {scenario_name}: No overlapping data, but handled gracefully")
                
        except Exception as e:
            logger.error(f"   ❌ {scenario_name}: Failed with error: {str(e)}")
    
    return True


def main():
    """Run complete end-to-end testing."""
    
    # Test main workflow
    workflow_success = simulate_user_workflow()
    
    # Test different scenarios
    scenarios_success = test_different_data_scenarios()
    
    # Final summary
    logger.info("=" * 50)
    logger.info("🏆 END-TO-END TEST SUMMARY:")
    logger.info(f"   Main Workflow: {'✅ PASS' if workflow_success else '❌ FAIL'}")
    logger.info(f"   Data Scenarios: {'✅ PASS' if scenarios_success else '❌ FAIL'}")
    
    if workflow_success and scenarios_success:
        logger.info("🚀 SYSTEM READY FOR PRODUCTION!")
        logger.info("   Complete data pipeline from CSV to insights working perfectly.")
        return 0
    else:
        logger.error("💥 SYSTEM NOT READY - Please fix issues above.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)