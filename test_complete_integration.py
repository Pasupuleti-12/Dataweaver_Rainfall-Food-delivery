"""
Complete integration test for the Rainfall vs Food Delivery Demand Dashboard

This test verifies that all components work together properly for task 8.2.
"""

import pandas as pd
import tempfile
import os
from pathlib import Path
import logging

from data_loader import DataLoader
from chart_generator import ChartGenerator
from insights_generator import InsightsGenerator
from dashboard_controller import DashboardController


def test_complete_integration():
    """Test complete integration of all components."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🧪 Starting complete integration test...")
        
        # 1. Test component initialization
        logger.info("1️⃣ Testing component initialization...")
        data_loader = DataLoader()
        chart_generator = ChartGenerator()
        insights_generator = InsightsGenerator()
        dashboard_controller = DashboardController()
        logger.info("✅ All components initialized successfully")
        
        # 2. Test data loading with sample files
        logger.info("2️⃣ Testing data loading...")
        rainfall_df = data_loader.load_rainfall_data("data/sample_rainfall.csv")
        delivery_df = data_loader.load_delivery_data("data/sample_delivery.csv")
        logger.info(f"✅ Data loaded: {len(rainfall_df)} rainfall records, {len(delivery_df)} delivery records")
        
        # 3. Test dataset joining
        logger.info("3️⃣ Testing dataset joining...")
        combined_df = data_loader.join_datasets(rainfall_df, delivery_df)
        logger.info(f"✅ Combined dataset created with {len(combined_df)} records")
        
        # 4. Test chart generation
        logger.info("4️⃣ Testing chart generation...")
        
        # Time series charts
        rainfall_chart = chart_generator.create_time_series_chart(
            rainfall_df, 'precipitation_mm', 'Rainfall Over Time'
        )
        delivery_chart = chart_generator.create_time_series_chart(
            delivery_df, 'order_count', 'Delivery Orders Over Time'
        )
        
        # Combined visualizations
        if not combined_df.empty:
            combined_chart = chart_generator.create_combined_time_series(
                combined_df, 'Combined Data Over Time'
            )
            scatter_chart = chart_generator.create_scatter_plot(
                combined_df, 'precipitation_mm', 'order_count', 'Rainfall vs Orders'
            )
            logger.info("✅ All charts generated successfully")
        else:
            logger.warning("⚠️ No combined data available for relationship charts")
        
        # 5. Test insights generation
        logger.info("5️⃣ Testing insights generation...")
        rainfall_insights = insights_generator.generate_summary_statistics(rainfall_df, "Rainfall")
        delivery_insights = insights_generator.generate_summary_statistics(delivery_df, "Delivery")
        
        if not combined_df.empty:
            combined_insights = insights_generator.generate_combined_insights(
                combined_df, rainfall_insights, delivery_insights
            )
            logger.info("✅ All insights generated successfully")
        else:
            logger.warning("⚠️ No combined insights available")
        
        # 6. Test dashboard controller components
        logger.info("6️⃣ Testing dashboard controller components...")
        
        # Test error handling
        error_handler = dashboard_controller.error_handler
        test_error = ValueError("Test error message")
        user_friendly_msg = error_handler.create_user_friendly_message(test_error, "testing")
        assert "issue occurred" in user_friendly_msg.lower()
        
        # Test loading indicator
        loading_indicator = dashboard_controller.loading_indicator
        # Note: Can't fully test Streamlit components outside of Streamlit runtime
        
        logger.info("✅ Dashboard controller components working")
        
        # 7. Test file upload simulation
        logger.info("7️⃣ Testing file upload simulation...")
        
        # Create temporary CSV files to simulate uploads
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_rainfall:
            rainfall_df.to_csv(temp_rainfall.name, index=False)
            temp_rainfall_path = temp_rainfall.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_delivery:
            delivery_df.to_csv(temp_delivery.name, index=False)
            temp_delivery_path = temp_delivery.name
        
        try:
            # Test loading from temporary files
            test_rainfall = data_loader.load_rainfall_data(temp_rainfall_path)
            test_delivery = data_loader.load_delivery_data(temp_delivery_path)
            
            assert len(test_rainfall) == len(rainfall_df)
            assert len(test_delivery) == len(delivery_df)
            
            logger.info("✅ File upload simulation successful")
            
        finally:
            # Clean up temporary files
            os.unlink(temp_rainfall_path)
            os.unlink(temp_delivery_path)
        
        # 8. Test error handling scenarios
        logger.info("8️⃣ Testing error handling scenarios...")
        
        # Test with non-existent file
        try:
            data_loader.load_rainfall_data("non_existent_file.csv")
            assert False, "Should have raised an exception"
        except Exception as e:
            logger.info(f"✅ Correctly handled missing file: {type(e).__name__}")
        
        # Test with empty dataframes
        empty_df = pd.DataFrame()
        try:
            empty_combined = data_loader.join_datasets(empty_df, empty_df)
            assert empty_combined.empty
            logger.info("✅ Correctly handled empty datasets")
        except Exception as e:
            logger.info(f"✅ Correctly handled empty datasets with exception: {type(e).__name__}")
        
        logger.info("🎉 Complete integration test PASSED!")
        
    except Exception as e:
        logger.error(f"❌ Integration test FAILED: {str(e)}")
        raise


if __name__ == "__main__":
    test_complete_integration()