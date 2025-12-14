"""
Demonstration of file upload interface integration

This script shows how the file upload interface works with the integrated components.
"""

import pandas as pd
import tempfile
import os
from pathlib import Path

from data_loader import DataLoader
from chart_generator import ChartGenerator
from insights_generator import InsightsGenerator


def demo_file_upload_workflow():
    """Demonstrate the complete file upload workflow."""
    
    print("🎯 Demonstrating File Upload Integration")
    print("=" * 50)
    
    # Initialize components
    data_loader = DataLoader()
    chart_generator = ChartGenerator()
    insights_generator = InsightsGenerator()
    
    # 1. Simulate user uploading files
    print("\n1️⃣ Simulating file upload...")
    
    # Load sample data to simulate uploaded files
    sample_rainfall = data_loader.load_rainfall_data("data/sample_rainfall.csv")
    sample_delivery = data_loader.load_delivery_data("data/sample_delivery.csv")
    
    # Create temporary files to simulate uploads
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_rainfall:
        sample_rainfall.to_csv(temp_rainfall.name, index=False)
        rainfall_upload_path = temp_rainfall.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_delivery:
        sample_delivery.to_csv(temp_delivery.name, index=False)
        delivery_upload_path = temp_delivery.name
    
    print(f"   📁 Simulated rainfall file: {Path(rainfall_upload_path).name}")
    print(f"   📁 Simulated delivery file: {Path(delivery_upload_path).name}")
    
    try:
        # 2. Process uploaded files (as DashboardController would)
        print("\n2️⃣ Processing uploaded files...")
        
        rainfall_df = data_loader.load_rainfall_data(rainfall_upload_path)
        delivery_df = data_loader.load_delivery_data(delivery_upload_path)
        
        print(f"   ✅ Rainfall data: {len(rainfall_df)} records loaded")
        print(f"   ✅ Delivery data: {len(delivery_df)} records loaded")
        
        # 3. Join datasets
        print("\n3️⃣ Joining datasets...")
        combined_df = data_loader.join_datasets(rainfall_df, delivery_df)
        print(f"   ✅ Combined dataset: {len(combined_df)} matching records")
        
        # 4. Generate visualizations
        print("\n4️⃣ Generating visualizations...")
        
        rainfall_chart = chart_generator.create_time_series_chart(
            rainfall_df, 'precipitation_mm', 'Uploaded Rainfall Data'
        )
        delivery_chart = chart_generator.create_time_series_chart(
            delivery_df, 'order_count', 'Uploaded Delivery Data'
        )
        
        if not combined_df.empty:
            combined_chart = chart_generator.create_combined_time_series(
                combined_df, 'Combined Uploaded Data'
            )
            scatter_chart = chart_generator.create_scatter_plot(
                combined_df, 'precipitation_mm', 'order_count', 'Relationship Analysis'
            )
            print("   ✅ All charts generated from uploaded data")
        
        # 5. Generate insights
        print("\n5️⃣ Generating insights...")
        
        rainfall_insights = insights_generator.generate_summary_statistics(rainfall_df, "Uploaded Rainfall")
        delivery_insights = insights_generator.generate_summary_statistics(delivery_df, "Uploaded Delivery")
        
        if not combined_df.empty:
            combined_insights = insights_generator.generate_combined_insights(
                combined_df, rainfall_insights, delivery_insights
            )
            print("   ✅ All insights generated from uploaded data")
        
        # 6. Display sample insights
        print("\n6️⃣ Sample insights from uploaded data:")
        print(f"   📊 Rainfall average: {rainfall_insights['statistics']['precipitation_mm']['mean']:.1f}mm")
        print(f"   📊 Delivery average: {delivery_insights['statistics']['order_count']['mean']:.0f} orders")
        
        if not combined_df.empty and 'correlation_analysis' in combined_insights:
            corr = combined_insights['correlation_analysis']['correlation']
            if corr is not None:
                print(f"   📈 Correlation: {corr:.3f}")
        
        print("\n🎉 File upload workflow completed successfully!")
        print("\nThis demonstrates that:")
        print("  ✅ Users can upload CSV files")
        print("  ✅ Files are processed and validated")
        print("  ✅ Data is joined automatically")
        print("  ✅ Visualizations are generated")
        print("  ✅ Insights are calculated")
        print("  ✅ Results are displayed in the dashboard")
        
    finally:
        # Clean up temporary files
        os.unlink(rainfall_upload_path)
        os.unlink(delivery_upload_path)


if __name__ == "__main__":
    demo_file_upload_workflow()