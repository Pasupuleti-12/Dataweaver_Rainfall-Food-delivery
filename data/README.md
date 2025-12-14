# Data Directory

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
