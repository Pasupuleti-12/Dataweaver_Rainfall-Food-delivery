"""
Property-based tests for InsightsGenerator functionality

Tests the insights generation components to ensure they properly create statistical
summaries and descriptive observations without causal language as specified in requirements.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, Any

from insights_generator import InsightsGenerator


# Hypothesis strategies for generating test data
valid_dates = st.dates(min_value=date(2020, 1, 1), max_value=date(2025, 12, 31))
valid_precipitation = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
valid_order_counts = st.integers(min_value=0, max_value=100000)


class TestInsightsGenerator:
    """Test cases for InsightsGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = InsightsGenerator()
    
    def test_empty_dataframe_handling(self):
        """Empty DataFrames should be handled gracefully"""
        empty_df = pd.DataFrame()
        
        summary = self.generator.generate_summary_statistics(empty_df, "Test Dataset")
        
        assert summary['record_count'] == 0
        assert summary['dataset_name'] == "Test Dataset"
        assert summary['statistics'] == {}
        assert any(word in summary['data_quality']['message'].lower() for word in ['empty', 'no records', 'contains no'])
    
    def test_basic_summary_statistics(self):
        """Basic summary statistics should be calculated correctly"""
        df = pd.DataFrame({
            'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            'precipitation_mm': [10.0, 20.0, 30.0],
            'order_count': [100, 200, 300]
        })
        
        summary = self.generator.generate_summary_statistics(df, "Test Dataset")
        
        assert summary['record_count'] == 3
        assert 'precipitation_mm' in summary['statistics']
        assert 'order_count' in summary['statistics']
        
        # Check precipitation statistics
        precip_stats = summary['statistics']['precipitation_mm']
        assert precip_stats['count'] == 3
        assert precip_stats['mean'] == 20.0
        assert precip_stats['min'] == 10.0
        assert precip_stats['max'] == 30.0
        
        # Check order count statistics
        order_stats = summary['statistics']['order_count']
        assert order_stats['count'] == 3
        assert order_stats['mean'] == 200.0
        assert order_stats['min'] == 100
        assert order_stats['max'] == 300
    
    def test_missing_data_handling(self):
        """Missing data should be handled appropriately in statistics"""
        df = pd.DataFrame({
            'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            'precipitation_mm': [10.0, np.nan, 30.0],
            'order_count': [100, 200, np.nan]
        })
        
        summary = self.generator.generate_summary_statistics(df, "Test Dataset")
        
        # Check that missing values are properly counted
        precip_stats = summary['statistics']['precipitation_mm']
        assert precip_stats['count'] == 2  # Only 2 valid values
        assert precip_stats['missing_count'] == 1
        assert precip_stats['missing_percentage'] == (1/3) * 100
        
        order_stats = summary['statistics']['order_count']
        assert order_stats['count'] == 2  # Only 2 valid values
        assert order_stats['missing_count'] == 1
    
    def test_non_causal_language_validation(self):
        """Language validation should detect causal words"""
        # Test text with causal language
        causal_text = "Rain causes more delivery orders because people stay inside."
        is_valid, found_words = self.generator.validate_non_causal_language(causal_text)
        
        assert not is_valid
        assert 'causes' in found_words
        assert 'because' in found_words
        
        # Test text without causal language
        descriptive_text = "Rain is associated with higher delivery orders. People tend to order more on rainy days."
        is_valid, found_words = self.generator.validate_non_causal_language(descriptive_text)
        
        assert is_valid
        assert len(found_words) == 0
    
    def test_correlation_insights_basic(self):
        """Basic correlation insights should be calculated correctly"""
        df = pd.DataFrame({
            'precipitation_mm': [1, 2, 3, 4, 5],
            'order_count': [10, 20, 30, 40, 50]  # Perfect positive correlation
        })
        
        insights = self.generator._calculate_correlation_insights(df, 'precipitation_mm', 'order_count')
        
        assert insights['correlation'] is not None
        assert abs(insights['correlation'] - 1.0) < 0.001  # Should be close to 1.0
        assert insights['sample_size'] == 5
        assert insights['strength'] == 'very strong'
        assert 'association' in insights['descriptive_text'].lower()
        
        # Validate that the descriptive text doesn't contain causal language
        is_valid, _ = self.generator.validate_non_causal_language(insights['descriptive_text'])
        assert is_valid


# **Feature: rainfall-delivery-dashboard, Property 12: Summary statistics accuracy**
# **Validates: Requirements 5.1**
@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation, valid_order_counts),
        min_size=1,
        max_size=20
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_summary_statistics_accuracy(test_data):
    """
    Property 12: Summary statistics accuracy
    For any dataset, calculated summary statistics (mean, median, standard deviation) 
    should match standard statistical calculations.
    """
    generator = InsightsGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in test_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data
    assume(len(df) >= 1)
    
    # Generate summary statistics
    summary = generator.generate_summary_statistics(df, "Test Dataset")
    
    # Property: Summary should be generated successfully
    assert isinstance(summary, dict)
    assert summary['dataset_name'] == "Test Dataset"
    assert summary['record_count'] == len(df)
    
    # Property: Statistics should be calculated for numeric columns
    assert 'precipitation_mm' in summary['statistics']
    assert 'order_count' in summary['statistics']
    
    # Property: Precipitation statistics should match pandas calculations
    precip_data = df['precipitation_mm'].dropna()
    if len(precip_data) > 0:
        precip_stats = summary['statistics']['precipitation_mm']
        
        # Verify count
        assert precip_stats['count'] == len(precip_data)
        
        # Verify mean (within floating point precision)
        expected_mean = float(precip_data.mean())
        assert abs(precip_stats['mean'] - expected_mean) < 1e-10
        
        # Verify median
        expected_median = float(precip_data.median())
        assert abs(precip_stats['median'] - expected_median) < 1e-10
        
        # Verify min and max
        assert precip_stats['min'] == float(precip_data.min())
        assert precip_stats['max'] == float(precip_data.max())
        
        # Verify standard deviation (if more than one data point)
        if len(precip_data) > 1:
            expected_std = float(precip_data.std())
            assert abs(precip_stats['std'] - expected_std) < 1e-10
        else:
            assert precip_stats['std'] == 0.0
        
        # Verify quartiles
        expected_q1 = float(precip_data.quantile(0.25))
        expected_q3 = float(precip_data.quantile(0.75))
        assert abs(precip_stats['q1'] - expected_q1) < 1e-10
        assert abs(precip_stats['q3'] - expected_q3) < 1e-10
        
        # Verify missing data counts
        expected_missing = len(df) - len(precip_data)
        assert precip_stats['missing_count'] == expected_missing
        
        expected_missing_pct = (expected_missing / len(df)) * 100
        assert abs(precip_stats['missing_percentage'] - expected_missing_pct) < 1e-10
    
    # Property: Order count statistics should match pandas calculations
    order_data = df['order_count'].dropna()
    if len(order_data) > 0:
        order_stats = summary['statistics']['order_count']
        
        # Verify count
        assert order_stats['count'] == len(order_data)
        
        # Verify mean
        expected_mean = float(order_data.mean())
        assert abs(order_stats['mean'] - expected_mean) < 1e-10
        
        # Verify median
        expected_median = float(order_data.median())
        assert abs(order_stats['median'] - expected_median) < 1e-10
        
        # Verify min and max
        assert order_stats['min'] == float(order_data.min())
        assert order_stats['max'] == float(order_data.max())
        
        # Verify standard deviation (if more than one data point)
        if len(order_data) > 1:
            expected_std = float(order_data.std())
            assert abs(order_stats['std'] - expected_std) < 1e-10
        else:
            assert order_stats['std'] == 0.0
    
    # Property: Data quality assessment should be present
    assert 'data_quality' in summary
    assert 'completeness' in summary['data_quality']
    assert 'message' in summary['data_quality']
    
    # Property: Observations should be generated
    assert 'observations' in summary
    assert isinstance(summary['observations'], list)
    
    # Property: All numeric values should be finite (no NaN or infinity)
    for col_name, col_stats in summary['statistics'].items():
        if col_stats['count'] > 0:
            for stat_name, stat_value in col_stats.items():
                if stat_name not in ['missing_count', 'count'] and stat_value is not None:
                    assert np.isfinite(stat_value), f"Non-finite value in {col_name}.{stat_name}: {stat_value}"
    
    # Property: Percentages should be between 0 and 100
    for col_name, col_stats in summary['statistics'].items():
        if 'missing_percentage' in col_stats:
            pct = col_stats['missing_percentage']
            assert 0 <= pct <= 100, f"Invalid percentage: {pct}"
    
    # Property: Counts should be non-negative integers
    for col_name, col_stats in summary['statistics'].items():
        assert col_stats['count'] >= 0
        assert col_stats['missing_count'] >= 0
        assert col_stats['count'] + col_stats['missing_count'] == len(df)


# Test with missing data specifically
@given(
    st.lists(
        st.tuples(
            valid_dates,
            st.one_of(valid_precipitation, st.none()),  # Allow None for missing precipitation
            st.one_of(valid_order_counts, st.none())    # Allow None for missing orders
        ),
        min_size=1,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=15, deadline=None)
def test_property_summary_statistics_accuracy_with_missing_data(test_data_with_missing):
    """
    Property 12: Summary statistics accuracy (with missing data)
    For any dataset with missing values, calculated summary statistics should 
    accurately reflect only the valid (non-missing) data points.
    """
    generator = InsightsGenerator()
    
    # Create DataFrame with potential missing values
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in test_data_with_missing
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have any data
    assume(len(df) >= 1)
    
    # Generate summary statistics
    summary = generator.generate_summary_statistics(df, "Test Dataset with Missing Data")
    
    # Property: Summary should handle missing data gracefully
    assert isinstance(summary, dict)
    assert summary['record_count'] == len(df)
    
    # Property: Statistics should only include valid (non-missing) data
    for col in ['precipitation_mm', 'order_count']:
        if col in df.columns:
            col_data = df[col].dropna()
            col_stats = summary['statistics'][col]
            
            # Property: Count should match number of non-missing values
            assert col_stats['count'] == len(col_data)
            
            # Property: Missing count should be accurate
            expected_missing = len(df) - len(col_data)
            assert col_stats['missing_count'] == expected_missing
            
            # Property: If all data is missing, statistics should be None or 0
            if len(col_data) == 0:
                assert col_stats['mean'] is None
                assert col_stats['median'] is None
                assert col_stats['min'] is None
                assert col_stats['max'] is None
                assert col_stats['missing_percentage'] == 100.0
            else:
                # Property: Statistics should match pandas calculations on valid data only
                assert abs(col_stats['mean'] - float(col_data.mean())) < 1e-10
                assert abs(col_stats['median'] - float(col_data.median())) < 1e-10
                assert col_stats['min'] == float(col_data.min())
                assert col_stats['max'] == float(col_data.max())
                
                # Property: Missing percentage should be calculated correctly
                expected_pct = (expected_missing / len(df)) * 100
                assert abs(col_stats['missing_percentage'] - expected_pct) < 1e-10
    
    # Property: Data quality should reflect missing data issues
    data_quality = summary['data_quality']
    
    # Check if there are any missing values in the dataset
    has_missing_data = df.isnull().any().any()
    
    if has_missing_data:
        # Should have completeness less than 100%
        assert data_quality['completeness'] < 100.0
        # Should have quality issues listed
        assert len(data_quality['issues']) > 0
    else:
        # Should have perfect completeness
        assert data_quality['completeness'] == 100.0


# Test edge cases
def test_property_summary_statistics_edge_cases():
    """Test edge cases for summary statistics accuracy"""
    generator = InsightsGenerator()
    
    # Test with single data point
    single_point_df = pd.DataFrame({
        'date': [date(2023, 1, 1)],
        'precipitation_mm': [15.5],
        'order_count': [100]
    })
    
    summary = generator.generate_summary_statistics(single_point_df, "Single Point")
    
    # Property: Single point statistics should be correct
    precip_stats = summary['statistics']['precipitation_mm']
    assert precip_stats['count'] == 1
    assert precip_stats['mean'] == 15.5
    assert precip_stats['median'] == 15.5
    assert precip_stats['min'] == 15.5
    assert precip_stats['max'] == 15.5
    assert precip_stats['std'] == 0.0  # Standard deviation of single point is 0
    
    # Test with identical values
    identical_df = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        'precipitation_mm': [10.0, 10.0, 10.0],
        'order_count': [200, 200, 200]
    })
    
    summary = generator.generate_summary_statistics(identical_df, "Identical Values")
    
    # Property: Identical values should have zero standard deviation
    precip_stats = summary['statistics']['precipitation_mm']
    assert precip_stats['std'] == 0.0
    assert precip_stats['mean'] == 10.0
    assert precip_stats['min'] == precip_stats['max'] == 10.0
    
    order_stats = summary['statistics']['order_count']
    assert order_stats['std'] == 0.0
    assert order_stats['mean'] == 200.0
    assert order_stats['min'] == order_stats['max'] == 200


# **Feature: rainfall-delivery-dashboard, Property 13: Non-causal language enforcement**
# **Validates: Requirements 5.2, 5.5**
@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation, valid_order_counts),
        min_size=2,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=25, deadline=None)
def test_property_non_causal_language_enforcement(test_data):
    """
    Property 13: Non-causal language enforcement
    For any generated text output, the content should not contain words or phrases 
    implying causation between variables.
    """
    generator = InsightsGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in test_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data
    assume(len(df) >= 2)
    
    # Generate individual summaries
    rainfall_df = df[['date', 'precipitation_mm']].copy()
    delivery_df = df[['date', 'order_count']].copy()
    
    rainfall_summary = generator.generate_summary_statistics(rainfall_df, "Rainfall Data")
    delivery_summary = generator.generate_summary_statistics(delivery_df, "Delivery Data")
    
    # Generate combined insights
    combined_insights = generator.generate_combined_insights(df, rainfall_summary, delivery_summary)
    
    # Property: All generated text should be free of causal language
    texts_to_check = []
    
    # Collect all text from rainfall summary
    texts_to_check.extend(rainfall_summary.get('observations', []))
    if rainfall_summary.get('data_quality', {}).get('message'):
        texts_to_check.append(rainfall_summary['data_quality']['message'])
    
    # Collect all text from delivery summary
    texts_to_check.extend(delivery_summary.get('observations', []))
    if delivery_summary.get('data_quality', {}).get('message'):
        texts_to_check.append(delivery_summary['data_quality']['message'])
    
    # Collect all text from combined insights
    texts_to_check.extend(combined_insights.get('relationship_observations', []))
    texts_to_check.extend(combined_insights.get('data_quality_notes', []))
    
    if combined_insights.get('summary_message'):
        texts_to_check.append(combined_insights['summary_message'])
    
    if combined_insights.get('correlation_analysis', {}).get('descriptive_text'):
        texts_to_check.append(combined_insights['correlation_analysis']['descriptive_text'])
    
    if combined_insights.get('correlation_analysis', {}).get('interpretation'):
        texts_to_check.append(combined_insights['correlation_analysis']['interpretation'])
    
    # Property: No text should contain causal language
    for text in texts_to_check:
        if text and isinstance(text, str) and len(text.strip()) > 0:
            is_valid, found_causal_words = generator.validate_non_causal_language(text)
            
            # If causal words are found, provide detailed error message
            if not is_valid:
                pytest.fail(
                    f"Causal language detected in generated text: '{text}'\n"
                    f"Found causal words: {found_causal_words}\n"
                    f"This violates requirements 5.2 and 5.5 for non-causal language."
                )
    
    # Property: Correlation analysis text should use descriptive language
    if combined_insights.get('correlation_analysis'):
        corr_analysis = combined_insights['correlation_analysis']
        
        if corr_analysis.get('descriptive_text'):
            descriptive_text = corr_analysis['descriptive_text'].lower()
            
            # Should contain descriptive terms, not causal ones
            descriptive_indicators = [
                'association', 'associated', 'relationship', 'pattern', 'tend to',
                'appears', 'shows', 'displays', 'exhibits', 'correlation',
                'coincide', 'alongside', 'together'
            ]
            
            # At least one descriptive indicator should be present
            has_descriptive_language = any(
                indicator in descriptive_text for indicator in descriptive_indicators
            )
            
            # Should not contain causal language (already checked above, but double-check key ones)
            causal_indicators = [
                'causes', 'caused', 'because', 'due to', 'results in', 'leads to',
                'triggers', 'influences', 'affects', 'drives', 'determines'
            ]
            
            has_causal_language = any(
                causal in descriptive_text for causal in causal_indicators
            )
            
            assert has_descriptive_language, f"Correlation text lacks descriptive language: {descriptive_text}"
            assert not has_causal_language, f"Correlation text contains causal language: {descriptive_text}"
    
    # Property: Summary message should avoid causal claims
    if combined_insights.get('summary_message'):
        summary_text = combined_insights['summary_message'].lower()
        
        # Should explicitly mention descriptive nature
        descriptive_disclaimers = [
            'descriptive', 'does not establish', 'patterns', 'insights',
            'not causal', 'no causal', 'association', 'relationship'
        ]
        
        # Should contain some form of disclaimer about causation
        has_disclaimer = any(
            disclaimer in summary_text for disclaimer in descriptive_disclaimers
        )
        
        # This is a strong requirement - summaries should always clarify the descriptive nature
        assert has_disclaimer, f"Summary lacks descriptive disclaimer: {combined_insights['summary_message']}"
    
    # Property: Data quality notes should mention analysis limitations
    quality_notes = combined_insights.get('data_quality_notes', [])
    
    # Should include a note about descriptive vs causal analysis
    causal_limitation_mentioned = False
    for note in quality_notes:
        if note and isinstance(note, str):
            note_lower = note.lower()
            if any(phrase in note_lower for phrase in [
                'descriptive only', 'does not establish causal', 'not causal',
                'no causal', 'descriptive analysis'
            ]):
                causal_limitation_mentioned = True
                break
    
    # This is a requirement - should always mention the limitation
    assert causal_limitation_mentioned, f"Data quality notes lack causal limitation disclaimer: {quality_notes}"


# Test specific causal language detection
@given(st.text(min_size=10, max_size=200))
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
def test_property_causal_language_detection_accuracy(random_text):
    """
    Property 13: Non-causal language enforcement (detection accuracy)
    The causal language detection should accurately identify causal words and phrases.
    """
    generator = InsightsGenerator()
    
    # Property: Text without causal words should pass validation
    # First check if the random text contains any causal words
    is_valid_original, found_words_original = generator.validate_non_causal_language(random_text)
    
    # If the original text is valid (no causal words), it should remain valid
    if is_valid_original:
        assert len(found_words_original) == 0
    else:
        # If invalid, should have found some causal words
        assert len(found_words_original) > 0
        # Each found word should actually be in the causal words set
        for word in found_words_original:
            assert word in generator.causal_words
    
    # Property: Adding known causal words should make text invalid
    # Test by adding a known causal word
    causal_word = 'causes'
    text_with_causal = random_text + f" This {causal_word} problems."
    
    is_valid_with_causal, found_words_with_causal = generator.validate_non_causal_language(text_with_causal)
    
    # Should definitely be invalid now
    assert not is_valid_with_causal
    assert causal_word in found_words_with_causal
    
    # Property: Case insensitive detection should work
    text_with_causal_caps = random_text + f" This {causal_word.upper()} problems."
    is_valid_caps, found_words_caps = generator.validate_non_causal_language(text_with_causal_caps)
    
    assert not is_valid_caps
    assert causal_word in found_words_caps  # Should find the lowercase version


# Test with realistic examples
def test_property_non_causal_language_realistic_examples():
    """Test non-causal language enforcement with realistic text examples"""
    generator = InsightsGenerator()
    
    # Examples of causal language that should be detected
    causal_examples = [
        "Rain causes people to order more food delivery.",
        "Higher precipitation leads to increased delivery orders.",
        "Weather affects customer behavior significantly.",
        "Rainy days result in more orders because people stay inside.",
        "The correlation is due to people avoiding going out.",
        "This pattern stems from weather influencing decisions."
    ]
    
    # Examples of acceptable descriptive language
    descriptive_examples = [
        "Rain is associated with higher delivery order volumes.",
        "Precipitation shows a positive correlation with order counts.",
        "Rainy days tend to coincide with increased delivery activity.",
        "Higher rainfall appears alongside elevated order volumes.",
        "The data shows a pattern of association between weather and orders.",
        "Orders and rainfall demonstrate a statistical relationship."
    ]
    
    # Property: All causal examples should be flagged as invalid
    for causal_text in causal_examples:
        is_valid, found_words = generator.validate_non_causal_language(causal_text)
        assert not is_valid, f"Failed to detect causal language in: '{causal_text}'"
        assert len(found_words) > 0, f"No causal words found in: '{causal_text}'"
    
    # Property: All descriptive examples should be valid
    for descriptive_text in descriptive_examples:
        is_valid, found_words = generator.validate_non_causal_language(descriptive_text)
        assert is_valid, f"Incorrectly flagged descriptive language as causal: '{descriptive_text}' (found: {found_words})"
        assert len(found_words) == 0, f"False positive causal words in: '{descriptive_text}'"


# Test the complete insights generation pipeline for causal language
@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation, valid_order_counts),
        min_size=5,
        max_size=10
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10, deadline=None)
def test_property_complete_insights_pipeline_non_causal(test_data):
    """
    Property 13: Non-causal language enforcement (complete pipeline)
    The entire insights generation pipeline should produce only non-causal language.
    """
    generator = InsightsGenerator()
    
    # Create DataFrame with some variation to ensure interesting insights
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in test_data
    ])
    
    # Remove duplicates and ensure we have enough data
    df = df.drop_duplicates(subset=['date'], keep='first')
    assume(len(df) >= 5)
    
    # Generate complete analysis
    rainfall_df = df[['date', 'precipitation_mm']].copy()
    delivery_df = df[['date', 'order_count']].copy()
    
    rainfall_summary = generator.generate_summary_statistics(rainfall_df, "Rainfall Data")
    delivery_summary = generator.generate_summary_statistics(delivery_df, "Delivery Data")
    combined_insights = generator.generate_combined_insights(df, rainfall_summary, delivery_summary)
    
    # Format for display (this also generates text)
    formatted_insights = generator.format_insights_for_display(combined_insights)
    
    # Property: The complete formatted output should be causal-language free
    is_valid, found_causal_words = generator.validate_non_causal_language(formatted_insights)
    
    if not is_valid:
        pytest.fail(
            f"Complete insights pipeline generated causal language:\n"
            f"Found causal words: {found_causal_words}\n"
            f"In formatted text: {formatted_insights[:500]}..."
        )
    
    # Property: The formatted output should contain substantive content
    assert len(formatted_insights.strip()) > 50, "Formatted insights should contain substantive content"
    
    # Property: Should contain section headers and structured information
    assert "**" in formatted_insights, "Formatted insights should contain section headers"
    
    # Property: Should mention the descriptive nature of the analysis
    formatted_lower = formatted_insights.lower()
    descriptive_mentions = [
        'descriptive', 'association', 'relationship', 'pattern', 'correlation'
    ]
    
    has_descriptive_content = any(
        mention in formatted_lower for mention in descriptive_mentions
    )
    
    assert has_descriptive_content, "Formatted insights should mention descriptive analysis concepts"


# **Feature: rainfall-delivery-dashboard, Property 14: Descriptive-only insights**
# **Validates: Requirements 5.3**
@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation, valid_order_counts),
        min_size=3,
        max_size=20
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_descriptive_only_insights(test_data):
    """
    Property 14: Descriptive-only insights
    For any generated insights, the content should be limited to descriptive statistics 
    and observable patterns without predictive claims.
    """
    generator = InsightsGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in test_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data
    assume(len(df) >= 3)
    
    # Generate individual summaries
    rainfall_df = df[['date', 'precipitation_mm']].copy()
    delivery_df = df[['date', 'order_count']].copy()
    
    rainfall_summary = generator.generate_summary_statistics(rainfall_df, "Rainfall Data")
    delivery_summary = generator.generate_summary_statistics(delivery_df, "Delivery Data")
    
    # Generate combined insights
    combined_insights = generator.generate_combined_insights(df, rainfall_summary, delivery_summary)
    
    # Collect all generated text for analysis
    all_insights_text = []
    
    # Collect observations from individual summaries
    all_insights_text.extend(rainfall_summary.get('observations', []))
    all_insights_text.extend(delivery_summary.get('observations', []))
    
    # Collect observations from combined insights
    all_insights_text.extend(combined_insights.get('relationship_observations', []))
    all_insights_text.extend(combined_insights.get('data_quality_notes', []))
    
    if combined_insights.get('summary_message'):
        all_insights_text.append(combined_insights['summary_message'])
    
    if combined_insights.get('correlation_analysis', {}).get('descriptive_text'):
        all_insights_text.append(combined_insights['correlation_analysis']['descriptive_text'])
    
    if combined_insights.get('correlation_analysis', {}).get('interpretation'):
        all_insights_text.append(combined_insights['correlation_analysis']['interpretation'])
    
    # Property: All insights should be descriptive only (no predictive language)
    predictive_words = [
        'will', 'would', 'should', 'predict', 'forecast', 'expect', 'anticipate',
        'future', 'next', 'upcoming', 'likely to happen', 'going to', 'shall',
        'estimate', 'project', 'extrapolate', 'trend suggests', 'indicates that',
        'implies that', 'means that', 'suggests that', 'points to', 'leads us to believe'
    ]
    
    for insight_text in all_insights_text:
        if insight_text and isinstance(insight_text, str) and len(insight_text.strip()) > 0:
            insight_lower = insight_text.lower()
            
            # Check for predictive language
            found_predictive = []
            for pred_word in predictive_words:
                if pred_word in insight_lower:
                    found_predictive.append(pred_word)
            
            assert len(found_predictive) == 0, \
                f"Insight contains predictive language: '{insight_text}'\nFound predictive words: {found_predictive}"
    
    # Property: Insights should contain descriptive language
    descriptive_indicators = [
        'shows', 'displays', 'exhibits', 'demonstrates', 'indicates', 'reveals',
        'contains', 'includes', 'has', 'ranges from', 'varies between', 'spans',
        'average', 'mean', 'median', 'minimum', 'maximum', 'standard deviation',
        'observed', 'recorded', 'measured', 'found', 'present', 'available',
        'pattern', 'trend', 'distribution', 'variation', 'consistency'
    ]
    
    # At least some insights should contain descriptive language
    descriptive_insights_count = 0
    for insight_text in all_insights_text:
        if insight_text and isinstance(insight_text, str) and len(insight_text.strip()) > 0:
            insight_lower = insight_text.lower()
            
            has_descriptive = any(desc_word in insight_lower for desc_word in descriptive_indicators)
            if has_descriptive:
                descriptive_insights_count += 1
    
    # Should have at least some descriptive content
    assert descriptive_insights_count > 0, f"Should have descriptive insights, found {len(all_insights_text)} total insights"
    
    # Property: Statistical summaries should focus on descriptive statistics only
    for dataset_name, summary in [('rainfall', rainfall_summary), ('delivery', delivery_summary)]:
        statistics = summary.get('statistics', {})
        
        for col_name, col_stats in statistics.items():
            # Should contain standard descriptive statistics
            expected_stats = ['count', 'mean', 'median', 'std', 'min', 'max', 'q1', 'q3']
            for stat_name in expected_stats:
                assert stat_name in col_stats, f"{dataset_name} {col_name} should have {stat_name} statistic"
            
            # Should not contain predictive statistics
            predictive_stats = ['forecast', 'prediction', 'future_value', 'trend_projection']
            for pred_stat in predictive_stats:
                assert pred_stat not in col_stats, f"{dataset_name} {col_name} should not have predictive statistic {pred_stat}"
    
    # Property: Correlation analysis should be descriptive, not predictive
    if combined_insights.get('correlation_analysis'):
        corr_analysis = combined_insights['correlation_analysis']
        
        if corr_analysis.get('descriptive_text'):
            desc_text = corr_analysis['descriptive_text'].lower()
            
            # Should use descriptive correlation language
            descriptive_corr_terms = [
                'association', 'relationship', 'correlation', 'pattern', 'tend to',
                'appears', 'shows', 'displays', 'observed', 'found'
            ]
            
            has_descriptive_corr = any(term in desc_text for term in descriptive_corr_terms)
            assert has_descriptive_corr, f"Correlation analysis should use descriptive language: {corr_analysis['descriptive_text']}"
            
            # Should not use predictive correlation language
            predictive_corr_terms = [
                'will increase', 'will decrease', 'predicts', 'forecasts', 'expects',
                'should result in', 'leads to future', 'indicates future'
            ]
            
            has_predictive_corr = any(term in desc_text for term in predictive_corr_terms)
            assert not has_predictive_corr, f"Correlation analysis should not use predictive language: {corr_analysis['descriptive_text']}"


@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation),
        min_size=5,
        max_size=15
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=15, deadline=None)
def test_property_descriptive_only_insights_individual_datasets(rainfall_data):
    """
    Property 14: Descriptive-only insights (individual datasets)
    Individual dataset insights should be purely descriptive without predictive elements.
    """
    generator = InsightsGenerator()
    
    # Create DataFrame from generated data
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip}
        for test_date, precip in rainfall_data
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have enough data
    assume(len(df) >= 5)
    
    # Generate summary statistics
    summary = generator.generate_summary_statistics(df, "Test Rainfall Data")
    
    # Property: Summary should contain descriptive observations only
    observations = summary.get('observations', [])
    assert len(observations) > 0, "Should generate some observations"
    
    for observation in observations:
        if observation and isinstance(observation, str):
            obs_lower = observation.lower()
            
            # Property: Should describe current/past data, not future predictions
            temporal_descriptive = [
                'range from', 'ranges from', 'average', 'shows', 'displays',
                'contains', 'includes', 'spans', 'covers', 'measured',
                'recorded', 'observed', 'found'
            ]
            
            temporal_predictive = [
                'will be', 'will have', 'will show', 'will increase', 'will decrease',
                'expected to', 'likely to be', 'should be', 'predicted to',
                'forecasted', 'projected', 'anticipated'
            ]
            
            # Should contain some descriptive temporal language
            has_descriptive_temporal = any(term in obs_lower for term in temporal_descriptive)
            
            # Should not contain predictive temporal language
            has_predictive_temporal = any(term in obs_lower for term in temporal_predictive)
            
            assert not has_predictive_temporal, \
                f"Observation should not contain predictive temporal language: '{observation}'"
            
            # Property: Should focus on data characteristics, not implications
            characteristic_terms = [
                'variability', 'variation', 'consistency', 'pattern', 'distribution',
                'range', 'spread', 'deviation', 'average', 'typical', 'common'
            ]
            
            implication_terms = [
                'this means', 'this implies', 'this suggests that', 'therefore',
                'as a result', 'consequently', 'this indicates that', 'this shows that'
            ]
            
            has_implications = any(term in obs_lower for term in implication_terms)
            assert not has_implications, \
                f"Observation should describe characteristics, not implications: '{observation}'"
    
    # Property: Data quality messages should be descriptive
    data_quality = summary.get('data_quality', {})
    if data_quality.get('message'):
        quality_msg = data_quality['message'].lower()
        
        # Should describe current data quality state
        quality_descriptive = [
            'shows', 'has', 'contains', 'includes', 'displays', 'exhibits',
            'quality', 'complete', 'missing', 'available', 'records'
        ]
        
        has_quality_descriptive = any(term in quality_msg for term in quality_descriptive)
        assert has_quality_descriptive, f"Data quality message should be descriptive: {data_quality['message']}"
        
        # Should not make predictions about data quality
        quality_predictive = [
            'will improve', 'should be better', 'expected to have', 'likely to contain',
            'predicted quality', 'future completeness'
        ]
        
        has_quality_predictive = any(term in quality_msg for term in quality_predictive)
        assert not has_quality_predictive, f"Data quality message should not be predictive: {data_quality['message']}"


def test_property_descriptive_only_insights_statistical_focus():
    """
    Property 14: Descriptive-only insights (statistical focus)
    Generated insights should focus on descriptive statistics rather than inferential statistics.
    """
    generator = InsightsGenerator()
    
    # Create test data with known statistical properties
    test_data = pd.DataFrame({
        'date': [date(2023, 1, i) for i in range(1, 11)],
        'precipitation_mm': [10.0, 15.0, 8.0, 20.0, 12.0, 18.0, 9.0, 16.0, 11.0, 14.0],
        'order_count': [100, 150, 80, 200, 120, 180, 90, 160, 110, 140]
    })
    
    # Generate individual summaries
    rainfall_df = test_data[['date', 'precipitation_mm']].copy()
    delivery_df = test_data[['date', 'order_count']].copy()
    
    rainfall_summary = generator.generate_summary_statistics(rainfall_df, "Rainfall Data")
    delivery_summary = generator.generate_summary_statistics(delivery_df, "Delivery Data")
    
    # Property: Statistics should be descriptive measures
    for summary_name, summary in [('rainfall', rainfall_summary), ('delivery', delivery_summary)]:
        statistics = summary.get('statistics', {})
        
        for col_name, col_stats in statistics.items():
            # Property: Should include standard descriptive statistics
            descriptive_stats = ['mean', 'median', 'std', 'min', 'max', 'q1', 'q3']
            for stat in descriptive_stats:
                if col_stats['count'] > 0:  # Only check if there's data
                    assert stat in col_stats, f"{summary_name} should include {stat}"
                    assert col_stats[stat] is not None, f"{summary_name} {stat} should not be None"
            
            # Property: Should not include inferential statistics
            inferential_stats = [
                'confidence_interval', 'p_value', 'significance', 'hypothesis_test',
                'regression_coefficient', 'prediction_interval', 'forecast_error'
            ]
            for inf_stat in inferential_stats:
                assert inf_stat not in col_stats, f"{summary_name} should not include inferential statistic {inf_stat}"
    
    # Property: Observations should describe what is observed, not what is inferred
    all_observations = []
    all_observations.extend(rainfall_summary.get('observations', []))
    all_observations.extend(delivery_summary.get('observations', []))
    
    for observation in all_observations:
        if observation and isinstance(observation, str):
            obs_lower = observation.lower()
            
            # Should use observational language
            observational_terms = [
                'measurements', 'records', 'data', 'values', 'observations',
                'shows', 'displays', 'contains', 'includes', 'ranges'
            ]
            
            # Should not use inferential language
            inferential_terms = [
                'statistically significant', 'confidence', 'hypothesis', 'test shows',
                'evidence suggests', 'statistically', 'inference', 'conclude'
            ]
            
            has_inferential = any(term in obs_lower for term in inferential_terms)
            assert not has_inferential, \
                f"Observation should not use inferential language: '{observation}'"


def test_property_descriptive_only_insights_language_patterns():
    """
    Property 14: Descriptive-only insights (language patterns)
    Test specific language patterns to ensure insights remain descriptive.
    """
    generator = InsightsGenerator()
    
    # Create test data
    test_data = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
        'precipitation_mm': [5.0, 15.0, 25.0, 10.0],
        'order_count': [50, 150, 250, 100]
    })
    
    # Generate combined insights
    rainfall_summary = generator.generate_summary_statistics(
        test_data[['date', 'precipitation_mm']], "Rainfall"
    )
    delivery_summary = generator.generate_summary_statistics(
        test_data[['date', 'order_count']], "Delivery"
    )
    combined_insights = generator.generate_combined_insights(test_data, rainfall_summary, delivery_summary)
    
    # Collect all text for pattern analysis
    all_text = []
    
    # Add all observations and messages
    all_text.extend(rainfall_summary.get('observations', []))
    all_text.extend(delivery_summary.get('observations', []))
    all_text.extend(combined_insights.get('relationship_observations', []))
    
    if combined_insights.get('summary_message'):
        all_text.append(combined_insights['summary_message'])
    
    # Property: Should use present tense for current data description
    present_tense_indicators = [
        'shows', 'displays', 'contains', 'includes', 'has', 'ranges',
        'varies', 'spans', 'covers', 'demonstrates', 'exhibits'
    ]
    
    # Should not use future tense
    future_tense_indicators = [
        'will show', 'will display', 'will contain', 'will include', 'will have',
        'will range', 'will vary', 'will span', 'will cover', 'will demonstrate'
    ]
    
    present_tense_count = 0
    future_tense_count = 0
    
    for text in all_text:
        if text and isinstance(text, str):
            text_lower = text.lower()
            
            for present_indicator in present_tense_indicators:
                if present_indicator in text_lower:
                    present_tense_count += 1
                    break
            
            for future_indicator in future_tense_indicators:
                if future_indicator in text_lower:
                    future_tense_count += 1
                    break
    
    # Property: Should primarily use present tense, not future tense
    assert future_tense_count == 0, f"Should not use future tense in insights, found {future_tense_count} instances"
    
    # Property: Should use quantitative descriptors appropriately
    quantitative_descriptors = [
        'average', 'mean', 'median', 'minimum', 'maximum', 'range',
        'standard deviation', 'variation', 'percent', '%'
    ]
    
    qualitative_predictors = [
        'probably', 'likely', 'possibly', 'maybe', 'perhaps', 'might',
        'could be', 'seems to', 'appears to suggest', 'tends to indicate'
    ]
    
    for text in all_text:
        if text and isinstance(text, str):
            text_lower = text.lower()
            
            # Should not use uncertain qualitative predictors
            for qual_pred in qualitative_predictors:
                assert qual_pred not in text_lower, \
                    f"Should not use uncertain qualitative language: '{text}' contains '{qual_pred}'"
    
    # Property: Correlation descriptions should be factual, not speculative
    if combined_insights.get('correlation_analysis', {}).get('descriptive_text'):
        corr_text = combined_insights['correlation_analysis']['descriptive_text'].lower()
        
        # Should use factual correlation language
        factual_corr_terms = [
            'correlation', 'association', 'relationship', 'pattern'
        ]
        
        # Should not use speculative correlation language
        speculative_corr_terms = [
            'might be related', 'could be associated', 'possibly connected',
            'may indicate', 'seems to suggest', 'appears to imply'
        ]
        
        has_factual = any(term in corr_text for term in factual_corr_terms)
        has_speculative = any(term in corr_text for term in speculative_corr_terms)
        
        assert has_factual, f"Correlation text should use factual language: {combined_insights['correlation_analysis']['descriptive_text']}"
        assert not has_speculative, f"Correlation text should not be speculative: {combined_insights['correlation_analysis']['descriptive_text']}"


# **Feature: rainfall-delivery-dashboard, Property 15: Data quality reporting**
# **Validates: Requirements 5.4**
@given(
    st.lists(
        st.tuples(
            valid_dates,
            st.one_of(valid_precipitation, st.none()),  # Allow None for missing precipitation
            st.one_of(valid_order_counts, st.none())    # Allow None for missing orders
        ),
        min_size=1,
        max_size=20
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=20, deadline=None)
def test_property_data_quality_reporting(test_data_with_missing):
    """
    Property 15: Data quality reporting
    For any dataset with identifiable quality issues (missing values, outliers, 
    inconsistencies), the system should note these limitations in the output.
    """
    generator = InsightsGenerator()
    
    # Create DataFrame with potential missing values
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in test_data_with_missing
    ])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Skip if we don't have any data
    assume(len(df) >= 1)
    
    # Generate summary statistics
    summary = generator.generate_summary_statistics(df, "Test Dataset")
    
    # Property: Data quality assessment should be present
    assert 'data_quality' in summary, "Summary should include data quality assessment"
    data_quality = summary['data_quality']
    
    assert 'completeness' in data_quality, "Data quality should include completeness metric"
    assert 'issues' in data_quality, "Data quality should include issues list"
    assert 'message' in data_quality, "Data quality should include user message"
    
    # Property: Completeness should be a valid percentage
    completeness = data_quality['completeness']
    assert isinstance(completeness, (int, float)), "Completeness should be numeric"
    assert 0 <= completeness <= 100, f"Completeness should be between 0-100%: {completeness}"
    
    # Property: Issues should be identified when present
    issues = data_quality['issues']
    assert isinstance(issues, list), "Issues should be a list"
    
    # Check for missing data and verify it's reported
    has_missing_data = df.isnull().any().any()
    
    if has_missing_data:
        # Property: Missing data should be identified as an issue
        missing_issues = [issue for issue in issues if 'missing' in issue.lower()]
        assert len(missing_issues) > 0, f"Missing data should be reported as an issue, found issues: {issues}"
        
        # Property: Completeness should be less than 100% when data is missing
        assert completeness < 100, f"Completeness should be < 100% when data is missing: {completeness}%"
        
        # Property: Each column with missing data should be reported
        for col in ['precipitation_mm', 'order_count']:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    # Should find an issue mentioning this column
                    col_issues = [issue for issue in issues if col in issue or col.replace('_', ' ') in issue.lower()]
                    assert len(col_issues) > 0, f"Missing data in {col} should be reported"
    else:
        # Property: No missing data should result in high completeness
        assert completeness == 100, f"Completeness should be 100% when no data is missing: {completeness}%"
    
    # Property: User message should be informative and user-friendly
    message = data_quality['message']
    assert isinstance(message, str), "Data quality message should be a string"
    assert len(message) > 0, "Data quality message should not be empty"
    
    # Should mention the dataset name
    assert "Test Dataset" in message or "dataset" in message.lower(), "Message should reference the dataset"
    
    # Should provide actionable information
    if completeness < 100:
        # Should mention quality level or issues
        quality_terms = ['quality', 'complete', 'missing', 'issue', 'limitation']
        has_quality_terms = any(term in message.lower() for term in quality_terms)
        assert has_quality_terms, f"Quality message should mention quality aspects: {message}"
    
    # Property: Statistics should reflect data quality
    statistics = summary.get('statistics', {})
    for col_name, col_stats in statistics.items():
        if col_name in df.columns:
            expected_missing = df[col_name].isnull().sum()
            actual_missing = col_stats.get('missing_count', 0)
            
            assert actual_missing == expected_missing, \
                f"Missing count for {col_name} should match actual: expected {expected_missing}, got {actual_missing}"
            
            # Missing percentage should be calculated correctly
            expected_pct = (expected_missing / len(df)) * 100
            actual_pct = col_stats.get('missing_percentage', 0)
            assert abs(actual_pct - expected_pct) < 0.01, \
                f"Missing percentage for {col_name} should be correct: expected {expected_pct}%, got {actual_pct}%"


def test_property_data_quality_reporting_specific_issues():
    """
    Property 15: Data quality reporting (specific issue types)
    Test that specific types of data quality issues are properly identified and reported.
    """
    generator = InsightsGenerator()
    
    # Test case 1: Dataset with missing values
    missing_data_df = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
        'precipitation_mm': [10.0, np.nan, 15.0, np.nan],
        'order_count': [100, 200, np.nan, 150]
    })
    
    summary = generator.generate_summary_statistics(missing_data_df, "Missing Data Test")
    data_quality = summary['data_quality']
    
    # Property: Should identify missing values in both columns
    issues = data_quality['issues']
    precip_issues = [issue for issue in issues if 'precipitation' in issue.lower()]
    order_issues = [issue for issue in issues if 'order' in issue.lower()]
    
    assert len(precip_issues) > 0, "Should identify missing precipitation values"
    assert len(order_issues) > 0, "Should identify missing order count values"
    
    # Property: Completeness should reflect missing data
    expected_completeness = (2 / 4) * 100  # Only 2 complete rows out of 4
    actual_completeness = data_quality['completeness']
    assert abs(actual_completeness - expected_completeness) < 1, \
        f"Completeness should reflect missing data: expected ~{expected_completeness}%, got {actual_completeness}%"
    
    # Test case 2: Empty dataset
    empty_df = pd.DataFrame({'date': [], 'precipitation_mm': [], 'order_count': []})
    
    summary = generator.generate_summary_statistics(empty_df, "Empty Dataset")
    data_quality = summary['data_quality']
    
    # Property: Should identify empty dataset as a quality issue
    assert data_quality['completeness'] == 0, "Empty dataset should have 0% completeness"
    assert len(data_quality['issues']) > 0, "Empty dataset should have quality issues"
    
    empty_issues = [issue for issue in data_quality['issues'] if 'empty' in issue.lower()]
    assert len(empty_issues) > 0, "Should specifically identify empty dataset issue"
    
    # Test case 3: Complete dataset (no issues)
    complete_df = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        'precipitation_mm': [10.0, 15.0, 8.0],
        'order_count': [100, 150, 80]
    })
    
    summary = generator.generate_summary_statistics(complete_df, "Complete Dataset")
    data_quality = summary['data_quality']
    
    # Property: Complete dataset should have high quality metrics
    assert data_quality['completeness'] == 100, "Complete dataset should have 100% completeness"
    
    # Should have no missing data issues
    missing_issues = [issue for issue in data_quality['issues'] if 'missing' in issue.lower()]
    assert len(missing_issues) == 0, "Complete dataset should have no missing data issues"
    
    # Test case 4: Dataset with date gaps
    date_gap_df = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 3), date(2023, 1, 6)],  # Missing Jan 2, 4, 5
        'precipitation_mm': [10.0, 15.0, 8.0],
        'order_count': [100, 150, 80]
    })
    
    summary = generator.generate_summary_statistics(date_gap_df, "Date Gap Dataset")
    
    # Property: Should identify date coverage issues if implemented
    # (This tests the date range analysis in the summary)
    date_range = summary.get('date_range')
    if date_range:
        total_days = date_range.get('total_days', 0)
        actual_days = date_range.get('actual_days', 0)
        
        if total_days > actual_days:
            # Should note the gap in some way
            coverage_pct = date_range.get('coverage_percentage', 100)
            assert coverage_pct < 100, "Date coverage should be less than 100% when gaps exist"


@given(
    st.lists(
        st.tuples(valid_dates, valid_precipitation, valid_order_counts),
        min_size=5,
        max_size=15
    ),
    st.floats(min_value=0.0, max_value=1.0)  # Missing data probability
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=15, deadline=None)
def test_property_data_quality_reporting_varying_completeness(test_data, missing_prob):
    """
    Property 15: Data quality reporting (varying completeness levels)
    Data quality reporting should accurately reflect different levels of data completeness.
    """
    generator = InsightsGenerator()
    
    # Create DataFrame and randomly introduce missing values
    df = pd.DataFrame([
        {'date': test_date, 'precipitation_mm': precip, 'order_count': orders}
        for test_date, precip, orders in test_data
    ])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date'], keep='first')
    assume(len(df) >= 5)
    
    # Randomly introduce missing values based on missing_prob
    np.random.seed(42)  # For reproducible results
    
    for col in ['precipitation_mm', 'order_count']:
        mask = np.random.random(len(df)) < missing_prob
        df.loc[mask, col] = np.nan
    
    # Generate summary
    summary = generator.generate_summary_statistics(df, "Variable Completeness Test")
    data_quality = summary['data_quality']
    
    # Property: Completeness should accurately reflect actual data completeness
    total_cells = len(df) * 2  # Two data columns
    missing_cells = df[['precipitation_mm', 'order_count']].isnull().sum().sum()
    expected_completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    actual_completeness = data_quality['completeness']
    assert abs(actual_completeness - expected_completeness) < 1, \
        f"Completeness should match actual: expected {expected_completeness}%, got {actual_completeness}%"
    
    # Property: Quality assessment should match completeness level
    if expected_completeness >= 95:
        # High quality - should have positive language
        message = data_quality['message'].lower()
        positive_terms = ['excellent', 'very good', 'good', 'complete']
        has_positive = any(term in message for term in positive_terms)
        assert has_positive, f"High completeness should have positive quality message: {data_quality['message']}"
        
    elif expected_completeness < 50:
        # Low quality - should indicate issues
        message = data_quality['message'].lower()
        concern_terms = ['poor', 'low', 'significant', 'many', 'issues']
        has_concerns = any(term in message for term in concern_terms)
        assert has_concerns, f"Low completeness should indicate quality concerns: {data_quality['message']}"
    
    # Property: Issues should be proportional to missing data
    issues = data_quality['issues']
    
    if missing_cells > 0:
        # Should have at least one issue reported
        assert len(issues) > 0, "Should report issues when data is missing"
        
        # Should mention specific columns with missing data
        for col in ['precipitation_mm', 'order_count']:
            col_missing = df[col].isnull().sum()
            if col_missing > 0:
                col_mentioned = any(col in issue or col.replace('_', ' ') in issue.lower() for issue in issues)
                assert col_mentioned, f"Should mention {col} in issues when it has missing data"
    else:
        # Should have no missing data issues
        missing_issues = [issue for issue in issues if 'missing' in issue.lower()]
        assert len(missing_issues) == 0, "Should not report missing data issues when none exist"


def test_property_data_quality_reporting_combined_analysis():
    """
    Property 15: Data quality reporting (combined analysis)
    Combined analysis should report quality limitations from both individual datasets.
    """
    generator = InsightsGenerator()
    
    # Create datasets with different quality issues
    rainfall_df = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
        'precipitation_mm': [10.0, np.nan, 15.0, 8.0]  # One missing value
    })
    
    delivery_df = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],  # Different dates
        'order_count': [100, 150, np.nan, 200]  # One missing value, different overlap
    })
    
    # Generate individual summaries
    rainfall_summary = generator.generate_summary_statistics(rainfall_df, "Rainfall Data")
    delivery_summary = generator.generate_summary_statistics(delivery_df, "Delivery Data")
    
    # Create combined dataset (only overlapping dates)
    combined_df = pd.DataFrame({
        'date': [date(2023, 1, 1), date(2023, 1, 3), date(2023, 1, 4)],
        'precipitation_mm': [10.0, 15.0, 8.0],
        'order_count': [100, 150, np.nan]  # Still has missing value
    })
    
    # Generate combined insights
    combined_insights = generator.generate_combined_insights(combined_df, rainfall_summary, delivery_summary)
    
    # Property: Combined analysis should report quality limitations
    quality_notes = combined_insights.get('data_quality_notes', [])
    assert len(quality_notes) > 0, "Combined analysis should include data quality notes"
    
    # Property: Should mention limitations from individual datasets
    rainfall_limitations = [note for note in quality_notes if 'rainfall' in note.lower()]
    delivery_limitations = [note for note in quality_notes if 'delivery' in note.lower()]
    
    # Should mention issues from at least one dataset
    assert len(rainfall_limitations) > 0 or len(delivery_limitations) > 0, \
        "Should mention limitations from individual datasets"
    
    # Property: Should mention overlap limitations
    overlap_notes = [note for note in quality_notes if 'overlap' in note.lower() or 'matching' in note.lower()]
    
    # Property: Should include general analysis limitations
    analysis_limitations = [note for note in quality_notes if 'descriptive' in note.lower() or 'causal' in note.lower()]
    assert len(analysis_limitations) > 0, "Should mention that analysis is descriptive only"
    
    # Property: Quality notes should be user-friendly
    for note in quality_notes:
        assert isinstance(note, str), "Quality notes should be strings"
        assert len(note) > 10, "Quality notes should be substantive"
        
        # Should not contain technical jargon
        technical_terms = ['null', 'nan', 'dataframe', 'index', 'dtype']
        note_lower = note.lower()
        has_technical = any(term in note_lower for term in technical_terms)
        assert not has_technical, f"Quality note should not contain technical jargon: {note}"


def test_property_data_quality_reporting_user_friendly_messages():
    """
    Property 15: Data quality reporting (user-friendly messages)
    Data quality messages should be understandable to non-technical users.
    """
    generator = InsightsGenerator()
    
    # Test various data quality scenarios
    test_cases = [
        # Perfect data
        {
            'name': 'Perfect Dataset',
            'data': pd.DataFrame({
                'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                'precipitation_mm': [10.0, 15.0, 8.0],
                'order_count': [100, 150, 80]
            })
        },
        # Partially missing data
        {
            'name': 'Partial Dataset',
            'data': pd.DataFrame({
                'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                'precipitation_mm': [10.0, np.nan, 8.0],
                'order_count': [100, 150, np.nan]
            })
        },
        # Mostly missing data
        {
            'name': 'Sparse Dataset',
            'data': pd.DataFrame({
                'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                'precipitation_mm': [10.0, np.nan, np.nan],
                'order_count': [np.nan, np.nan, 80]
            })
        }
    ]
    
    for test_case in test_cases:
        summary = generator.generate_summary_statistics(test_case['data'], test_case['name'])
        data_quality = summary['data_quality']
        message = data_quality['message']
        
        # Property: Message should be user-friendly
        assert isinstance(message, str), f"Message should be string for {test_case['name']}"
        assert len(message) > 0, f"Message should not be empty for {test_case['name']}"
        
        # Should use plain language
        plain_language_terms = [
            'data', 'records', 'information', 'complete', 'missing', 'quality',
            'good', 'excellent', 'issues', 'available'
        ]
        
        message_lower = message.lower()
        has_plain_language = any(term in message_lower for term in plain_language_terms)
        assert has_plain_language, f"Message should use plain language: {message}"
        
        # Should not use technical jargon
        technical_jargon = [
            'dataframe', 'null', 'nan', 'dtype', 'index', 'iloc', 'loc',
            'boolean', 'integer', 'float', 'object', 'categorical'
        ]
        
        has_jargon = any(jargon in message_lower for jargon in technical_jargon)
        assert not has_jargon, f"Message should not contain technical jargon: {message}"
        
        # Should be appropriately descriptive
        word_count = len(message.split())
        assert 5 <= word_count <= 50, f"Message should be appropriately sized: {word_count} words in '{message}'"
        
        # Should end with proper punctuation
        assert message.rstrip().endswith('.'), f"Message should end with period: {message}"