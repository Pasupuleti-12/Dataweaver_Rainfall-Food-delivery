"""
Tests for the DashboardController class

This module contains unit tests and property-based tests for the dashboard controller
functionality, including error handling and user feedback as specified in requirements.
"""

import pytest
import pandas as pd
from datetime import date, datetime
from hypothesis import given, strategies as st, assume
from hypothesis.extra.pandas import data_frames, column
import tempfile
import os
from pathlib import Path
import io

from dashboard_controller import DashboardController, ErrorHandler, LoadingIndicator


class TestErrorHandler:
    """Test cases for the ErrorHandler utility class."""
    
    def test_file_not_found_error(self):
        """Test user-friendly message for file not found errors."""
        error = FileNotFoundError("No such file or directory: 'missing_file.csv'")
        message = ErrorHandler.create_user_friendly_message(error, "loading data")
        
        assert "file could not be found" in message.lower()
        assert "technical" not in message.lower()
        assert "exception" not in message.lower()
    
    def test_permission_error(self):
        """Test user-friendly message for permission errors."""
        error = PermissionError("Permission denied")
        message = ErrorHandler.create_user_friendly_message(error, "accessing file")
        
        assert "unable to access" in message.lower()
        assert "permission" in message.lower()
        assert "try again" in message.lower()
    
    def test_empty_file_error(self):
        """Test user-friendly message for empty file errors."""
        error = ValueError("Empty CSV file")
        message = ErrorHandler.create_user_friendly_message(error, "processing file")
        
        assert "empty" in message.lower()
        assert "check your file" in message.lower()
    
    def test_missing_columns_error(self):
        """Test user-friendly message for missing columns errors."""
        error = ValueError("Missing required columns: ['date', 'precipitation_mm']")
        message = ErrorHandler.create_user_friendly_message(error, "validating data")
        
        assert "missing required columns" in message.lower()
        assert "correct column names" in message.lower()
    
    def test_date_parsing_error(self):
        """Test user-friendly message for date parsing errors."""
        error = ValueError("Failed to parse date: invalid format")
        message = ErrorHandler.create_user_friendly_message(error, "processing dates")
        
        assert "date" in message.lower()
        assert "format" in message.lower()
        assert "yyyy-mm-dd" in message.lower()
    
    def test_generic_error_fallback(self):
        """Test fallback message for unknown errors."""
        error = RuntimeError("Some unexpected technical error")
        message = ErrorHandler.create_user_friendly_message(error, "processing data")
        
        assert "issue occurred" in message.lower()
        assert "try again" in message.lower()
        assert "technical error" not in message.lower()
    
    def test_helpful_suggestions(self):
        """Test that helpful suggestions are appropriate for error types."""
        # File format suggestion
        suggestion = ErrorHandler.get_helpful_suggestion('file_format', 'rainfall')
        assert 'csv' in suggestion.lower()
        assert 'rainfall' in suggestion.lower()
        
        # Missing columns suggestion
        suggestion = ErrorHandler.get_helpful_suggestion('missing_columns', 'delivery')
        assert 'order_count' in suggestion.lower()
        assert 'date' in suggestion.lower()
        
        # Date format suggestion
        suggestion = ErrorHandler.get_helpful_suggestion('date_format')
        assert 'date format' in suggestion.lower()
        assert '2023' in suggestion  # Should show example


class TestDashboardController:
    """Test cases for the DashboardController class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.controller = DashboardController()
    
    def test_initialization(self):
        """Test that DashboardController initializes correctly."""
        assert self.controller.data_loader is not None
        assert self.controller.chart_generator is not None
        assert self.controller.insights_generator is not None
        assert self.controller.error_handler is not None
        assert self.controller.loading_indicator is not None
    
    def test_create_user_friendly_error_message_file_not_found(self):
        """Test user-friendly error message creation for file not found."""
        error_details = "FileNotFoundError: No such file or directory"
        message = self.controller._create_user_friendly_error_message(error_details, "rainfall")
        
        assert "rainfall file could not be found" in message.lower()
        assert "filenotfounderror" not in message.lower()
    
    def test_create_user_friendly_error_message_missing_columns(self):
        """Test user-friendly error message for missing columns."""
        error_details = "Missing required columns: ['date', 'precipitation_mm']"
        message = self.controller._create_user_friendly_error_message(error_details, "rainfall")
        
        assert "missing required columns" in message.lower()
        assert "precipitation_mm" in message.lower()
        assert "date" in message.lower()
    
    def test_create_user_friendly_error_message_date_parsing(self):
        """Test user-friendly error message for date parsing errors."""
        error_details = "Failed to parse date: 2023-13-45"
        message = self.controller._create_user_friendly_error_message(error_details, "delivery")
        
        assert "date" in message.lower()
        assert "format" in message.lower()
        assert "yyyy-mm-dd" in message.lower()


# Property-based tests for error message quality
class TestErrorMessageQuality:
    """
    Property-based tests for error message quality.
    
    **Feature: rainfall-delivery-dashboard, Property 16: User-friendly error messages**
    **Validates: Requirements 6.4**
    """
    
    @given(st.text(min_size=1, max_size=200))
    def test_error_messages_avoid_technical_jargon(self, error_text):
        """
        Property test: Error messages should not contain technical jargon.
        
        **Feature: rainfall-delivery-dashboard, Property 16: User-friendly error messages**
        **Validates: Requirements 6.4**
        """
        # Create a mock error with the given text
        error = ValueError(error_text)
        message = ErrorHandler.create_user_friendly_message(error, "testing")
        
        # Technical jargon that should not appear in user-friendly messages
        technical_terms = [
            'exception', 'traceback', 'stacktrace', 'null pointer', 'segfault',
            'errno', 'stderr', 'stdout', 'buffer overflow', 'memory leak',
            'thread', 'mutex', 'semaphore', 'deadlock', 'race condition',
            'assertion failed', 'core dump', 'segmentation fault'
        ]
        
        message_lower = message.lower()
        
        # Check that technical terms don't appear in the user message
        for term in technical_terms:
            assert term not in message_lower, f"Technical term '{term}' found in user message: {message}"
        
        # Message should be helpful and actionable
        assert len(message) > 10, "Error message should be substantial"
        assert any(word in message_lower for word in ['try', 'check', 'ensure', 'please']), \
            "Error message should provide actionable guidance"
    
    @given(st.sampled_from(['rainfall', 'delivery', 'combined']))
    def test_error_messages_are_context_specific(self, data_type):
        """
        Property test: Error messages should be specific to the data type context.
        
        **Feature: rainfall-delivery-dashboard, Property 16: User-friendly error messages**
        **Validates: Requirements 6.4**
        """
        # Test with a missing columns error
        error = ValueError(f"Missing required columns for {data_type}")
        message = ErrorHandler.create_user_friendly_message(error, f"processing {data_type}")
        
        # Message should mention the specific data type
        assert data_type in message.lower() or "file" in message.lower(), \
            f"Error message should reference the data type context: {message}"
        
        # Message should be specific and helpful
        assert len(message.split()) >= 5, "Error message should be descriptive"
        assert message.endswith('.'), "Error message should be properly formatted"
    
    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
    def test_error_messages_handle_multiple_issues(self, error_list):
        """
        Property test: Error handling should work with multiple error types.
        
        **Feature: rainfall-delivery-dashboard, Property 16: User-friendly error messages**
        **Validates: Requirements 6.4**
        """
        # Test that we can handle multiple different error messages
        messages = []
        
        for error_text in error_list:
            error = ValueError(error_text)
            message = ErrorHandler.create_user_friendly_message(error, "testing")
            messages.append(message)
        
        # All messages should be non-empty and different (unless input was identical)
        assert all(len(msg) > 0 for msg in messages), "All error messages should be non-empty"
        
        # Messages should not contain raw error text (should be transformed)
        for i, (original_error, user_message) in enumerate(zip(error_list, messages)):
            # If the original error is very generic, the user message might contain it
            # But it should be in a more user-friendly context
            if len(original_error) > 10:  # Only check for longer, more specific errors
                assert original_error.lower() not in user_message.lower() or \
                       any(word in user_message.lower() for word in ['please', 'try', 'check']), \
                       f"Error message should transform technical details: {user_message}"
    
    @given(st.sampled_from([
        'FileNotFoundError', 'PermissionError', 'ValueError', 'TypeError', 
        'IOError', 'UnicodeDecodeError', 'KeyError', 'IndexError'
    ]))
    def test_error_messages_handle_common_exception_types(self, exception_type):
        """
        Property test: Error messages should handle common Python exception types appropriately.
        
        **Feature: rainfall-delivery-dashboard, Property 16: User-friendly error messages**
        **Validates: Requirements 6.4**
        """
        # Create a mock error message that includes the exception type
        error_text = f"{exception_type}: Something went wrong"
        error = Exception(error_text)
        message = ErrorHandler.create_user_friendly_message(error, "processing data")
        
        # The user message should not contain the raw exception type name
        assert exception_type not in message, \
            f"User message should not contain exception type '{exception_type}': {message}"
        
        # Message should be user-friendly
        assert any(word in message.lower() for word in [
            'unable', 'issue', 'problem', 'error', 'try', 'check', 'please'
        ]), f"Message should use user-friendly language: {message}"
        
        # Message should end with proper punctuation
        assert message.rstrip().endswith('.'), "Error message should end with proper punctuation"
    
    def test_helpful_suggestions_are_relevant(self):
        """
        Test that helpful suggestions are relevant to the error type.
        
        **Feature: rainfall-delivery-dashboard, Property 16: User-friendly error messages**
        **Validates: Requirements 6.4**
        """
        # Test file format suggestions
        suggestion = ErrorHandler.get_helpful_suggestion('file_format', 'rainfall')
        assert 'csv' in suggestion.lower()
        assert 'tip' in suggestion.lower()
        
        # Test missing columns suggestions
        suggestion = ErrorHandler.get_helpful_suggestion('missing_columns', 'delivery')
        assert 'order_count' in suggestion.lower()
        assert 'date' in suggestion.lower()
        
        # Test date format suggestions
        suggestion = ErrorHandler.get_helpful_suggestion('date_format')
        assert any(format_example in suggestion for format_example in ['2023', 'yyyy', 'mm', 'dd'])
        
        # All suggestions should start with tip indicator
        for error_type in ['file_format', 'missing_columns', 'date_format', 'empty_data']:
            suggestion = ErrorHandler.get_helpful_suggestion(error_type, 'rainfall')
            assert suggestion.startswith('💡'), f"Suggestion should start with tip indicator: {suggestion}"
    
    @given(st.text(min_size=1, max_size=100))
    def test_error_message_length_is_reasonable(self, error_input):
        """
        Property test: Error messages should be of reasonable length.
        
        **Feature: rainfall-delivery-dashboard, Property 16: User-friendly error messages**
        **Validates: Requirements 6.4**
        """
        error = ValueError(error_input)
        message = ErrorHandler.create_user_friendly_message(error, "testing")
        
        # Message should not be too short (uninformative) or too long (overwhelming)
        assert 10 <= len(message) <= 500, \
            f"Error message length should be reasonable (10-500 chars): {len(message)} chars - '{message}'"
        
        # Message should not be just the raw error repeated
        if len(error_input.strip()) > 5:
            assert message.lower() != error_input.lower(), \
                "Error message should be transformed, not just the raw error"
    
    def test_error_messages_provide_actionable_guidance(self):
        """
        Test that error messages provide actionable guidance to users.
        
        **Feature: rainfall-delivery-dashboard, Property 16: User-friendly error messages**
        **Validates: Requirements 6.4**
        """
        # Test various error scenarios
        test_cases = [
            (FileNotFoundError("file not found"), "loading file"),
            (ValueError("missing columns"), "validating data"),
            (PermissionError("access denied"), "reading file"),
            (UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"), "decoding file")
        ]
        
        for error, context in test_cases:
            message = ErrorHandler.create_user_friendly_message(error, context)
            
            # Should contain actionable words
            actionable_words = ['try', 'check', 'ensure', 'please', 'verify', 'make sure']
            assert any(word in message.lower() for word in actionable_words), \
                f"Error message should provide actionable guidance: {message}"
            
            # Should not blame the user
            blame_words = ['you failed', 'your fault', 'you did wrong', 'you broke']
            assert not any(word in message.lower() for word in blame_words), \
                f"Error message should not blame the user: {message}"


if __name__ == "__main__":
    pytest.main([__file__])