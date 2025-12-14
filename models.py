"""
Data models for the Rainfall vs Food Delivery Demand Dashboard

This module defines the core data structures used throughout the application
for representing rainfall data, delivery data, and combined data points.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class RainfallRecord:
    """
    Represents a single day's rainfall measurement.
    
    Attributes:
        date: The date of the measurement
        precipitation_mm: Daily precipitation in millimeters (must be >= 0)
    """
    date: date
    precipitation_mm: float
    
    def validate(self) -> bool:
        """
        Validates the rainfall record data.
        
        Returns:
            bool: True if the record is valid, False otherwise
        """
        if not isinstance(self.date, date):
            return False
        if not isinstance(self.precipitation_mm, (int, float)):
            return False
        return self.precipitation_mm >= 0
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not self.validate():
            raise ValueError(f"Invalid RainfallRecord: date={self.date}, precipitation_mm={self.precipitation_mm}")


@dataclass
class DeliveryRecord:
    """
    Represents a single day's delivery order count.
    
    Attributes:
        date: The date of the delivery data
        order_count: Number of delivery orders (must be >= 0)
    """
    date: date
    order_count: int
    
    def validate(self) -> bool:
        """
        Validates the delivery record data.
        
        Returns:
            bool: True if the record is valid, False otherwise
        """
        if not isinstance(self.date, date):
            return False
        if not isinstance(self.order_count, int):
            return False
        return self.order_count >= 0
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not self.validate():
            raise ValueError(f"Invalid DeliveryRecord: date={self.date}, order_count={self.order_count}")


@dataclass
class CombinedDataPoint:
    """
    Represents a combined data point with both rainfall and delivery information for a single date.
    
    Attributes:
        date: The date for this data point
        precipitation_mm: Daily precipitation in millimeters
        order_count: Number of delivery orders
    """
    date: date
    precipitation_mm: float
    order_count: int
    
    def validate(self) -> bool:
        """
        Validates the combined data point.
        
        Returns:
            bool: True if the data point is valid, False otherwise
        """
        if not isinstance(self.date, date):
            return False
        if not isinstance(self.precipitation_mm, (int, float)):
            return False
        if not isinstance(self.order_count, int):
            return False
        return self.precipitation_mm >= 0 and self.order_count >= 0
    
    def to_dict(self) -> dict:
        """
        Converts the data point to a dictionary representation.
        
        Returns:
            dict: Dictionary with date, rainfall, and orders keys
        """
        return {
            'date': self.date,
            'rainfall': self.precipitation_mm,
            'orders': self.order_count
        }
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not self.validate():
            raise ValueError(f"Invalid CombinedDataPoint: date={self.date}, precipitation_mm={self.precipitation_mm}, order_count={self.order_count}")


def create_rainfall_record(date_value: date, precipitation: float) -> Optional[RainfallRecord]:
    """
    Factory function to create a RainfallRecord with validation.
    
    Args:
        date_value: The date for the record
        precipitation: Precipitation amount in mm
        
    Returns:
        RainfallRecord if valid, None if invalid
    """
    try:
        return RainfallRecord(date_value, precipitation)
    except ValueError:
        return None


def create_delivery_record(date_value: date, orders: int) -> Optional[DeliveryRecord]:
    """
    Factory function to create a DeliveryRecord with validation.
    
    Args:
        date_value: The date for the record
        orders: Number of delivery orders
        
    Returns:
        DeliveryRecord if valid, None if invalid
    """
    try:
        return DeliveryRecord(date_value, orders)
    except ValueError:
        return None


def create_combined_data_point(date_value: date, precipitation: float, orders: int) -> Optional[CombinedDataPoint]:
    """
    Factory function to create a CombinedDataPoint with validation.
    
    Args:
        date_value: The date for the data point
        precipitation: Precipitation amount in mm
        orders: Number of delivery orders
        
    Returns:
        CombinedDataPoint if valid, None if invalid
    """
    try:
        return CombinedDataPoint(date_value, precipitation, orders)
    except ValueError:
        return None