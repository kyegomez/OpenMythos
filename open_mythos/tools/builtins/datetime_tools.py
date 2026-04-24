"""
DateTime Built-in Tools

Provides date and time operations.
"""

import datetime
from typing import Any, Dict, Optional

from open_mythos.tools.registry import tool_result, tool_error, register_tool


def _register_datetime_tools():
    """Register all datetime tools."""
    
    def get_current_time(format: Optional[str] = None, timezone: Optional[str] = None) -> str:
        """Get current time."""
        now = datetime.datetime.now()
        
        if timezone:
            # Simple timezone handling - just append to format
            if format is None:
                format = "%Y-%m-%d %H:%M:%S %Z"
        
        if format:
            try:
                result = now.strftime(format)
                return tool_result({"time": result, "format": format})
            except Exception as e:
                return tool_error(f"Invalid format: {e}")
        else:
            return tool_result({
                "iso": now.isoformat(),
                "timestamp": now.timestamp(),
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "weekday": now.strftime("%A"),
            })
    
    def parse_datetime(date_string: str, format: Optional[str] = None) -> str:
        """Parse a datetime string."""
        if format:
            try:
                parsed = datetime.datetime.strptime(date_string, format)
                return tool_result({"parsed": parsed.isoformat(), "timestamp": parsed.timestamp()})
            except Exception as e:
                return tool_error(f"Parse error: {e}")
        else:
            # Try ISO format first
            try:
                parsed = datetime.datetime.fromisoformat(date_string.replace("Z", "+00:00"))
                return tool_result({"parsed": parsed.isoformat(), "timestamp": parsed.timestamp()})
            except:
                return tool_error(f"Could not parse: {date_string}")
    
    def add_time(start_date: str, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> str:
        """Add time to a date."""
        try:
            # Try to parse the input
            if "T" in start_date:
                dt = datetime.datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            else:
                dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            
            delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
            result = dt + delta
            
            return tool_result({
                "result": result.isoformat(),
                "original": start_date,
                "days_added": days,
                "hours_added": hours,
            })
        except Exception as e:
            return tool_error(f"Error: {e}")
    
    def date_diff(date1: str, date2: str) -> str:
        """Calculate difference between two dates."""
        try:
            dt1 = datetime.datetime.fromisoformat(date1.replace("Z", "+00:00"))
            dt2 = datetime.datetime.fromisoformat(date2.replace("Z", "+00:00"))
            
            diff = dt1 - dt2
            total_seconds = diff.total_seconds()
            
            return tool_result({
                "difference_seconds": total_seconds,
                "difference_days": diff.days,
                "difference_hours": total_seconds / 3600,
                "difference_minutes": total_seconds / 60,
                "date1": date1,
                "date2": date2,
            })
        except Exception as e:
            return tool_error(f"Error: {e}")
    
    def format_date(date_string: str, output_format: str) -> str:
        """Format a date string to a different format."""
        try:
            # Parse input (try ISO first)
            if "T" in date_string:
                dt = datetime.datetime.fromisoformat(date_string.replace("Z", "+00:00"))
            else:
                dt = datetime.datetime.strptime(date_string, "%Y-%m-%d")
            
            result = dt.strftime(output_format)
            return tool_result({"result": result, "format": output_format})
        except Exception as e:
            return tool_error(f"Error: {e}")
    
    def is_valid_date(date_string: str, format: Optional[str] = None) -> str:
        """Check if a date string is valid."""
        if format:
            try:
                datetime.datetime.strptime(date_string, format)
                return tool_result({"valid": True, "format": format})
            except:
                return tool_result({"valid": False, "format": format})
        else:
            # Try common formats
            formats = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"]
            for fmt in formats:
                try:
                    datetime.datetime.strptime(date_string, fmt)
                    return tool_result({"valid": True, "format": fmt})
                except:
                    continue
            
            return tool_result({"valid": False})
    
    # Register tools
    register_tool(
        name="datetime_now",
        toolset="datetime",
        schema={
            "name": "datetime_now",
            "description": "Get current date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "description": "Optional strftime format"},
                    "timezone": {"type": "string", "description": "Optional timezone"}
                }
            }
        },
        handler=get_current_time,
        description="Get current time",
        emoji="[🕐]"
    )
    
    register_tool(
        name="datetime_parse",
        toolset="datetime",
        schema={
            "name": "datetime_parse",
            "description": "Parse a datetime string",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_string": {"type": "string", "description": "Date string to parse"},
                    "format": {"type": "string", "description": "Optional format string"}
                },
                "required": ["date_string"]
            }
        },
        handler=parse_datetime,
        description="Parse datetime string",
        emoji="[📅]"
    )
    
    register_tool(
        name="datetime_add",
        toolset="datetime",
        schema={
            "name": "datetime_add",
            "description": "Add time to a date",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Starting date (ISO format)"},
                    "days": {"type": "integer", "description": "Days to add", "default": 0},
                    "hours": {"type": "integer", "description": "Hours to add", "default": 0},
                    "minutes": {"type": "integer", "description": "Minutes to add", "default": 0},
                    "seconds": {"type": "integer", "description": "Seconds to add", "default": 0}
                },
                "required": ["start_date"]
            }
        },
        handler=add_time,
        description="Add time to date",
        emoji="[➕]"
    )
    
    register_tool(
        name="datetime_diff",
        toolset="datetime",
        schema={
            "name": "datetime_diff",
            "description": "Calculate difference between two dates",
            "parameters": {
                "type": "object",
                "properties": {
                    "date1": {"type": "string", "description": "First date (ISO format)"},
                    "date2": {"type": "string", "description": "Second date (ISO format)"}
                },
                "required": ["date1", "date2"]
            }
        },
        handler=date_diff,
        description="Date difference",
        emoji="[➖]"
    )
    
    register_tool(
        name="datetime_format",
        toolset="datetime",
        schema={
            "name": "datetime_format",
            "description": "Format a date to a different format",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_string": {"type": "string", "description": "Date to format"},
                    "output_format": {"type": "string", "description": "Output format (strftime)"}
                },
                "required": ["date_string", "output_format"]
            }
        },
        handler=format_date,
        description="Format date",
        emoji="[🔄]"
    )
    
    register_tool(
        name="datetime_validate",
        toolset="datetime",
        schema={
            "name": "datetime_validate",
            "description": "Check if date string is valid",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_string": {"type": "string", "description": "Date string to validate"},
                    "format": {"type": "string", "description": "Optional specific format"}
                },
                "required": ["date_string"]
            }
        },
        handler=is_valid_date,
        description="Validate date",
        emoji="[✓]"
    )


# Auto-register when imported
_register_datetime_tools()


__all__ = ["_register_datetime_tools"]
