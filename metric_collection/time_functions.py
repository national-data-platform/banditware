"""
Time related functions for metric collection and processing.
utc_datetime_ify: Given some representation of a time, convert it to a datetime in UTC
str_to_utc_datetime: Given a string representation of a time, convert it to a datetime in UTC
delta_to_time_str: Given a datetime.timedelta, return a PromQL time string
time_str_to_delta: Given a PromQL time string, convert it to a datetime.timedelta
"""

from typing import Union, List
from datetime import datetime, timedelta, timezone
import re
import pandas as pd
from dateutil import parser


def utc_datetime_ify(
    time: Union[str, int, float, pd.Timestamp, datetime],
    assume_naive_is_local: bool = False,
) -> datetime:
    """
    Convert a time representation into a timezone aware datetime object in UTC time.
    Parameters:
        time: The time to convert. Can be one of the following types:
            * str: An ISO 8601 time string (e.g. '2025-09-25T10:30:00Z' or '2025-09-25T10:30:00-04:00') or some other common formats
            * int or float: Seconds since the epoch (01/01/1970)
            * pd.Timestamp: A pandas Timestamp object
            * datetime: A datetime object (must be timezone-aware)
        assume_naive_is_local: If True, assume naive datetimes and strings are in local time and converts them to UTC. If False, raises an error if a naive datetime or string is given.
    Raises:
        ValueError: If the time string is not in a recognized format or if a datetime object
                    is naive (no timezone info).
    """
    # handle if time is already of type pandas datetime or actual datetime
    if isinstance(time, pd.Timestamp):
        # Do not return `time`. Instead, change it to a datetime object.
        # Later, check if that datetime is naive or not and handle accordingly.
        time = time.to_pydatetime(warn=False)
    if isinstance(time, datetime):
        if datetime.tzinfo is not None:
            # non-naive datetime, convert to UTC
            return time.astimezone(timezone.utc)
        # naive datetime, decide whether to assume local time and convert to UTC
        if assume_naive_is_local:
            return time.astimezone(timezone.utc)
        raise ValueError(
            "Datetime object is naive (no timezone info). Please provide a timezone-aware datetime."
        )
    # handle if time is a float (seconds since the epoch: 01/01/1970)
    if isinstance(time, (float, int)):
        return datetime.fromtimestamp(time, tz=timezone.utc)
    type_error_msg = f"time was type {type(time)}, not one of the expected formats:\
          str | int | float | pd.Timestamp | datetime "
    assert isinstance(time, str), type_error_msg
    # get time as datetime object. Time format should be one of three patterns.
    if not assume_naive_is_local:
        return str_to_utc_datetime(time, assume_naive_is_local=False)

    # Assuming naive strings are local time, convert to utc time
    expected_format_strings = [
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y, %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]
    time_dt = _try_strptime(time, expected_format_strings)
    if time_dt is not None:
        return time_dt.astimezone(timezone.utc)
    return str_to_utc_datetime(time, assume_naive_is_local=True)


def str_to_utc_datetime(timestr: str, assume_naive_is_local: bool = False) -> datetime:
    """
    Given an ISO time string, convert it to a timezone-aware datetime object in UTC.
    Parameters:
        timestr: The time string to convert.
            - Must be in ISO 8601 format and include a timezone (e.g. 'Z' for UTC or an offset like '-04:00').
            - Ex: '2025-09-25T10:30:00Z' or '2025-09-25T10:30:00-04:00'
        assume_naive_is_local: If True, assume naive time strings (no timezone info) are in local time and convert them to UTC. If False, raises an error if a naive time string is given.
    Returns:
        utc_datetime: A timezone-aware datetime object in UTC time.
    Raises:
        ValueError: If the time string is naive (no timezone/offset).
    """
    # dt is timezone-aware if Z/offset present, naive otherwise
    dt = parser.isoparse(timestr)
    if dt.tzinfo is None and not assume_naive_is_local:
        raise ValueError(
            "Timestamp is naive (no timezone/offset). Please use UTC time (end timestring with 'Z') or an offset (e.g. '2025-09-25T10:30:00-04:00'). Or set assume_naive_is_local=True to assume local timezone."
        )
    return dt.astimezone(timezone.utc)


def delta_to_time_str(delta: timedelta) -> str:
    """
    Given a timedelta, return it as a time string for use with querying
    Parameters:
        delta: the datetime.timedelta to convert to a time string
    Returns:
        time_str: a time string in the form "_d_h_m_s"
            - Ex: 2d15h20m3s
    """
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = ""
    if days > 0:
        time_str += f"{days}d"
    if hours > 0:
        time_str += f"{hours}h"
    if minutes > 0:
        time_str += f"{minutes}m"
    if seconds > 0 or time_str == "":  # always include seconds if no other units
        time_str += f"{seconds}s"
    return time_str


def time_str_to_delta(time_str: str) -> timedelta:
    """
    Given a string in the form 5w3d6h30m5s, save the times to a dict accesible
    by the unit as their key. The int times can be any length (500m160s is allowed).
    Works given as many or few of the time units.
        - e.g. 12h also works and sets everything but h to None
    """
    # regex pattern: groups by optional int+unit but only keeps the int
    pattern = r"(?:(\d+)w)?(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
    feedback = re.search(pattern, time_str)
    if feedback is None:
        raise ValueError(f"Invalid time_str: {time_str}")
    # save time variables (if not in time_str they will be set to None)
    w, d, h, m, s = feedback.groups()
    # put time variables into a dictionary
    time_dict = {"weeks": w, "days": d, "hours": h, "minutes": m, "seconds": s}

    # get rid of null values in time_dict
    time_dict = {
        unit: float(value) for unit, value in time_dict.items() if value is not None
    }
    # create new datetime timedelta to represent the time
    # and pass in parameters as values from time_dict
    time_delta = timedelta(**time_dict)

    return time_delta


def _try_strptime(time: str, format_strings: List[str]) -> Union[datetime, None]:
    """returns datetime of time if it matches one of the format strings, otherwise none"""
    for format_str in format_strings:
        try:
            time_dt = datetime.strptime(time, format_str)
            return time_dt
        except ValueError:
            continue
    return None
