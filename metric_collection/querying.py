"""
This module defines a function for querying performance metrics for BanditWare on NDP runs
`query_performance_metrics()`: query for performance metrics of application runs on NDP's JupyterHub
"""

from typing import List, Dict, Tuple, Union
from datetime import datetime, timezone
import pandas as pd
import requests
from tqdm import tqdm
from hardware_manager import HardwareManager
from metric_collection.time_functions import utc_datetime_ify, delta_to_time_str

QUERY_TIMEOUT_SEC = 60
# allows for pd.dataframe.progress_apply() which shows a progress bar for dataframe.apply operations
tqdm.pandas()


def query_performance_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    ndp_username: str,
    progress_bar: bool = True,
    start_row: int = 0,
    end_row: Union[int, None] = None,
) -> None:
    """
    Given a dataframe with a 'start' and 'end' column and a list of metrics, query for each metric
    over the duration of the application run and add the results as new columns to the dataframe.
    Parameters:
        df: The dataframe to add queries to. Modifies the dataframe in place.
            * Must contain "start" and "end" columns
        metrics: list of metric strings to query
        ndp_username: The username of the NDP user who ran the application that's being queried
        progress_bar: Whether to show a progress bar for each metric being queried. Default is True
        start_row: The row index to start querying from (inclusive).
            * Default is 0 (beginning of the dataframe))
        end_row: The row index to end querying at (exclusive).
            * Default is None (include the rest of the dataframe, including the last row)
    Side Effects:
        Modifies the dataframe in place, adding a new column for each metric in `metrics`
            * New columns will be named the same as the metric strings
            * Values of -1 will be given to queries that didn't succeed or had 0 runtime
    """
    assert "start" in df.columns, "'start' column must be in `df`"
    assert "end" in df.columns, "'end' column must be in `df`"

    all_metrics = _get_performance_metrics()
    for metric_col in metrics:
        # Make sure metric col is valid
        err_msg = f"Metric {metric_col} not in accepted metrics: {all_metrics}"
        assert metric_col in all_metrics, err_msg
        # Make sure dataframe has all specified metric columns
        if metric_col not in df.columns:
            df[metric_col] = None

    # Query only the requested rows
    query_df = df.iloc[start_row:end_row].copy()
    query_func = query_df.progress_apply if progress_bar else query_df.apply
    for metric_col in metrics:
        if progress_bar:
            print(f"Querying {metric_col} column...")
        # query the metric column, putting it into the temp df
        query_df[metric_col] = query_func(
            lambda row, m=metric_col, username=ndp_username: _query_performance_metric(
                row, m, username
            ),
            axis=1,
        )
    # Put the queried column section back into the original dataframe
    for metric_col in metrics:
        df.loc[query_df.index, metric_col] = query_df[metric_col]


def _query_performance_metric(row: pd.Series, metric: str, ndp_username: str) -> float:
    """
    Given a row of a dataframe with 'start' and 'end' columns, query the nrp nautilus endpoint for the ndp user's given metric over that period of time.
    If the runtime is 0 seconds or the query doesn't succeed, return -1
    Parameters:
        row: A row of a dataframe with 'start' and 'end' columns
        metric: The metric to query
        ndp_username: The username of the NDP user who ran the application that's being queried
    Returns:
        value: The value of the metric between 'start' and 'end' for that user
    Raises:
        AssertionError: if row['start'] or row['end'] are invalid
        KeyError: if the metric is not one of the valid metrics
    """
    start = row["start"]
    end = row["end"]
    assert pd.notna(start), "'start' must not be NaN"
    assert pd.notna(end), "'end' must not be NaN"
    start_dt = utc_datetime_ify(start)
    end_dt = utc_datetime_ify(end)
    assert start_dt <= end_dt, "'start' must be less than or equal to 'end'"
    duration = end_dt - start_dt
    # Check if metric already queried
    if pd.notna(row[metric]):
        return row[metric]

    # Handle 0 or invalid runtime - cannot query
    if duration.total_seconds() <= 0:
        return -1

    hardware_spec = HardwareManager.spec_from_hardware_idx(row["hardware"])

    query = _get_metric_query(
        metric,
        start=start_dt,
        end=end_dt,
        ndp_username=ndp_username,
        hardware_spec=hardware_spec,
    )

    # TODO: Handle if query_data() throws an error
    result = _query_data(query)
    if len(result) == 0 or len(result[0]["value"]) < 2:
        return -1
    value = float(result[0]["value"][1])
    return value


def _get_performance_metrics() -> List[str]:
    """
    Returns a list of all performance metric names that can be queried
    """
    metrics = [
        "cpu_usage_%",
        "max_cpu_count",
        "mem_usage_%",
        "gpu_usage_%",
        "max_gpu_count",
    ]
    return metrics


def _get_metric_query(
    metric: str,
    start: datetime,
    end: datetime,
    ndp_username: str,
    hardware_spec: Tuple[int, int],
) -> str:
    """
    Given a metric name and start, end times, return the corresponding PROMQL query string.
    Note: start and end times must be timezone-aware datetime objects in UTC time
    Parameters:
        metric: The metric to query. Must be one of the following:
            * "cpu_usage_%"
            * "max_cpu_count"
            * "mem_usage_%"
            * "gpu_usage_%"
            * "max_gpu_count"
        start: The start time of the application run (timezone-aware datetime in UTC)
        end: The end time of the application run (timezone-aware datetime in UTC)
        ndp_username: The username of the NDP user who ran the application that's being queried
        hardware_spec: the tuple of (CPU count, gigabytes of RAM) that the application was run on
    Returns:
        query: The PROMQL query string for the given metric and time range
    Raises:
        KeyError: if the metric is not one of the valid metrics
    """
    metric_queries = {
        # CPU usage % = avg cpus used / total cpus = (increase in cpu_seconds / total seconds) / total cpus
        "cpu_usage_%": 'sum(increase(container_cpu_usage_seconds_total{{pod=~"{pod_re}", container="notebook"}}[{duration_str}] offset {offset})) / {duration_sec} / {hardware_cpus}',
        # Max CPUs used at once = max (rate of new cpu_seconds per second, sampled several times)
        "max_cpu_count": 'max_over_time(sum(irate(container_cpu_usage_seconds_total{{pod=~"{pod_re}", container="notebook"}}[1m]))[{duration_str}:30s] offset {offset})',
        # Mem usage % = max bytes used / given bytes = (max bytes used / (given GB * bytes per GB)
        "mem_usage_%": 'sum(max_over_time(container_memory_working_set_bytes{{pod=~"{pod_re}", container="notebook"}}[{duration_str}] offset {offset})) / (1024*1024*1024) / {hardware_gb}',
        # no max_mem_count because that would be the same as mem_usage_% * hardware_mem
        # max_cpu_count gives us new information: max cpus used at once, while max_mem_count and mem_usage_% measure the same thing
        # TODO: add gpu queries
        "gpu_usage_%": "",
        "max_gpu_count": "",
    }
    if metric not in metric_queries:
        raise KeyError(
            f"Metric {metric} not in accepted metrics: {list(metric_queries.keys())}"
        )

    duration = end - start
    duration_str = delta_to_time_str(duration)
    offset = datetime.now().astimezone(timezone.utc) - end
    offset_str = delta_to_time_str(offset)
    pod_regex_str = f"jupyter-{ndp_username}-.*"
    query = metric_queries[metric].format(
        pod_re=pod_regex_str,
        offset=offset_str,
        duration_str=duration_str,
        duration_sec=duration.total_seconds(),
        hardware_cpus=hardware_spec[0],
        hardware_gb=hardware_spec[1],
    )
    return query


def _query_data(
    query: str, timeout_sec: int = QUERY_TIMEOUT_SEC, handle_fail: bool = True
) -> List[Dict]:
    """
    Given a PROMQL query string, query the nrp nautilus endpoint and return the data
    Parameters:
        query: PROMQL query string
        timeout_sec: how long to wait (in seconds) before abandoning a query
        handle_fail: re request the api if no response from query. Default is True
            * There is a bug with the api itself where every fifth request comes back with no data.
            * This parameter set to True will re request to deal with that
            * It is highly recommended that handle_fail is always set to True.
    Returns:
        result_list: a list of dictionaries of metrics and values
    """
    # set up url
    base_url = "https://thanos.nrp-nautilus.io/api/v1/"
    query_component = f"query?query={query}"
    full_url = base_url + query_component
    # query database
    queried_data = requests.get(full_url, timeout=timeout_sec).json()

    # re-request data if it comes back with no value
    if handle_fail:
        try:
            res_list = queried_data["data"]["result"]
            if len(res_list) == 0:
                queried_data = requests.get(full_url, timeout=timeout_sec).json()
        except KeyError:
            # pylint: disable=raise-missing-from
            raise RuntimeError(
                f"\n\nBad query string:\n{query}\n\nqueried_data is:\n{queried_data}\n\n"
            )

    return queried_data["data"]["result"]
