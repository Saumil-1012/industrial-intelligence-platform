"""
Cascade Delay Detection - NetworkX Graph Engine

Models an airport network as a directed graph:
  Nodes = flights
  Edges = passenger/aircraft connections (same tail number, tight turnaround)

Algorithm:
  1. Build directed flight graph for a given day
  2. When flight A is delayed X minutes:
     - Find all downstream flights connected via same aircraft or tight connection
     - Propagate delay = max(0, X - buffer_minutes)
     - Repeat recursively until delay dissipates or no more connections
  3. Return list of affected flights with estimated new departure times
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Minimum connection buffer - delays shorter than this don't propagate
DEFAULT_BUFFER_MINUTES = 30
# Max propagation depth - prevents infinite loops in cyclic schedules
MAX_DEPTH = 6


@dataclass
class FlightNode:
    flight_id:      str
    tail_number:    str
    origin:         str
    dest:           str
    scheduled_dep:  float          # decimal hour e.g. 14.5 = 14:30
    scheduled_arr:  float
    carrier:        str
    flight_date:    str
    delay_minutes:  float = 0.0
    is_affected:    bool  = False
    cascade_depth:  int   = 0


@dataclass
class CascadeResult:
    source_flight_id:   str
    source_delay_min:   float
    affected_flights:   list[dict] = field(default_factory=list)
    total_affected:     int = 0
    max_propagated_delay: float = 0.0
    cascade_depth_reached: int = 0


class CascadeDetector:
    """
    NetworkX-based cascade delay propagation engine.
    """

    def __init__(self, buffer_minutes: int = DEFAULT_BUFFER_MINUTES):
        self.buffer_minutes = buffer_minutes
        self.graph: nx.DiGraph = nx.DiGraph()
        self._flight_data: dict[str, FlightNode] = {}

    def build_graph(self, schedule_df: pd.DataFrame) -> None:
        """
        Build flight connection graph from a schedule DataFrame.

        Required columns: flight_id, tail_number, origin, dest,
                          scheduled_dep (decimal hour), scheduled_arr (decimal hour),
                          carrier, flight_date
        """
        self.graph.clear()
        self._flight_data.clear()

        # Add all flights as nodes
        for _, row in schedule_df.iterrows():
            fid = str(row["flight_id"])
            node = FlightNode(
                flight_id=fid,
                tail_number=str(row.get("tail_number", "UNKNOWN")),
                origin=str(row.get("origin", "")),
                dest=str(row.get("dest", "")),
                scheduled_dep=float(row.get("scheduled_dep", 0)),
                scheduled_arr=float(row.get("scheduled_arr", 0)),
                carrier=str(row.get("carrier", "")),
                flight_date=str(row.get("flight_date", "")),
            )
            self.graph.add_node(fid, **node.__dict__)
            self._flight_data[fid] = node

        # Add edges: same tail number + tight turnaround
        tail_groups = schedule_df.groupby("tail_number")
        edges_added = 0

        for tail, group in tail_groups:
            group = group.sort_values("scheduled_dep")
            flights = group["flight_id"].astype(str).tolist()
            deps    = group["scheduled_dep"].tolist()
            arrs    = group["scheduled_arr"].tolist()

            for i in range(len(flights) - 1):
                f_curr  = flights[i]
                f_next  = flights[i + 1]
                arr_h   = arrs[i]
                dep_h   = deps[i + 1]

                # Turnaround in minutes
                turnaround = (dep_h - arr_h) * 60

                # Only connect if tight (< 3 hours — longer gaps absorb delays)
                if 0 <= turnaround <= 180:
                    self.graph.add_edge(
                        f_curr, f_next,
                        turnaround_min=turnaround,
                        connection_type="aircraft_rotation",
                    )
                    edges_added += 1

        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} flights, "
            f"{edges_added} connections"
        )

    def propagate_delay(
        self,
        source_flight_id: str,
        delay_minutes: float,
    ) -> CascadeResult:
        """
        Propagate a delay from source_flight_id through the network.
        Returns CascadeResult with all affected downstream flights.
        """
        result = CascadeResult(
            source_flight_id=source_flight_id,
            source_delay_min=delay_minutes,
        )

        if source_flight_id not in self.graph:
            logger.warning(f"Flight {source_flight_id} not in graph")
            return result

        # BFS propagation
        queue = [(source_flight_id, delay_minutes, 0)]
        visited = {source_flight_id}

        while queue:
            current_id, current_delay, depth = queue.pop(0)

            if depth > MAX_DEPTH or current_delay <= 0:
                continue

            for neighbor_id in self.graph.successors(current_id):
                if neighbor_id in visited:
                    continue

                edge_data    = self.graph[current_id][neighbor_id]
                turnaround   = edge_data.get("turnaround_min", self.buffer_minutes)
                prop_delay   = max(0.0, current_delay - turnaround)

                if prop_delay <= 0:
                    continue

                visited.add(neighbor_id)

                node     = self._flight_data.get(neighbor_id)
                orig_dep = node.scheduled_dep if node else 0.0
                new_dep  = orig_dep + (prop_delay / 60.0)

                affected_entry = {
                    "flight_id":            neighbor_id,
                    "tail_number":          node.tail_number if node else "N/A",
                    "origin":               node.origin if node else "N/A",
                    "dest":                 node.dest if node else "N/A",
                    "original_dep_hour":    round(orig_dep, 2),
                    "estimated_new_dep_hour": round(new_dep, 2),
                    "propagated_delay_min": round(prop_delay, 1),
                    "cascade_depth":        depth + 1,
                    "connection_type":      edge_data.get("connection_type", "aircraft_rotation"),
                }
                result.affected_flights.append(affected_entry)
                result.max_propagated_delay = max(result.max_propagated_delay, prop_delay)
                result.cascade_depth_reached = max(result.cascade_depth_reached, depth + 1)

                queue.append((neighbor_id, prop_delay, depth + 1))

        result.total_affected = len(result.affected_flights)
        logger.info(
            f"Cascade from {source_flight_id} ({delay_minutes:.0f}min): "
            f"{result.total_affected} flights affected, "
            f"max propagated delay: {result.max_propagated_delay:.0f}min"
        )
        return result

    def get_graph_summary(self) -> dict:
        return {
            "total_flights":     self.graph.number_of_nodes(),
            "total_connections": self.graph.number_of_edges(),
            "avg_degree":        round(
                sum(d for _, d in self.graph.out_degree()) / max(1, self.graph.number_of_nodes()), 2
            ),
            "buffer_minutes":    self.buffer_minutes,
        }


def build_demo_schedule() -> pd.DataFrame:
    """
    Generate a realistic demo schedule for testing cascade logic.
    3 aircraft, multiple legs each, creating cascade opportunities.
    """
    flights = [
        # Tail N001 — tight rotation chain
        {"flight_id": "AA101", "tail_number": "N001", "carrier": "AA",
         "origin": "JFK", "dest": "LAX", "scheduled_dep": 7.0, "scheduled_arr": 10.5,
         "flight_date": "2024-01-15"},
        {"flight_id": "AA102", "tail_number": "N001", "carrier": "AA",
         "origin": "LAX", "dest": "SFO", "scheduled_dep": 11.5, "scheduled_arr": 12.5,
         "flight_date": "2024-01-15"},
        {"flight_id": "AA103", "tail_number": "N001", "carrier": "AA",
         "origin": "SFO", "dest": "ORD", "scheduled_dep": 13.5, "scheduled_arr": 17.0,
         "flight_date": "2024-01-15"},
        {"flight_id": "AA104", "tail_number": "N001", "carrier": "AA",
         "origin": "ORD", "dest": "JFK", "scheduled_dep": 18.0, "scheduled_arr": 21.5,
         "flight_date": "2024-01-15"},

        # Tail N002 — separate chain
        {"flight_id": "UA201", "tail_number": "N002", "carrier": "UA",
         "origin": "DEN", "dest": "ORD", "scheduled_dep": 8.0, "scheduled_arr": 11.0,
         "flight_date": "2024-01-15"},
        {"flight_id": "UA202", "tail_number": "N002", "carrier": "UA",
         "origin": "ORD", "dest": "BOS", "scheduled_dep": 12.0, "scheduled_arr": 15.5,
         "flight_date": "2024-01-15"},

        # Tail N003 — isolated
        {"flight_id": "DL301", "tail_number": "N003", "carrier": "DL",
         "origin": "ATL", "dest": "MIA", "scheduled_dep": 9.0, "scheduled_arr": 10.5,
         "flight_date": "2024-01-15"},
    ]
    return pd.DataFrame(flights)
