"""Shared fixtures. All geometry is invented (see wind_grid_interconnect.synthetic)."""
import pytest

from wind_grid_interconnect.network import HybridProblem
from wind_grid_interconnect.roadgraph import build_road_graph
from wind_grid_interconnect.synthetic import make_scenario
from wind_grid_interconnect.water import build_water_barrier


@pytest.fixture(scope="session")
def scenario():
    return make_scenario(seed=0)


@pytest.fixture(scope="session")
def barrier(scenario):
    return build_water_barrier(scenario.river_lines, scenario.lakes)


@pytest.fixture(scope="session")
def graph(scenario):
    return build_road_graph(scenario.roads)


@pytest.fixture(scope="session")
def problem(scenario, graph, barrier):
    return HybridProblem(scenario.points, graph, barrier)
