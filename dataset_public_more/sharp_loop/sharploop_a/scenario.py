from pathlib import Path

from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio import types as t
from smarts.sstudio.types import RandomRoute
from smarts.sstudio.types import (
    Traffic,
    Flow,
    TrafficActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
)

scenario_path = str(Path(__file__).parent)

sv_num = 30
seed = 42

# generate agent missions
gen_missions(
    scenario_path,
    [
        t.LapMission(
            t.Route(
                begin=("edge1", 0, 50), via=("edge0",), end=("edge1", (0, 1, 2), 50),
            ),
            num_laps=1,
        ),
    ],
    overwrite=True,
)


print(f"generate lap mission finished")

# generate social agent routes

impatient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=0.5),
    lane_changing_model=LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=JunctionModel(
        drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=1.0
    ),
)

patient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=0.5),
    lane_changing_model=LaneChangingModel(impatience=0, cooperative=0.5),
    junction_model=JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
)

car_type = patient_car
car_name = "patient_car"

traffic = Traffic(
    flows=[
        Flow(
            route=RandomRoute(),
            begin=0,
            end=1
            * 60
            * 60,  # make sure end time is larger than the time of one episode
            rate=30,
            actors={car_type: 1},
        )
        for i in range(sv_num)
    ]
)

print(f"generate flow with {sv_num} social {car_name} vehicles in {scenario_path} ")

gen_traffic(
    scenario_path,
    traffic,
    name=f"{sv_num}_{car_name}",
    output_dir=scenario_path,
    seed=seed,
    overwrite=True,
)
