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

sv_num = 17 
seed = 865

# generate agent missions
gen_missions(
    scenario_path,
    [
        t.LapMission(
            t.Route(
                begin=("gneE92", 0, 10),
                via=("edge-south-SE", "gneE82", "gneE93"),
                end=("gneE92", (0,), 10),
            ),
            num_laps=1,
        ),
    ],
    overwrite=True,
)


print(f"generate lap mission finished")

# generate social agent routes

impatient_car = TrafficActor(
    name="patient_car",
    speed=Distribution(sigma=0.2, mean=0.5),
    lane_changing_model=LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=JunctionModel(
        drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=1.0
    ),
)

patient_car = TrafficActor(
    name="impatient_car",
    speed=Distribution(sigma=0.2, mean=0.5),
    lane_changing_model=LaneChangingModel(impatience=0, cooperative=0.5),
    junction_model=JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
)

car_type_patient = patient_car
car_type_patient_ratio = 0.5
car_name_patient = "patient_car"

car_type_impatient = impatient_car
car_type_impatient_ratio = 0.5
car_name_impatient = "impatient_car"



traffic = Traffic(
    flows = [
        Flow(
            route=RandomRoute(),
            begin=0,
            end=1
                * 60
                * 60,  # make sure end time is larger than the time of one episode
            rate=30,
            actors={car_type_patient: car_type_patient_ratio, car_type_impatient: car_type_impatient_ratio},
        )
        for i in range(sv_num)
    ]
)
print(f"generate flow with {sv_num} social {car_name_patient, car_name_impatient} vehicles in {scenario_path} ")


gen_traffic(
    scenario_path,
    traffic,
    name=f"{sv_num * 2}_{car_name_patient}_{car_type_patient_ratio}_{car_name_impatient}_{car_type_impatient_ratio}",
    output_dir=scenario_path,
    seed=seed,
    overwrite=True,
)

