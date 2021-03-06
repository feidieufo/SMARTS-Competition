from pathlib import Path

from smarts.sstudio import gen_scenario
import smarts.sstudio.types as t


missions = [
    t.Mission(t.Route(begin=("edge-south-SN", 1, 40), end=("edge-west-EW", 0, 60))),
]

impatient_car = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.2, mean=1.0),
    lane_changing_model=t.LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=t.JunctionModel(
        drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=1.0
    ),
)

patient_car = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.2, mean=0.8),
    lane_changing_model=t.LaneChangingModel(impatience=0, cooperative=0.5),
    junction_model=t.JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
)

vertical_routes = [("north-NS", "south-NS"), ("south-SN", "north-SN")]

horizontal_routes = [("west-WE", "east-WE"), ("east-EW", "west-EW")]


traffic = {
    name: t.Traffic(
        flows=[
            t.Flow(
                route=t.Route(
                    begin=(f"edge-{r[0]}", 0, "random"),
                    end=(f"edge-{r[1]}", 0, "random"),
                ),
                rate=1,
                actors={impatient_car: 0.5, patient_car: 0.5},
            )
            for r in routes
        ]
    )
    for (name, routes) in {
        "horizontal": horizontal_routes,

    }.items()
}

gen_scenario(
    t.Scenario(ego_missions=missions, traffic=traffic,),
    output_dir=Path(__file__).parent,
)