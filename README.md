## Track1 Quick Start Guides

#### overview
The starter-kit includes training example, submission example, setup environment guides.

In training example，there are:
- `train_single_scenario.py`: RLlib PPO example to train on one map
- `train_multi_scenario.py`: RLlib PPO example to train on multi maps
- `keeplane_example.py`: simple keep lane agent example
- `randompolicy_example.py`: simple random policy agent example
- `run_evaluation.py`: evaluate agent performance with saved model
- `utils/continuous_space.py`: self defined space and adapters, using continuous action space
- `utils/discrete_space.py`: self defined space and adapters, using discrete action space
- `utils/saved_model.py`: module example to load trained model and do infer
- `utils/callback.py`: callbacks at the end of an training episode in RLlib

In submission example, there are:
- `agent.py`: agent interface which will be called by codalab scoring program.
- `run.py`: a evaluation example act like what codalab does(not necessary to be included into submission). 
- `checkpoint_2 `: trained RLlib checkpoint file, can be other saved format, see more in `utils/saved_model.py`
- `utils/continuous_space.py`: self defined space and adapters, using continuous action space
- `utils/discrete_space.py`: self defined space and adapters, using discrete action space
- `utils/saved_model.py`: module example to load trained model and do infer

!!!**Important** Note that in competition, only four ActionSpaceType are allowed, they are ``ActionSpaceType.Continuous``, 
``ActionSpaceType.ActuatorDynamic``, ``ActionSpaceType.Lane``  and ``ActionSpaceType.LaneWithContinuousSpeed``. To know  
about their differences, please refer to documents.

For build smarts environment, there are:
- `setup.md`: build guides.
- `Dockerfile`: build smarts environment using docker.
- `install_deps.sh`: build smarts environment from scratch.

zip your policy (and any associated files) and upload to CodaLab under “Participate > Submit/View Files”.

#### command example

training example:
```bash
python train_single_scenario.py --headless --scenario xxx --num_workers 20 --horizon 1000
python train_multi_scenario.py --headless --num_workers 20 --horizon 1000
python keeplane_example.py --scenario xxx
python randompolicy_example.py --scenario xxx
python run_evaluation.py --scenario xxx --load_path checkpoint_xxx/checkpoint-xxx
```

**Note**: 

for training, we recommend to set `headless` to be `True` to speed training and avoid memory error. Also, set the `num_worker` to be larger will
also accelerate sample speed.

#### submission example
```bash
python run.py --scenario xxx
```

#### submission
When you submit your solution we will put it through an automated evaluation similar to your local `run.py script`. However we’ll be evaluating it across a different set of scenarios with different maps and varying numbers of social vehicles. We also run with different seed, max step count, and episode count.

When you’re happy with your solution and ready to submit to CodaLab for evaluation, you zip your policy (and any associated files) and upload to CodaLab under “Participate > Submit/View Files”. Be careful to make sure your solutions run locally, and perform well before submitting as the upload limit is fixed.

Your example submission zip dir structure can be like this:

```python
 submission_dir
    - agent.py # defines agent so that codalab evaluation will import like from agent import agent
    - model/ # stores training model so that policy class in agent.py will restore from it.
```


####  quick_start
```bash
# install environment dependencies according to setup.md

# build scenarios
scl scenario build-all ../dataset_public

# open one tab to run envision, or using scl envision start -s dataset_public
supervisord

# run example
python keeplane_example.py --scenario xxx

# open localhost:8081 to see render

# train example
python train_single_scenario.py --headless --scenario xxx --num_workers 20 --horizon 1000

# open localhost:8082 to see SMARTS docs to get to know more details
scl docs
```

