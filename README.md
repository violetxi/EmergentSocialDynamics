# EmergentSocialDynamics
Context dependent intrinsic motivation allows diverse social dynamics to emerge from different environment.

## Installation

Install this repo on a remote cluster without full previlige, first install SWIG from source:

Download SWIG 
```
wget http://prdownloads.sourceforge.net/swig/swig-4.1.1.tar.gz
tar -xvf swig-4.1.1.tar.gz && cd swig-4.1.1/
```
Configure the makefile for SWIG. This is where you specify the prefix to a directory that you have write access to:
```
./configure --prefix=/path/to/your/directory/  --without-pcre
```
Build and install SWIG 
```
make && make install
```

Then add the `bin` directory to your ~/.bashrc
```
echo 'export PATH=/path/to/your/home/directory/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
Install repo and dependency (using *venv* or *conda env* with `python=3.8`)
```
pip install -e .
```
## Running experiments
### Training
Experiments are seperated into different folders inside `experiments/`. The suggestion is to create a folder for each model and have 
all the checkpoints and experiment configuration and parameter sweeping related stuff stored inside an `outputs/` folder. As an example, to run 
an experiment, go to one of the folders, e.g., `experiments/icm_5_agents_gru/` and run it using 
```
cd experiments/icm_5_agents_gru/
CUDA_VISIBLE_DEVICES={} python main.py
```
To run hyper-parameter search with `wandb`, go to an experiment folder and start the sweep with this command
```
wandb sweep --project {PROJECT_NAME} --name {SWEEP_NAME} config/sweep.yaml
```
Once the command started, you will get the search run command, and run many wandb agents in parallel using 
```
CUDA_VISIBLE_DEVICES={GPU_NUM} wandb agent stanford_autonomous_agent/{SWEEP_NAME}/{SWEEP_ID}
```
### Evaluation
To run evaluation with default setting, go to an experiment folder:
```
CUDA_VISIBLE_DEVICES={GPU_NUM} python main.py exp_run.eval_only=True exp_run.ckpt_dir={CKPT_DIR}
```
You can also change evaluation setting (anything in the config is modifiable) using commands similar to the following
  - change number of tested episode, steps in an episode and result directory, e.g., change number of test_eps. Changing model to what you want to save the model result as
    ```
    CUDA_VISIBLE_DEVICES=1 python main.py exp_run.eval_only=True exp_run.ckpt_dir={} exp_run.result_dir={} trainer.test_eps=5 model.name=social_influence_visible
    ```
This will add step-wise reward for each agent per-episode to new `.pkl` with other evaluation saved in the same directory, and save frames for all the episodes inside a folder called frames. The frames for each episode will be saved in the one folder.
#### Plots
To get evaluation plots, the the plotting functions are inside `plot_utils.py` 
```
python analysis.py
```
### Visualize behavior
Once the frames are created, u can copy `create_video.sh` into the `frames/` folder and create videos 
```
bash create_video.sh
```
