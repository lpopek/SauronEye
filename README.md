# Sauron Eye
This is the repository for Sauron Eye project. It contains all of the code used
in training, tuning and evaluating all of the models.

# Setup
To clone this project use
```
git clone https://github.com/lpopek/SauronEye.git
cd SauronEye
pip install -r requirements.txt
```

# Project structure
The project is split into subfolders that encapsulate certain functionalities.

## eval
Eval directory contains all of the code related to evaluating a configuration (
model + tracker). To start evaluating, setup the `eval/config.json` file:

```
{
  "models": [list, of, YOLO, models, to, evaluate],
  "trackers": [list, of, trackers, to, evaluate],
  "dataset_path": "path/to/dataset"
}
```

A valid configuration file should meet these criteria:
* have all of the keywords: "models", "trackers", "dataset_path"
* models should be a list of strings containing model names that end in a `.pt` file extension
* trackers should be a list of strings containing tracker names that end in a `.yaml` file extension 
* all of the trackers listed should have `.yaml` configuration files in `eval/trackers`
* "dataset_path" should point to a directory containing subfolders with MOT Challenge scenes

The evaluation data will be written into `results/<model_name>/<dataset_name>/<tracker_name>/data`
directory and can be later evaluated by the official MOT Challenge tracking 
evaluation tool - [TrackEval](https://github.com/JonathonLuiten/TrackEval). Models
listed in the `eval/config.json` file will be downloaded into `eval/models` directory.

To run the script use
```
python3 eval/eval.py
```
from the project's root directory. You can also call the script from wherever you
want, as all of the paths used in the script are *absolute paths*, so it should 
work just fine.