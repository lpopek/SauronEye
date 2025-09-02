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

For configuring the tracker `.yaml` files visit the official [YOLO documentation](https://docs.ultralytics.com/modes/track/#tracker-selection:~:text=of%20each%20parameter.-,Tracker%20Arguments,-Some%20tracking%20behaviors)

Supported MOT datasets can be found on the official [MOT Challenge website](https://motchallenge.net/)

## tune
Tune directory contains all of the code related to tuning tracker hyperparameters. It utilizes eval
scripts and TrackEval repository to find the best tracker settings based on its MOTA score.

To enable tuning:
1. Go to `eval/trackers` directory and add `test_tracker.yaml` file if it doesn't already exist
2. In `eval/config.json` file change `trackers` value to `["test_tracker.yaml"]`
3. Update the rest of the configuration file if needed
3. Run `python tune/tune.py` from the projects root directory.

The script will automatically modify `eval/trackers/test_tracker.yaml` file and evaluate it with
the scripts mentioned above. The results of each tuning pass will be written into the `tune/tune.log`
file.