import os
import csv
import copy
import pathlib
import logging
import subprocess

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
TEST_TRACKER_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../eval/trackers/test_tracker.yaml"))
RESULTS_DIR: str = os.path.join(CURRENT_DIR, "../eval/results/yolo11n/MOT15-train/test_tracker")
AVAILABLE_SETUP_OPTIONS: dict = {
    "tracker_type": ["bytetrack", "botsort"],
    "track_high_thresh": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "track_low_thresh": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "new_track_thresh": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
    "track_buffer": [10, 20, 30, 40, 50, 60],
    "match_thresh": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "fuse_score": [False, True],
    "gmc_method": ["orb", "sift", "ecc", "sparseOptFlow", "None"],
    "proximity_thresh": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "appearance_thresh": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
    "with_reid": [False, True]
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(
            CURRENT_DIR, "tune.log"
        )),
        logging.StreamHandler()
    ]
)
LOGGER: logging.Logger = logging.getLogger(__name__)

def getDefaultSetup() -> dict:
    LOGGER.info("Creating default tracker setup")
    return {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.4,
        "track_low_thresh": 0.2,
        "new_track_thresh": 0.6,
        "track_buffer": 30,
        "match_thresh": 0.6,
        "fuse_score": True,
        "gmc_method": "orb",
        "proximity_thresh": 0.2,
        "appearance_thresh": 0.4,
        "with_reid": True
    }

def writeYaml(trackerConfig: dict) -> None:
    LOGGER.info("Overwriting tracker .yaml file")
    with open(TEST_TRACKER_DIR, "w+") as trackerFile:
        for key, value in trackerConfig.items():
            trackerFile.write(f"{key}: {value}\n")

def runEvalSctips() -> None:
    LOGGER.info("Running eval script")

    evalScriptPath: str = os.path.abspath(os.path.join(CURRENT_DIR, "../eval/eval.py"))
    cmd = ["/workspace/miniconda/bin/conda", "run", "-n", "YOLO", "python", evalScriptPath]
    subprocess.run(cmd, capture_output=False)

    LOGGER.info("Running TrackEval")

    trackEvalScriptPath: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../TrackEval/scripts/run_mot_challenge.py"))
    cmd = [
        "/workspace/miniconda/bin/conda", "run", "-n", "py37env", 
        "python", trackEvalScriptPath,
        "--GT_FOLDER", "/workspace/evaluation/MOTA/MOT15", 
        "--TRACKERS_FOLDER", "/workspace/SauronEye/eval/results/yolo11n",
        "--BENCHMARK", "MOT15", 
        "--TRACKERS_TO_EVAL", "test_tracker"
    ]
    subprocess.run(cmd, capture_output=False)

def getMotaResults() -> float:
    LOGGER.info("Reading MOTA results")
    with open(os.path.join(RESULTS_DIR, "pedestrian_detailed.csv"), "r") as evaluationData:
        reader = csv.DictReader(evaluationData)
        for row in reader:
            if row["seq"] != "COMBINED": continue
            return float(row["MOTA"])

def evaluate(trackerConfig: dict) -> None:
    writeYaml(trackerConfig)
    runEvalSctips()
    return getMotaResults()

def main():
    LOGGER.info("Starting main")

    LOGGER.info("Evaluating base setup")
    setup: dict = getDefaultSetup()
    bestSetup: dict = copy.deepcopy(setup)
    bestMOTA: float = evaluate(setup)
    LOGGER.info(f"Achieved base MOTA score of {bestMOTA} with setup:\n {setup}")

    for key in AVAILABLE_SETUP_OPTIONS:
        options: list = AVAILABLE_SETUP_OPTIONS[key]
        for index in range(len(options)):
            LOGGER.info(f"Evaluating {key}-{options[index]}")

            setup[key] = options[index]
            setupMOTA: float = evaluate(setup)

            LOGGER.info(f"MOTA result: {setupMOTA}")
            if setupMOTA > bestMOTA:
                LOGGER.info(f"New MOTA best with setup:\n{setup}")

                bestSetup = copy.deepcopy(setup)
                bestMOTA = setupMOTA
        setup = copy.deepcopy(bestSetup)

    LOGGER.info(f"Found best setup:\n {setup}\n\n"
        + f"with the overall MOTA score of {bestMOTA}")

if __name__ == "__main__":
    LOGGER.info("Running script: " + __file__)
    main()
    LOGGER.info("Exiting from script: " + __file__)