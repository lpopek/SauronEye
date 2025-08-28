import os
import json
import pathlib
import logging
import traceback

from ultralytics import YOLO




CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_DIR: str = os.path.join(CURRENT_DIR, "models")
TRACKERS_DIR: str = os.path.join(CURRENT_DIR, "trackers")
DATASET_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../datasets"))
RESULTS_DIR: str = os.path.join(CURRENT_DIR, "results")




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(
            CURRENT_DIR, "eval.log"
        )),
        logging.StreamHandler()
    ]
)
LOGGER: logging.Logger = logging.getLogger(__name__)




def LoadConfig() -> dict:
    LOGGER.info("Loading config.json")
    with open(os.path.join(CURRENT_DIR, "config.json")) as json_file:
        try: config = json.load(json_file)
        except: raise Exception("Invalid config.json file")
    return config




def GetModelPaths(config: dict) -> list[str]:
    LOGGER.info("Fetching YOLO models")

    try: models: list[str] = config["models"]
    except KeyError: 
        raise Exception("Config error: 'models' " 
            + "property missing.")

    if not isinstance(models, list):
        raise Exception("Config error: 'models' " 
            + "property should be a list.")
    
    if len(models) <= 0:
        raise Exception("Config error: 'models' " 
            + "property is an empty list.")

    for model_name in models:
        if not isinstance(model_name, str):
            raise Exception("Config error: 'models' " 
                + "list should only contain strings.")
        
        _, file_extension = os.path.splitext(model_name)
        if file_extension != ".pt":
            raise Exception("Config error: all models " 
                + "should have a '.pt' file extension.")
    
    return [os.path.join(MODEL_DIR, model_name) for model_name in models]




def GetTrackers(config: dict) -> list[str]:
    LOGGER.info("Fetching tracker paths")

    try: trackers: list[str] = config["trackers"]
    except KeyError:
        raise Exception("Config error: 'trackers' "
            + "property missing.")
    
    if not isinstance(trackers, list):
        raise Exception("Config error: 'trackers' "
            + "property is an empty list.")
    
    for tracker_name in trackers:
        if not isinstance(tracker_name, str):
            raise Exception("Config error: 'trackers' "
                + "list should only contain strings.")
        
        _, file_extension = os.path.splitext(tracker_name)
        if file_extension != ".yaml":
            raise Exception("Config error: all trackers "
                + "should have a .yaml file extension.")
        
    return [os.path.join(TRACKERS_DIR, tracker) for tracker in trackers]




def GetDatasetPath(config: dict) -> str:
    LOGGER.info("Fetching MOT dataset path")
    
    try: path = config["dataset_path"]
    except KeyError:
        raise Exception("Config error: 'dataset_path' "
            + "property missing.")
    
    if not isinstance(path, str):
        raise Exception("Config error: 'dataset_path' "
            + "property should be a string.")
    
    return os.path.abspath(os.path.join(CURRENT_DIR, path))




def CreateOutputDirectory(
    model_name: str, 
    tracker_name: str, 
    dataset_name: str
) -> str:
    output_directory = RESULTS_DIR
    output_directory = os.path.join(output_directory, model_name)
    output_directory = os.path.join(output_directory, dataset_name)
    output_directory = os.path.join(output_directory, tracker_name)
    output_directory = os.path.join(output_directory, "data")
    try: os.makedirs(output_directory)
    except FileExistsError: pass
    return output_directory




def EvaluateModels(
    model_paths: list[str], 
    trackers: list[str], 
    dataset_path: str
) -> None:
    LOGGER.info("Starting evaluation") 

    dataset_name: str = os.path.basename(dataset_path)
    scene_paths: list[str] = [
        os.path.join(dataset_path, path) for path in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, path))
    ]

    for model_path in model_paths:
        model_name, _ = os.path.splitext(os.path.basename(model_path))
        LOGGER.info("Evaluating " + model_name)
        
        for tracker in trackers:
            
            tracker_name, _ = os.path.splitext(os.path.basename(tracker))
            LOGGER.info("Attached tracker: " + tracker_name)
            output_directory: str = CreateOutputDirectory(model_name, tracker_name, dataset_name)

            for scene in scene_paths:

                model = YOLO(model_path)
                
                scene_name = os.path.basename(scene)
                LOGGER.info("Evaluating scene " + scene)
                output_file = open(os.path.join(output_directory, scene_name + ".txt"), "w+")

                print(tracker)

                results = model.track(
                    source=os.path.join(scene, "img1"),
                    tracker=tracker,
                    classes=[0], # only detect 'pearson' class
                    persist=True
                )

                for frame_index, result in enumerate(results):
                    if result.boxes == None: continue

                    boxes = result.boxes.xywh.cpu().numpy()
                    confs = [conf * 100 for conf in result.boxes.conf.cpu().numpy()] # YOLO outputs normalized 0-1 values, MOT expects 0-100
                    if result.boxes.id is not None:
                        ids = [id for id in result.boxes.id.cpu().numpy()]
                    else:
                        ids = [-1 for _ in range(len(boxes))]

                    for i in range(len(boxes)):
                        x_center, y_center, width, height = boxes[i]
                        x = x_center - (width / 2)
                        y = y_center - (height / 2)

                        if ids[i] == -1: continue

                        output_file.write(
                            f"{frame_index + 1},{int(ids[i])},{x:.3f},{y:.3f},{width:.3f},{height:.3f},{confs[i]:.3f},-1,-1,-1\n"
                        )

                output_file.close()




def main() -> None:
    LOGGER.info("Starting main")

    try: config: dict = LoadConfig()
    except Exception:
        LOGGER.error("Failed to load config.json file\n\n" 
            + traceback.format_exc())
        return

    try: model_paths: list[str] = GetModelPaths(config)
    except Exception: 
        LOGGER.error("Invalid config.json file\n\n" 
            + traceback.format_exc())
        return
    
    try: trackers: list[str] = GetTrackers(config)
    except Exception:
        LOGGER.error("Invalid config.json file\n\n"
            + traceback.format_exc())
        return

    try: dataset_path: str = GetDatasetPath(config)
    except Exception:
        LOGGER.error("Invalid config.json file\n\n" 
            + traceback.format_exc())
        return

    EvaluateModels(
        model_paths, 
        trackers, 
        dataset_path
    )




if __name__ == "__main__":
    LOGGER.info("Running script: " + __file__)
    main()
    LOGGER.info("Exiting from script: " + __file__)