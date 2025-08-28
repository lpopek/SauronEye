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
    with open(os.path.join(CURRENT_DIR, "config.json")) as jsonFile:
        try: config = json.load(jsonFile)
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

    for modelName in models:
        if not isinstance(modelName, str):
            raise Exception("Config error: 'models' " 
                + "list should only contain strings.")
        
        _, fileExtension = os.path.splitext(modelName)
        if fileExtension!= ".pt":
            raise Exception("Config error: all models " 
                + "should have a '.pt' file extension.")
    
    return [os.path.join(MODEL_DIR, modelName) for modelName in models]




def GetTrackers(config: dict) -> list[str]:
    LOGGER.info("Fetching tracker paths")

    try: trackers: list[str] = config["trackers"]
    except KeyError:
        raise Exception("Config error: 'trackers' "
            + "property missing.")
    
    if not isinstance(trackers, list):
        raise Exception("Config error: 'trackers' "
            + "property is an empty list.")
    
    for trackerName in trackers:
        if not isinstance(trackerName, str):
            raise Exception("Config error: 'trackers' "
                + "list should only contain strings.")
        
        _, file_extension = os.path.splitext(trackerName)
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
    modelName: str, 
    trackerName: str, 
    datasetName: str
) -> str:
    outputDirectory = RESULTS_DIR
    outputDirectory = os.path.join(outputDirectory, modelName)
    outputDirectory = os.path.join(outputDirectory, datasetName)
    outputDirectory = os.path.join(outputDirectory, trackerName)
    outputDirectory = os.path.join(outputDirectory, "data")
    try: os.makedirs(outputDirectory)
    except FileExistsError: pass
    return outputDirectory 




def EvaluateModels(
    modelPaths: list[str], 
    trackers: list[str], 
    datasetPath: str
) -> None:
    LOGGER.info("Starting evaluation") 

    datasetName: str = os.path.basename(datasetPath)
    scenePaths: list[str] = [
        os.path.join(datasetPath, path) for path in os.listdir(datasetPath) 
        if os.path.isdir(os.path.join(datasetPath, path))
    ]

    for modelPath in modelPaths:
        modelName, _ = os.path.splitext(os.path.basename(modelPath))
        LOGGER.info("Evaluating " + modelName)
        
        for tracker in trackers:
            
            trackerName, _ = os.path.splitext(os.path.basename(tracker))
            LOGGER.info("Attached tracker: " + trackerName)
            outputDirectory : str = CreateOutputDirectory(modelName, trackerName, datasetName)

            for scene in scenePaths:

                model = YOLO(modelPath)
                
                sceneName = os.path.basename(scene)
                LOGGER.info("Evaluating scene " + scene)
                outputFile = open(os.path.join(outputDirectory, sceneName+ ".txt"), "w+")

                print(tracker)

                results = model.track(
                    source=os.path.join(scene, "img1"),
                    tracker=tracker,
                    classes=[0], # only detect 'pearson' class
                    persist=True
                )

                for frameIndex, result in enumerate(results):
                    if result.boxes == None: continue

                    boxes = result.boxes.xywh.cpu().numpy()
                    confs = [conf * 100 for conf in result.boxes.conf.cpu().numpy()] # YOLO outputs normalized 0-1 values, MOT expects 0-100
                    if result.boxes.id is not None:
                        ids = [id for id in result.boxes.id.cpu().numpy()]
                    else:
                        ids = [-1 for _ in range(len(boxes))]

                    for i in range(len(boxes)):
                        xCenter, yCenter, width, height = boxes[i]
                        x = xCenter - (width / 2)
                        y = yCenter - (height / 2)

                        if ids[i] == -1: continue

                        outputFile.write(
                            f"{frameIndex + 1},{int(ids[i])},{x:.3f},{y:.3f},{width:.3f},{height:.3f},{confs[i]:.3f},-1,-1,-1\n"
                        )

                outputFile.close()




def main() -> None:
    LOGGER.info("Starting main")

    try: config: dict = LoadConfig()
    except Exception:
        LOGGER.error("Failed to load config.json file\n\n" 
            + traceback.format_exc())
        return

    try: modelPaths: list[str] = GetModelPaths(config)
    except Exception: 
        LOGGER.error("Invalid config.json file\n\n" 
            + traceback.format_exc())
        return
    
    try: trackers: list[str] = GetTrackers(config)
    except Exception:
        LOGGER.error("Invalid config.json file\n\n"
            + traceback.format_exc())
        return

    try: datasetPath: str = GetDatasetPath(config)
    except Exception:
        LOGGER.error("Invalid config.json file\n\n" 
            + traceback.format_exc())
        return

    EvaluateModels(
        modelPaths, 
        trackers, 
        datasetPath 
    )




if __name__ == "__main__":
    LOGGER.info("Running script: " + __file__)
    main()
    LOGGER.info("Exiting from script: " + __file__)