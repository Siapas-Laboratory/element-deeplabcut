import os
import datajoint as dj
from collections import abc
from element_lab import lab
from element_animal import subject
from element_session import session_with_datetime as session
from element_deeplabcut import train, model
import cv2
from element_interface.utils import find_full_path
from element_animal.subject import Subject
from element_lab.lab import Source, Lab, Protocol, User, Project

MODEL_DIR = os.path.join(os.getcwd(), "dlc_models")
os.makedirs(MODEL_DIR, exist_ok= True)

VID_ID_ATTR = "recording_id"

if "custom" not in dj.config:
    dj.config["custom"] = {}

# overwrite dj.config['custom'] values with environment variables if available

dj.config["custom"]["database.prefix"] = os.getenv(
    "DATABASE_PREFIX", dj.config["custom"].get("database.prefix", "")
)

dj.config["custom"]["dlc_root_data_dir"] = os.getenv(
    "DLC_ROOT_DATA_DIR", dj.config["custom"].get("dlc_root_data_dir", "")
)

dj.config["custom"]["dlc_processed_data_dir"] = os.getenv(
    "DLC_PROCESSED_DATA_DIR", dj.config["custom"].get("dlc_processed_data_dir", "")
)

if "custom" not in dj.config:
    dj.config["custom"] = {}

db_prefix = dj.config["custom"].get("database.prefix", "")


# Declare functions for retrieving data
def get_dlc_root_data_dir() -> list:
    """Returns a list of root directories for Element DeepLabCut"""
    dlc_root_dirs = dj.config.get("custom", {}).get("dlc_root_data_dir")
    if not dlc_root_dirs:
        return None
    elif not isinstance(dlc_root_dirs, abc.Sequence):
        return list(dlc_root_dirs)
    else:
        return dlc_root_dirs


def get_dlc_processed_data_dir() -> str:
    """Returns an output directory relative to custom 'dlc_output_dir' root"""
    from pathlib import Path

    dlc_output_dir = dj.config.get("custom", {}).get("dlc_processed_data_dir")
    if dlc_output_dir:
        return Path(dlc_output_dir)
    else:
        return None


def get_dlc_root_model_dir() -> str:
    """Returns a root directory for Element DeepLabCut to store frozen copies of models"""
    return MODEL_DIR

__all__ = ["lab", "subject", "session", "train", "model", "Device"]

# Activate schemas -------------

lab.activate(db_prefix + "lab")
subject.activate(db_prefix + "subject", linking_module=__name__)
Experimenter = lab.User
Session = session.Session
session.activate(db_prefix + "session", linking_module=__name__)


@lab.schema
class Device(dj.Lookup):
    """Table for managing lab equipment.

    In Element DeepLabCut, this table is referenced by `model.VideoRecording`.
    The primary key is also used to generate inferred output directories when
    running pose estimation inference. Refer to the `definition` attribute
    for the table design.

    Attributes:
        device ( varchar(32) ): Device short name.
        modality ( varchar(64) ): Modality for which this device is used.
        description ( varchar(256) ): Optional. Description of device.
    """

    definition = """
    device             : varchar(32)
    ---
    modality           : varchar(64)
    description=null   : varchar(256)
    """
    contents = [
        ["Camera1", "Pose Estimation", "Panasonic HC-V380K"],
        ["Camera2", "Pose Estimation", "Panasonic HC-V770K"],
    ]

# Activate DeepLabCut schema -----------------------------------


train.activate(db_prefix + "train", linking_module=__name__)
model.activate(db_prefix + "model", linking_module=__name__)


@model.schema
class VideoRecording(dj.Manual):
    """Set of video recordings for DLC inferences.

    Attributes:
        Session (foreign key): Session primary key.
        recording_id (int): Unique recording ID.
        Device (foreign key): Device table primary key, used for default output
            directory path information.
    """

    definition = """
    -> Session
    recording_id: int
    ---
    -> Device
    """

    class File(dj.Part):
        """File IDs and paths associated with a given recording_id

        Attributes:
            VideoRecording (foreign key): Video recording primary key.
            file_path ( varchar(255) ): file path of video, relative to root data dir.
        """

        definition = """
        -> master
        file_id: int
        ---
        file_path: varchar(255)  # filepath of video, relative to root data directory
        """


@model.schema
class RecordingInfo(dj.Imported):
    """Automated table with video file metadata.

    Attributes:
        VideoRecording (foreign key): Video recording key.
        px_height (smallint): Height in pixels.
        px_width (smallint): Width in pixels.
        nframes (int): Number of frames.
        fps (int): Optional. Frames per second, Hz.
        recording_datetime (datetime): Optional. Datetime for the start of recording.
        recording_duration (float): video duration (s) from nframes / fps."""

    definition = """
    -> VideoRecording
    ---
    px_height                 : smallint  # height in pixels
    px_width                  : smallint  # width in pixels
    nframes                   : int  # number of frames 
    fps = NULL                : int       # (Hz) frames per second
    recording_datetime = NULL : datetime  # Datetime for the start of the recording
    recording_duration        : float     # video duration (s) from nframes / fps
    """

    @property
    def key_source(self):
        """Defines order of keys for make function when called via `populate()`"""
        return VideoRecording & VideoRecording.File

    def make(self, key):
        """Populates table with video metadata using CV2."""
        file_paths = (VideoRecording.File & key).fetch("file_path")

        nframes = 0
        px_height, px_width, fps = None, None, None

        for file_path in file_paths:
            file_path = (find_full_path(get_dlc_root_data_dir(), file_path)).as_posix()

            cap = cv2.VideoCapture(file_path)
            info = (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FPS)),
            )
            if px_height is not None:
                assert (px_height, px_width, fps) == info
            px_height, px_width, fps = info
            nframes += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        self.insert1(
            {
                **key,
                "px_height": px_height,
                "px_width": px_width,
                "nframes": nframes,
                "fps": fps,
                "recording_duration": nframes / fps,
            }
        )

def get_vid_paths(key) -> list:
    files = (VideoRecording.File & key).fetch("file_path")
    return [find_full_path(get_dlc_root_data_dir(), f) for f in files]

def get_vid_devices(key) -> list:
    recording_key = VideoRecording & key
    return [str(v) for v in (Device & recording_key).fetch1("KEY").values()]