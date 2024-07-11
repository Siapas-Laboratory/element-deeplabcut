"""
Code adapted from the Mathis Lab
MIT License Copyright (c) 2022 Mackenzie Mathis
DataJoint Schema for DeepLabCut 2.x, Supports 2D and 3D DLC via triangulation.
"""

import datajoint as dj
import os
import cv2
import csv
from ruamel.yaml import YAML
import inspect
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
from element_interface.utils import find_full_path, find_root_directory
from .readers import dlc_reader
import shutil
import hashlib

schema = dj.schema()
_linking_module = None


def activate(
    model_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: bool = None,
):
    """Activate this schema.

    Args:
        model_schema_name (str): schema name on the database server
        create_schema (bool): when True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): when True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:
    Upstream tables:
        VideoRecording: a parent table to PoseEstimationTask identifying a video recording
    Functions:
        get_dlc_root_model_dir(): Returns absolute path for root directory
                                  for copying frozen models to.
        get_dlc_processed_data_dir(): Optional. Returns absolute path for processed
                                      data. Defaults to session video subfolder.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"
    assert hasattr(
        linking_module, "get_dlc_root_model_dir"
    ), "The linking module must specify a lookup function for a root model directory"

    global _linking_module
    _linking_module = linking_module

    # activate
    schema.activate(
        model_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# -------------- Functions required by element-deeplabcut ---------------

def get_dlc_processed_data_dir() -> Optional[str]:
    """Pulls relevant func from parent namespace. Defaults to DLC's project /videos/.

    Method in parent namespace should provide a string to a directory where DLC output
    files will be stored. If unspecified, output files will be stored in the
    session directory 'videos' folder, per DeepLabCut default.
    """
    if hasattr(_linking_module, "get_dlc_processed_data_dir"):
        return _linking_module.get_dlc_processed_data_dir()
    else:
        return None

def get_dlc_root_model_dir() -> str:
    return _linking_module.get_dlc_root_model_dir()
# ----------------------------- Table declarations ----------------------

@schema
class VideoRecording(dj.Manual):
    """Set of video recordings for DLC inferences.
    
    NOTE: this version of this table holds reference
    to a table called Video which in our case already associates
    a provided video with a Device. the version of this table
    in the public element-deeplabcut holds reference to a table 
    called Device instead. technically in our table Video
    refers to a single Video file but if we keep the Device 
    reference in the parent table and reference Video in the subtable
    we will get errors because again the Video table already references
    Device sow we can't 

    Attributes:
        Session (foreign key): Session primary key.
        recording_id (int): Unique recording ID.
        Device (foreign key): Device table primary key, used for default output
            directory path information.
    """

    definition = """
    -> Session
    recording_id: int
    -> Device
    ---
    good_frames = NULL : longblob # array where each row specifies a range of indices corresponding to valid frames
    """

    def get_vid_paths(self):
        files = (VideoRecording.File * self).fetch('file_path', order_by='file_id', squeeze=True)
        return [Path(f) for f in files]
    

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
        file_path: varchar(255)  # absolute filepath of video
        -> Video
        """

@schema
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
        file_paths = (VideoRecording & key).get_vid_paths()

        nframes = 0
        px_height, px_width, fps = None, None, None

        for file_path in file_paths:
            file_path = file_path.as_posix()

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

        
@schema
class BodyPart(dj.Lookup):
    """Body parts tracked by DeepLabCut models

    Attributes:
        body_part ( varchar(32) ): Body part short name.
        body_part_description ( varchar(1000),optional ): Full description

    """

    definition = """
    body_part                : varchar(32)
    ---
    body_part_description='' : varchar(1000)
    """

    @classmethod
    def extract_new_body_parts(cls, dlc_config: dict, verbose: bool = True):
        """Returns list of body parts present in dlc config, but not BodyPart table.

        Args:
            dlc_config ( varchar(255) ): Path to a config.y*ml.
            verbose (bool): Default True. Print both existing and new items to console.
        """
        if not isinstance(dlc_config, dict):
            dlc_config_fp = find_full_path(get_dlc_root_model_dir(), Path(dlc_config))
            assert dlc_config_fp.exists() and dlc_config_fp.suffix in (
                ".yml",
                ".yaml",
            ), f"dlc_config is neither dict nor filepath\n Check: {dlc_config_fp}"
            if dlc_config_fp.suffix in (".yml", ".yaml"):
                yaml = YAML(typ="safe", pure=True)
                with open(dlc_config_fp, "rb") as f:
                    dlc_config = yaml.load(f)
        # -- Check and insert new BodyPart --
        assert "bodyparts" in dlc_config, f"Found no bodyparts section in {dlc_config}"
        tracked_body_parts = cls.fetch("body_part")
        new_body_parts = np.setdiff1d(dlc_config["bodyparts"], tracked_body_parts)
        if verbose:  # Added to silence duplicate prompt during `insert_new_model`
            print(f"Existing body parts: {tracked_body_parts}")
            print(f"New body parts: {new_body_parts}")
        return new_body_parts

    @classmethod
    def insert_from_config(
        cls, dlc_config: dict, descriptions: list = None, prompt=True
    ):
        """Insert all body parts from a config file.

        Args:
            dlc_config ( varchar(255) ): Path to a config.y*ml.
            descriptions (list): Optional. List of strings describing new body parts.
            prompt (bool): Optional, default True. Prompt for confirmation before insert.
        """

        # handle dlc_config being a yaml file
        new_body_parts = cls.extract_new_body_parts(dlc_config, verbose=False)
        if new_body_parts is not None:  # Required bc np.array is ambiguous as bool
            if descriptions:
                assert len(descriptions) == len(new_body_parts), (
                    "Descriptions list does not match "
                    + " the number of new_body_parts"
                )
                print(f"New descriptions: {descriptions}")
            if descriptions is None:
                descriptions = ["" for x in range(len(new_body_parts))]

            if (
                prompt
                and dj.utils.user_choice(
                    f"Insert {len(new_body_parts)} new body " + "part(s)?"
                )
                != "yes"
            ):
                print("Canceled insert.")
                return
            cls.insert(
                [
                    {"body_part": b, "body_part_description": d}
                    for b, d in zip(new_body_parts, descriptions)
                ]
            )

@schema
class BodyPartSet(dj.Lookup):
    definition = """
    bp_set_hash     : varchar(255)  # hash for the set of body parts
    """

    class BodyPart(dj.Part):
        definition = """
        -> master
        -> BodyPart
        """

    @classmethod
    def get_bp_set(cls, body_parts):
        group_str = '_'.join(sorted(body_parts))
        group_hash = hashlib.sha256(group_str.encode()).hexdigest()
        ins_tuple = {'bp_set_hash': group_hash}
        cls.insert1(ins_tuple, skip_duplicates=True)
        for bp in body_parts:
            _ins_tuple = {
                **ins_tuple,
                'body_part': bp
            }
            cls.BodyPart.insert1(_ins_tuple, 
                                skip_duplicates=True)
        return ins_tuple

@schema
class Model(dj.Manual):
    """DeepLabCut Models applied to generate pose estimations.

    Attributes:
        model_name ( varchar(64) ): User-friendly model name.
        task ( varchar(32) ): Task in the config yaml.
        date ( varchar(16) ): Date in the config yaml.
        iteration (int): Iteration/version of this model.
        snapshotindex (int): Which snapshot for prediction (if -1, latest).
        shuffle (int): Which shuffle of the training dataset.
        trainingsetindex (int): Which training set fraction to generate model.
        scorer ( varchar(64) ): Scorer/network name - DLC's GetScorerName().
        config_template (longblob): Dictionary of the config for analyze_videos().
        project_path ( varchar(255) ): DLC's project_path in config relative to root.
        model_prefix ( varchar(32) ): Optional. Prefix for model files.
        model_description ( varchar(300) ): Optional. User-entered description.
        TrainingParamSet (foreign key): Optional. Training parameters primary key.

    Note:
        Models are uniquely identified by the union of task, date, iteration, shuffle,
        snapshotindex, and trainingsetindex.
    """

    definition = """
    model_name           : varchar(64)  # User-friendly model name
    version              : int          # version of the frozen model
    ---
    task                 : varchar(32)  # Task in the config yaml
    date                 : varchar(16)  # Date in the config yaml
    iteration            : int          # Iteration/version of this model
    snapshotindex        : int          # which snapshot for prediction (if -1, latest)
    shuffle              : int          # Shuffle (1) or not (0)
    trainingsetindex     : int          # Index of training fraction list in config.yaml
    scorer               : varchar(64)  # Scorer/network name - DLC's GetScorerName()
    config_template      : longblob     # Dictionary of the config for analyze_videos()
    project_path         : varchar(255) # DLC's project_path in config relative to root
    model_prefix=''      : varchar(32)
    model_description='' : varchar(300)
    -> BodyPartSet
    -> [nullable] train.TrainingParamSet
    """
    # project_path is the only item required downstream in the pose schema

    def delete(self, **kwargs):
        root_dir = Path(get_dlc_root_model_dir())
        for model_name, v, path in zip(*self.fetch('model_name', 'version', 'project_path')):
            full_path = root_dir/path
            reply = input(f"Are you sure you want to delete {model_name} version {v} at {full_path.as_posix()}? [y/n]: ")
            if reply.lower()[0] == 'y':
                row = Model & {'model_name': model_name, 'version': v}
                del_count = super(Model, row).delete()
                if del_count == 1:
                    print(f'successfully deleted {model_name} version {v}, deleting associated project directory')
                    shutil.rmtree(full_path)

    @classmethod
    def insert_new_model(
        cls,
        model_name: str,
        experimenter_name: str,
        video_file_keys: list,
    ):
        raise NotImplemented
        import deeplabcut as dlc
        root_dit = get_dlc_root_model_dir()
        config_file = dlc.create_new_project(model_name, experimenter_name)
    
    def extract_frames(self):
        raise NotImplemented
    
    @classmethod
    def insert_trained_model(
        cls,
        model_name: str,
        dlc_config,
        *,
        shuffle: int,
        trainingsetindex,
        model_description="",
        model_prefix="",
        paramset_idx: int = None,
        prompt=True,
        params=None,
    ):
        """Insert new model into the dlc.Model table.

        Args:
            model_name (str): User-friendly name for this model.
            dlc_config ( varchar(255) ): Path to a config.y*ml.
            shuffle (int): Which shuffle of the training dataset.
            trainingsetindex (int): Index of training fraction list in config.yaml.
            model_description (str): Optional. Description of this model.
            model_prefix (str): Optional. Filename prefix used across DLC project
            paramset_idx (int): Optional. Index from the TrainingParamSet table
            prompt (bool): Optional. Prompt the user with all info before inserting.
            params (dict): Optional. If dlc_config is path, dict of override items
        """

        from deeplabcut.utils.auxiliaryfunctions import GetScorerName, write_config  # isort:skip

        assert Path(dlc_config).suffix in (".yml", ".yaml"), "Config file was expexted to be a yaml file"

        # copy frozen model directory to database managed directory
        root_dir = get_dlc_root_model_dir()
        project_path = Path(root_dir)/model_name
        version = 0
        while project_path.exists():
            version += 1
            project_path = Path(root_dir)/f"{model_name}_v{version}"
        os.mkdir(project_path)

        yaml = YAML(typ="safe", pure=True)
        with open(dlc_config, "rb") as f:
            it = yaml.load(f)['iteration']

        for i in Path(dlc_config).parent.iterdir():
            print(f'copying {i}')
            if i.is_dir():
                if i.name in ['dlc-models', 'training-datasets']:
                    os.mkdir(project_path/i.name)
                    it_dir = f'iteration-{it}'
                    if it_dir in [j.name for j in i.iterdir()]:
                        shutil.copytree(i/it_dir, project_path/i.name/it_dir, symlinks=True)
                else:
                    shutil.copytree(i, project_path/i.name, symlinks=True)
            elif i.is_file():
                shutil.copyfile(i, project_path/i.name)
        
        dlc_config_fp = project_path/Path(dlc_config).name
        assert dlc_config_fp.exists(), (
            "dlc_config is not a filepath" + f"\n Check: {dlc_config_fp}"
        )

        # handle dlc_config being a yaml file
        yaml = YAML(typ="safe", pure=True)
        with open(dlc_config_fp, "rb") as f:
            dlc_config = yaml.load(f)
        if isinstance(params, dict):
            dlc_config.update(params)

        # ---- Get and resolve project path ----
        dlc_config["project_path"] = project_path.as_posix()  # update if different
        write_config(dlc_config_fp.as_posix(), dlc_config)

        # ---- Verify config ----
        needed_attributes = [
            "Task",
            "date",
            "iteration",
            "snapshotindex",
            "TrainingFraction",
        ]
        for attribute in needed_attributes:
            assert attribute in dlc_config, f"Couldn't find {attribute} in config"

        # ---- Get scorer name ----
        # "or 'f'" below covers case where config returns None. str_to_bool handles else
        scorer_legacy = str_to_bool(dlc_config.get("scorer_legacy", "f"))

        dlc_scorer = GetScorerName(
            cfg=dlc_config,
            shuffle=shuffle,
            trainFraction=dlc_config["TrainingFraction"][int(trainingsetindex)],
            modelprefix=model_prefix,
        )[scorer_legacy]
        if dlc_config["snapshotindex"] == -1:
            dlc_scorer = "".join(dlc_scorer.split("_")[:-1])

        # ---- Insert ----
        model_dict = {
            "model_name": model_name,
            "version": version,
            "model_description": model_description,
            "scorer": dlc_scorer,
            "task": dlc_config["Task"],
            "date": dlc_config["date"],
            "iteration": dlc_config["iteration"],
            "snapshotindex": dlc_config["snapshotindex"],
            "shuffle": shuffle,
            "trainingsetindex": int(trainingsetindex),
            "project_path": project_path.relative_to(root_dir).as_posix(),
            "paramset_idx": paramset_idx,
            "config_template": dlc_config,
        }

        # -- prompt for confirmation --
        if prompt:
            print("--- DLC Model specification to be inserted ---")
            for k, v in model_dict.items():
                if k != "config_template":
                    print("\t{}: {}".format(k, v))
                else:
                    print("\t-- Template/Contents of config.yaml --")
                    for k, v in model_dict["config_template"].items():
                        print("\t\t{}: {}".format(k, v))

        if (
            prompt
            and dj.utils.user_choice("Proceed with new DLC model insert?") != "yes"
        ):
            print("Canceled insert.")
            return

        def _do_insert():
            # Returns array, so check size for unambiguous truth value
            if BodyPart.extract_new_body_parts(dlc_config, verbose=False).size > 0:
                BodyPart.insert_from_config(dlc_config, prompt=prompt)
            model_dict['bp_set_hash'] = BodyPartSet.get_bp_set(dlc_config['bodyparts'])['bp_set_hash']
            cls.insert1(model_dict)

        # ____ Insert into table ----
        if cls.connection.in_transaction:
            _do_insert()
        else:
            with cls.connection.transaction:
                _do_insert()

        return {'model_name': model_name, 'version': version}


@schema
class ModelEvaluation(dj.Computed):
    """Performance characteristics model calculated by `deeplabcut.evaluate_network`

    Attributes:
        Model (foreign key): Model name.
        train_iterations (int): Training iterations.
        train_error (float): Optional. Train error (px).
        test_error (float): Optional. Test error (px).
        p_cutoff (float): Optional. p-cutoff used.
        train_error_p (float): Optional. Train error with p-cutoff.
        test_error_p (float): Optional. Test error with p-cutoff."""

    definition = """
    -> Model
    ---
    train_iterations   : int   # Training iterations
    train_error=null   : float # Train error (px)
    test_error=null    : float # Test error (px)
    p_cutoff=null      : float # p-cutoff used
    train_error_p=null : float # Train error with p-cutoff
    test_error_p=null  : float # Test error with p-cutoff
    """

    def make(self, key):
        from deeplabcut import evaluate_network  # isort:skip
        from deeplabcut.utils.auxiliaryfunctions import (
            get_evaluation_folder,
        )  # isort:skip

        """.populate() method will launch evaluation for each unique entry in Model."""
        dlc_config, project_path, model_prefix, shuffle, trainingsetindex = (
            Model & key
        ).fetch1(
            "config_template",
            "project_path",
            "model_prefix",
            "shuffle",
            "trainingsetindex",
        )

        project_path = find_full_path(get_dlc_root_model_dir(), project_path)
        yml_path, _ = dlc_reader.read_yaml(project_path)

        evaluate_network(
            yml_path,
            Shuffles=[shuffle],  # this needs to be a list
            trainingsetindex=trainingsetindex,
            comparisonbodyparts="all",
        )

        eval_folder = get_evaluation_folder(
            trainFraction=dlc_config["TrainingFraction"][trainingsetindex],
            shuffle=shuffle,
            cfg=dlc_config,
            modelprefix=model_prefix,
        )
        eval_path = project_path / eval_folder
        assert eval_path.exists(), f"Couldn't find evaluation folder:\n{eval_path}"

        eval_csvs = list(eval_path.glob("*csv"))
        max_modified_time = 0
        for eval_csv in eval_csvs:
            modified_time = os.path.getmtime(eval_csv)
            if modified_time > max_modified_time:
                eval_csv_latest = eval_csv
        with open(eval_csv_latest, newline="") as f:
            results = list(csv.DictReader(f, delimiter=","))[0]
        # in testing, test_error_p returned empty string
        self.insert1(
            dict(
                key,
                train_iterations=results["Training iterations:"],
                train_error=results[" Train error(px)"],
                test_error=results[" Test error(px)"],
                p_cutoff=results["p-cutoff used"],
                train_error_p=results["Train error with p-cutoff"],
                test_error_p=results["Test error with p-cutoff"],
            )
        )


@schema
class PoseEstimationTask(dj.Manual):
    """Staging table for pairing of video recording and model before inference.

    Attributes:
        VideoRecording (foreign key): Video recording key.
        Model (foreign key): Model name.
        task_mode (load or trigger): Optional. Default load. Or trigger computation.
        pose_estimation_output_dir ( varchar(255) ): Optional. Output dir
        pose_estimation_params (longblob): Optional. Params for DLC's analyze_videos
                                           params, if not default."""

    definition = """
    -> VideoRecording                           # Session -> Recording + File part table
    -> Model                                    # Must specify a DLC project_path
    ---
    task_mode='load' : enum('load', 'trigger')  # load results or trigger computation
    pose_estimation_output_dir='': varchar(255) # output dir
    pose_estimation_params=null  : longblob     # analyze_videos params, if not default
    """

    @classmethod
    def infer_output_dir(cls, key: dict, relative: bool = False, mkdir: bool = False):
        """Return the expected pose_estimation_output_dir.

        Spaces in model name are replaced with hyphens.
        Based on convention: / video_dir / Device_{}_Recording_{}_Model_{}

        Args:
            key: DataJoint key specifying a pairing of VideoRecording and Model.
            relative (bool): Report directory relative to get_dlc_processed_data_dir().
            mkdir (bool): Default False. Make directory if it doesn't exist.
        """
        video_filepath =  (VideoRecording & key).get_vid_paths()[0]
        root_dir = video_filepath.parent
        recording_key = VideoRecording & key
        device = "-".join(
            str(v)
            for v in (_linking_module.Device & recording_key).fetch1("KEY").values()
        )

        if get_dlc_processed_data_dir():
            processed_dir = Path(get_dlc_processed_data_dir())
        else:  # if processed not provided, default to where video is
            processed_dir = root_dir

        output_dir = (
            processed_dir
            / (
                f'device_{device}_'
                + f'session_{key[_linking_module.Session.primary_key[0]]}_'
                + f'recording_{key["recording_id"]}_model_'
                + f'{key["model_name"].replace(" ", "-")}_'
                + f'version_{key["version"]}'
            )
        )
        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def generate(
        cls,
        video_recording_key: dict,
        model_name: str,
        version: int,
        *,
        task_mode: str = None,
        analyze_videos_params: dict = None,
    ):
        """Insert PoseEstimationTask in inferred output dir.

        Based on the convention / video_dir / device_{}_recording_{}_model_{}

        Args:
            video_recording_key (dict): DataJoint key specifying a VideoRecording.

            model_name (str): Name of DLC model (from Model table) to be used for inference.
            task_mode (str): Default 'trigger' computation. Or 'load' existing results.
            analyze_videos_params (dict): Optional. Parameters passed to DLC's analyze_videos:
                videotype, gputouse, save_as_csv, batchsize, cropping, TFGPUinference,
                dynamic, robust_nframes, allow_growth, use_shelve
        """
        output_dir = cls.infer_output_dir(
            {**video_recording_key, "model_name": model_name, "version": version},
            relative=False,
            mkdir=True,
        )

        if task_mode is None:
            try:
                _ = dlc_reader.PoseEstimation(output_dir)
            except FileNotFoundError:
                task_mode = "trigger"
            else:
                task_mode = "load"

        cls.insert1(
            {
                **video_recording_key,
                "model_name": model_name,
                "version": version,
                "task_mode": task_mode,
                "pose_estimation_params": analyze_videos_params,
                "pose_estimation_output_dir": output_dir.as_posix(),
            }
        )

    insert_estimation_task = generate


@schema
class PoseEstimation(dj.Computed):
    """Results of pose estimation.

    Attributes:
        PoseEstimationTask (foreign key): Pose Estimation Task key.
        post_estimation_time (datetime): time of generation of this set of DLC results.
    """

    definition = """
    -> PoseEstimationTask
    ---
    pose_estimation_time: datetime  # time of generation of this set of DLC results
    """

    # class BodyPartPosition(dj.Part):
    #     """Position of individual body parts by frame index

    #     Attributes:
    #         PoseEstimation (foreign key): Pose Estimation key.
    #         Model.BodyPart (foreign key): Body Part key.
    #         frame_index (longblob): Frame index in model.
    #         x_pos (longblob): X position.
    #         y_pos (longblob): Y position.
    #         z_pos (longblob): Optional. Z position.
    #         likelihood (longblob): Model confidence."""

    #     definition = """ # uses DeepLabCut h5 output for body part position
    #     -> master
    #     -> Model.BodyPart
    #     ---
    #     frame_index : longblob     # frame index in model
    #     x_pos       : longblob
    #     y_pos       : longblob
    #     z_pos=null  : longblob
    #     likelihood  : longblob
    #     """

    def make(self, key):
        """.populate() method will launch training for each PoseEstimationTask"""
        # ID model and directories
        dlc_model = (Model & key).fetch1()
        task_mode, output_dir = (PoseEstimationTask & key).fetch1(
            "task_mode", "pose_estimation_output_dir"
        )


        # Triger PoseEstimation
        if task_mode == "trigger":
            # Triggering dlc for pose estimation required:
            # - project_path: full path to the directory containing the trained model
            # - video_filepaths: full paths to the video files for inference
            # - analyze_video_params: optional parameters to analyze video
            project_path = find_full_path(
                get_dlc_root_model_dir(), dlc_model["project_path"]
            )
            video_filepaths = [v.as_posix() for v in (VideoRecording & key).get_vid_paths()]
            analyze_video_params = (PoseEstimationTask & key).fetch1(
                "pose_estimation_params"
            ) or {}

            dlc_reader.do_pose_estimation(
                key,
                video_filepaths,
                dlc_model,
                project_path,
                output_dir,
                **analyze_video_params,
            )

        dlc_result = dlc_reader.PoseEstimation(output_dir)
        creation_time = datetime.fromtimestamp(dlc_result.creation_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # body_parts = [
        #     {
        #         **key,
        #         "body_part": k,
        #         "frame_index": np.arange(dlc_result.nframes),
        #         "x_pos": v["x"],
        #         "y_pos": v["y"],
        #         "z_pos": v.get("z"),
        #         "likelihood": v["likelihood"],
        #     }
        #     for k, v in dlc_result.data.items()
        # ]

        self.insert1({**key, "pose_estimation_time": creation_time})
        # self.BodyPartPosition.insert(body_parts)

    # @classmethod
    # def get_trajectory(cls, key: dict, body_parts: list = "all") -> pd.DataFrame:
    #     """Returns a pandas dataframe of coordinates of the specified body_part(s)

    #     Args:
    #         key (dict): A DataJoint query specifying one PoseEstimation entry.
    #         body_parts (list, optional): Body parts as a list. If "all", all joints

    #     Returns:
    #         df: multi index pandas dataframe with DLC scorer names, body_parts
    #             and x/y coordinates of each joint name for a camera_id, similar to
    #              output of DLC dataframe. If 2D, z is set of zeros
    #     """
    #     model_name = key["model_name"]

    #     if body_parts == "all":
    #         body_parts = (cls.BodyPartPosition & key).fetch("body_part")
    #     elif not isinstance(body_parts, list):
    #         body_parts = list(body_parts)

    #     df = None
    #     for body_part in body_parts:
    #         x_pos, y_pos, z_pos, likelihood = (
    #             cls.BodyPartPosition & {**key, "body_part": body_part}
    #         ).fetch1("x_pos", "y_pos", "z_pos", "likelihood")
    #         if not z_pos:
    #             z_pos = np.zeros_like(x_pos)

    #         a = np.vstack((x_pos, y_pos, z_pos, likelihood))
    #         a = a.T
    #         pdindex = pd.MultiIndex.from_product(
    #             [[model_name], [body_part], ["x", "y", "z", "likelihood"]],
    #             names=["scorer", "bodyparts", "coords"],
    #         )
    #         frame = pd.DataFrame(a, columns=pdindex, index=range(0, a.shape[0]))
    #         df = pd.concat([df, frame], axis=1)
    #     return df


def str_to_bool(value) -> bool:
    """Return whether the provided string represents true. Otherwise false.

    Args:
        value (any): Any input

    Returns:
        bool (bool): True if value in ("y", "yes", "t", "true", "on", "1")
    """
    # Due to distutils equivalent depreciation in 3.10
    # Adopted from github.com/PostHog/posthog/blob/master/posthog/utils.py
    if not value:
        return False
    return str(value).lower() in ("y", "yes", "t", "true", "on", "1")
