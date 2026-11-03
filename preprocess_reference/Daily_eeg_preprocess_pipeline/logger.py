"""
Contains the PreprocessLogger class to be imported.

Specific to this preproces pipeline.

Author: ddh

Data: July 3, 2026
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Tuple, Optional, Literal, Any

import mne # Because I used the type hint mne.io.BaseRaw... But import mne here shouldn't increase expenses.
 

class PreprocessLogger:
    """
    A unified logger for preprocess metadata.
    """
    def __init__(
        self, 
        subject_id: Union[int, str], 
        session: Union[int, str],
        ) -> None:
        """
        Args:
            subject_id (Union[int, str]): the ID of the subject, from 1 to 15.
            session (Union[int, str]): The session of this trial (e.g. 01, 10).
            trial_path (Union[str, Path]): The path to directory containing trial data (.bdf).
        """
        self.subject_id: int = int(subject_id)
        self.session: int = int(session)

        self.log_data: Dict[str, Any] = {
            "Subject": int(subject_id),
            "Session": int(session),
            "Error": {}, # TODO?
            "BeginTime": None,
            "FinalLength": None, # The final length after cropping.

            "SamplingFrequency": None,
            "Filters": {}, # {type: "bandpass", range: (0, 47)}, {type: "notch", center: 50, width: 4}, in Hz.
            "1stBadChannels": None, # In full time length. All channels.
            "2ndBadChannels": None, # In one minu segments. Exclude some channels.
            "3rdBadChannels": None, # In one minu segments. All channels.

            "ICAComponents": {},
            "ICAComponentsRemoved": {},
            "ReReference": None, # A str demonstrating re-referencing method (e.g. "average").
        }

    def log_error_trial(self, error: str, details: Any) -> None:
        """
        Log the information of abnormal trials.

        Args:
            error (str): A short str describing the error reason.
            details (Any): An item contains the detailed information (e.g. Dict).
        ---
        """
        self.log_data["Error"]["error"] = error
        self.log_data["Error"]["details"] = details 

    def log_begin_time(self, begin_time: Union[str, datetime]) -> None:
        """
        Log the initial time of target trial.

        Args:
            begin_time (str): The time/date info for target trial (e.g. "2026-03-28 14:37:08+00:00").
        ---
        """
        self.log_data["BeginTime"] = str(begin_time)

    def log_data_length(self, raw: mne.io.BaseRaw, ) -> None:
        """
        Receive a raw object, then calculate and log its physical length.

        Args:  
            raw (mne.io.BaseRaw): The raw object who's length need to be calculated and logged.
        ---
        """
        total_secs: float = raw.n_times / raw.info['sfreq']

        # Calculated the time.
        minutes, seconds = divmod(total_secs, 60)
        hours, minutes = divmod(minutes, 60)

        output_str = f"{int(hours):02d}h {int(minutes):02d}m {seconds:06.3f}s, total {total_secs} secs"
        self.log_data["FinalLength"] = output_str

    def log_sfreq(self, sfreq: float) -> None:
        self.log_data["SamplingFrequency"] = float(sfreq) # Use float is standard, rather than integer.

    def log_filtering(
        self, 
        filter_type: Literal["bandpass", "notch"], # Use "Literal" as type hints to limit the input.
        filter_params: Tuple[float, float]
    ) -> None: 
        """
        Log the filter info into the metadata.

        Args:
            filter_type Literal["bandpass", "notch"]: The specific type of filter applied.
            filter_parms: (Tuple[float, float]):
                - If "bandpass": A tuple of (high_pass_freq, low_pass_freq) in Hz.
                - If "notch": A tuple of (center_freq, bandwidth) in Hz.

        Raises:
            ValueError: If an unsupported "filter_type" is provided.
        """
        if filter_type == "bandpass":
            self.log_data["Filters"]["bandpass"] = {
                "type": "bandpass",
                "range": filter_params
            }
        elif filter_type == "notch":
            self.log_data["Filters"]["notch"] = {
                "type": "notch",
                "center": filter_params[0],
                "width": filter_params[1]
            }

    def log_bad_channels(
        self, 
        time: Literal["1st", "2nd", "3rd"],
        bad_channels: Union[List[Tuple[int, str, str]], Dict[str, Any]]
        ) -> None:
        """
        Log the bad channels information after detection.

        Supports both full-length list logs and chunked dictionary logs.

        Args:
            time (Literal["1st", "2nd", "3rd"]): Specify which time/phase this bad channel 
                                                 detection is responsible for.
            bad_channels (Union[List[Tuple[int, str, str]], Dict[str, Any]):
                - The detail information of the bad channels.
                - A list containing bad channel details per item for the 1st detection.
                - A dictionary containing detailed information of all processed chunks.
            ---
        """
        self.log_data[f"{time}BadChannels"] = bad_channels

    def log_ica_removal(
        self,
        full_list: List[Tuple[int, str, float]],
        exclude_list: List[Tuple[int, str, float]],
        total_count: int,
        exclude_count: int,
        ) -> None:
        """
        Log the information of removed components after ICA.

        Args:
            full_list (List[Tuple[int, str, float]]): A list containing info for all components as a packed tuple.
                                                      Format (component_idx, identified_name, component_confidence).
            exclude_list (List[Tuple[int, str, float]]): A list containing info for rejected components as a packed tuple.
            total_count (int): The total number of all components.
            exclude_count (int): The total number of the removed components.
        ---
        """
        self.log_data["ICAComponents"]["total"] = int(total_count)
        self.log_data["ICAComponents"]["detail"] = [
            f'component idx:{a} name:"{b}" confidence:{c:.4f}' for a, b, c in full_list
            ]

        self.log_data["ICAComponentsRemoved"]["total"] = int(exclude_count)
        self.log_data["ICAComponentsRemoved"]["detail"] = [
            f'component idx:{a} name:"{b}" confidence:{c:.4f}' for a, b, c in exclude_list
            ]

    def log_re_reference(self, method: str) -> None:
        """
        Log the re-reference method adopted during the preprocessing.

        Args:
            method (str): A short str explaining the re-referencing method (e.g. "average").
        ---
        """
        self.log_data["ReReference"] = method

    def save_log(self, output_dir: Union[str, Path], filename: str) -> None:
        """
        Save the logged data as a JSON file. Default save to the trial_path,
        next to the EEG data file.
        """
        # if output_dir is None:
        #     output_dir = self.trial_path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath: Path = output_dir / f"{Path(filename).stem}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=4)
        # print(f"The log has been saved to {filepath}")
        logging.info(f"The log has been saved to {filepath}")
