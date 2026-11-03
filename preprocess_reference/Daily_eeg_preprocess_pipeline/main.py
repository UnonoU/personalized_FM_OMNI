"""
The main pipeline for EEG data preprocess.

Support reading mutiple formats of EEG data (.bdf, .edf, .fif, BrainVision, EEGLab).

For single-processing only. Run "run_multiprocessing.py" for multiprocessing tasks.

Author: ddh

Date: July 3, 2026
"""

import argparse
import time
import logging
import warnings
from typing import List, Optional, Dict, Tuple, Literal, Union
from pathlib import Path

import mne

from logger import PreprocessLogger
from utils import parse_sub_ses_from_path, detect_channel_noise, detect_channel_flatline, apply_ica_denoise, process_local_bad_channels, save_as_fif


# Global constants.
TARGET_SFREQ: int = 200  # Target sampling frequency, unit Hz.

FILTER_FREQ: Tuple[float, float] = (1.0, 47.0)  # Bandpass filter frequencies in Hz.
NOTCH_FREQ: int = 50  # Notch filter frequency in Hz.
NOTCH_WIDTH: int = 4  # Notch filter width in Hz.
CUT_LENGTH: List[int] = [3, 60]  # The lengths to be cut after filtering on each side, in seconds.

ICA_PROB_THRESHOLD: float = 0.8  # Confidence threshold (0 to 1) for ICLabel against artifacts.

BAD_CHANNELS_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "constant_noise": (3.0, 0.4),  # (multiplier, ratio) for bad channels detection logic.
    "abrupt_noise": (30.0, 0.01),
}

INTERPOLATION_THRESHOLD: int = 6  # Threshold for quitting interpolate the segment if too many bad channels.

EEG_CHANNELS: List[str] = [  # The EEG channels for 10-20 system. Total 59.
    'Fpz', 'Fp1', 'Fp2',
    'AF3', 'AF4', 'AF7', 'AF8',
    'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
    'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
    'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
    'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
    'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
    'Oz', 'O1', 'O2'
]

ICA_EXCLUDE_CHANNELS: List[str] = ['Fp1', 'Fp2', 'F7', 'F8', 'AF7', 'AF8']


def process_single_trial(
    trial_path: Union[str, Path], 
    output_directory: Union[str, Path],
    subject_id: Optional[int] = None, 
    session: Optional[int] = None,
    verbose: bool = False
    ) -> None:
    """
    The core pipeline logic to process a single trial.

    Isolated into a function to support both single processing and multiprocessing.

    Can auto detect both subject_id and session if not specified.

    Args:
        trial_path (Union[str, Path]): Path to the EEG data file.
        output_directory (str): Path to the output directory.
        subject_id (int): The ID of the subject.
        verbose (bool): If True, print all the tracing info when running.
                        Only recommended for single processing.
        ---
    """
    if verbose:
        print(f"\n--- Starting processing for subject {subject_id} at {trial_path} ---")

    # Initialization.
    trial_path = Path(trial_path)
    output_directory = Path(output_directory)

    mne.set_log_level("INFO" if verbose else "ERROR")  # Eliminate the mne log printing during filtering and ica.

    logging.basicConfig(
            level=logging.INFO if verbose else logging.ERROR,
            force=True 
        )
    #-------------------------------------------------------------------------------------------

    # Try to parse the subject_id and session if not specified.
    found_sub, found_ses = parse_sub_ses_from_path(trial_path)

    # Assign or compare the subject_id.
    found_subject_id: Optional[int] = int(found_sub.split("-")[1]) if found_sub is not None else None

    if found_subject_id is None and subject_id is None:
        raise ValueError(f"Missing data: 'subject_id' was neither provided in arguments nor found in the path")
    
    elif found_subject_id is not None and subject_id is not None:
        if found_subject_id != subject_id:
            raise ValueError(f"Conflict detected for 'subject_id': "
                             f"Argument provided is '{subject_id:02d}', but path indicated '{found_subject_id:02d}'")
    else:
        subject_id = subject_id if subject_id is not None else found_subject_id

    # Assign or compare the session.
    found_session: Optional[int] = int(found_ses.split("-")[1]) if found_ses is not None else None

    if found_session is None and session is None:
        raise ValueError(f"Missing data: 'session' was neither provided in arguments nor found in the path")
    
    elif found_session is not None and session is not None:
        if found_session != session:
            raise ValueError(f"Conflict detected for 'session': "
                             f"Argument provided is '{session:02d}', but path indicated '{found_session:02d}'")
    else:
        session = session if session is not None else found_session

    output_basename = f"sub-{subject_id:02d}_ses-{session:02d}_task-dailylife_prep_eeg"
    
    logger = PreprocessLogger(
        subject_id=subject_id,
        session=session
    )       
 
    #-----------------------------------------------------------------------------------------------------
    
    # Load raw data and only pick the EEG channels.
    if verbose:
        print("Initialization finished. Begin loading data...")
    load_data_init_time = time.time()

    supported_extensions = ('.bdf', '.edf', '.fif', '.vhdr', '.set')
#     file_candidates: List[Path] = [
#         f for f in trial_path.iterdir()
#         if f.is_file() and f.suffix.lower() in supported_extensions
#     ]
# 
#     if len(file_candidates) == 0:
#         raise FileNotFoundError(
#             f"Error: No supported EEG files found in {trial_path}"
#         )
#     elif len(file_candidates) > 1:
#         raise ValueError(  # Make sure that there are only one data file.
#             f"Error: Multiple EEG files detected in {trial_path}."
#             f"Expected exactly one, but found {len(file_candidates)}"
#         )
# 
#     target_file = file_candidates[0]
#     suffix = target_file.suffix.lower()

    suffix: str = trial_path.suffix

    try:
        if suffix == '.edf':
            raw = mne.io.read_raw_edf(trial_path, preload=True)
        elif suffix == '.bdf':
            raw = mne.io.read_raw_bdf(trial_path, preload=True)  # Another optional argument "verbose=INFO/ERROR".
        elif suffix == '.fif':
            raw = mne.io.read_raw_fif(trial_path, preload=True)
        elif suffix == '.vhdr':
            raw = mne.io.read_raw_brainvision(trial_path, preload=True)
        elif suffix == '.set':
            raw = mne.io.read_raw_eeglab(trial_path, preload=True)
        else:
            raise ValueError(f"Not supported filetype: {suffix}")

    except Exception as e:
        raise ValueError(f"Failed to load the data from {trial_path}: {e}")

    raw.pick_channels(EEG_CHANNELS)  # Pick out from "stand_1020". Prevent later crash.

    logger.log_begin_time(raw.info['meas_date'])

    if verbose:
        print(f"Data loaded successfully, took {time.time()-load_data_init_time:.3f}s\n")

    #-----------------------------------------------------------------------------------------------------------

    if verbose:
        print("Begin data filtering...")

    # Downsampling the raw data to 200 Hz.
    raw.resample(TARGET_SFREQ)
    logger.log_sfreq(raw.info['sfreq'])

    # Filter the data and then cut the edge (3s and 60s).
    raw.filter(l_freq=FILTER_FREQ[0], h_freq=FILTER_FREQ[1])
    raw.notch_filter(freqs=NOTCH_FREQ, notch_widths=NOTCH_WIDTH)

    logger.log_filtering("bandpass", FILTER_FREQ)
    logger.log_filtering("notch", (NOTCH_FREQ, NOTCH_WIDTH)) 

    raw.crop(tmin=CUT_LENGTH[0], tmax=raw.times[-1] - CUT_LENGTH[1])  # Should pass time in secs here.
    
    if verbose:
        print(f"Data filtered and cropped.\n")

    #---------------------------------------------------------------------------

    # 1st bad channels detection and interpolation.
    raw.set_montage('standard_1020') # Set the montage before interpolation.

    if verbose:
        print(f"Begin the 1st bad channels detection and interpolation...")
    init_time_1st: float = time.time()

    # Detect the high noise channels.
    noise_bads = detect_channel_noise(
        raw,
        thresholds=[BAD_CHANNELS_THRESHOLDS["constant_noise"]]
    )
    
    # Detect the flatline error channels.
    flatline_bads = detect_channel_flatline(
        raw,
        min_duration_sec=raw.times[-1] // 10 # TODO, This is my arbitrary choice. To be further discussed.
    )

    # Sort out the bad channels information.
    all_bads = noise_bads + flatline_bads
    unique_bad_dict: Dict[str, Tuple[int, str, str]] = {}

    for b in all_bads:
        if b[1] not in unique_bad_dict:
            unique_bad_dict[b[1]] = b 
        else:
            # Concatenate reasons if both flatline and noise (can be multiple) are detected in one channel.
            existing = unique_bad_dict[b[1]]
            unique_bad_dict[b[1]] = (existing[0], existing[1], f"{existing[2]} | {b[2]}") # Bravo.

    bad_list_1st = list(unique_bad_dict.values()) # Containing all the bad channels details.
    bad_ch_names = list(unique_bad_dict.keys())

    if len(bad_ch_names) > INTERPOLATION_THRESHOLD:
        save_as_fif(
            raw, 
            out_dir=output_directory, 
            filename=f"sub-{subject_id:02d}_ses-{session:02d}_task-dailylife_prep_abnormal_eeg"
        )

        logger.log_data_length(raw)
        logger.log_error_trial(
            error=f"'{len(bad_ch_names)}' bad channels are detected before the 1st interpolation. "
                  f"Exceeding the threshold '{INTERPOLATION_THRESHOLD}'",
            details=bad_list_1st
        )
        logger.save_log(
            output_dir=output_directory,
            filename=f"sub-{subject_id:02d}_ses-{session:02d}_task-dailylife_prep_abnormal_eeg")

        # Exit the preprocess pipeline here if too many whole-length bad channels.
        raise ValueError(f"Abnormal trial: subject-{subject_id}_session-{session} can't be interpolated. "
                         f"{len(bad_ch_names)} bad channels are detected before the 1st interpolatioin, "
                         f"exceeding the threshold '{INTERPOLATION_THRESHOLD}'")
    
    else:
        raw.info['bads'] = bad_ch_names
        raw.interpolate_bads(reset_bads=True)

        logger.log_bad_channels("1st", bad_list_1st)

    if verbose:
        print(f"The 1st bad channels detection and interpolation has been finished. Took: {time.time()-init_time_1st:.3f}s\n")

    #-----------------------------------------------------------------------------------------

    # 2nd bad channel detection and interpolation.
    if verbose:
        print(f"Begin the 2nd bad channels detection and interpolation...")
    init_time_2nd = time.time()

    raw, bad_dict_2_nd = process_local_bad_channels(
        raw=raw, 
        chunk_sec=60.0, 
        detection_thresholds=[BAD_CHANNELS_THRESHOLDS['constant_noise'], BAD_CHANNELS_THRESHOLDS["abrupt_noise"]], 
        interpolation_threshold=INTERPOLATION_THRESHOLD,
        exclude_channels=ICA_EXCLUDE_CHANNELS
    )
    logger.log_bad_channels("2nd", bad_dict_2_nd)

    #---------------------------------------------------------------------------------------------------------

    # Apply ICA to remove artifacts and log the rejected components.
    if verbose:
        print(f"The 2nd bad channels processed. Took {time.time()-init_time_2nd:.3f}s.\n")
        print("Begin ICA...")
    init_time_ica = time.time()

    try:
        raw, full_list, exc_list, total_count, exc_count = apply_ica_denoise(
            raw, 
            prob_threshold=ICA_PROB_THRESHOLD
            )
        logger.log_ica_removal(  # It seems that this part won't be logged if error happens in during ICA...
            full_list=full_list, 
            exclude_list=exc_list, 
            total_count=total_count, 
            exclude_count=exc_count
            )
    # Exit pipeline if ICA has occurred, then mark this trial as abnormal.
    except Exception as e:  
        save_as_fif(
            raw, 
            out_dir=output_directory, 
            filename=f"sub-{subject_id:02d}_ses-{session:02d}_task-dailylife_prep_abnormal_eeg"
        )

        logger.log_data_length(raw)
        logger.log_error_trial(
            error=f"ICA fitting failed due to extreme data corruption or singular matrix: {e}",
            details=None  # TODO Can be prioritezed here, theoretically...
        )
        logger.save_log(
            output_dir=output_directory,
            filename=f"sub-{subject_id:02d}_ses-{session:02d}_task-dailylife_prep_abnormal_eeg")

        raise ValueError(f"Abnormal trial: subject-{subject_id}_session-{session} crashing during ICA. "
                         f"Reason {e}")

    #------------------------------------------------------------------------------------------

    # 3rd bad channels detection and interpolation.
    if verbose:
        print(f"ICA completed, took {time.time()-init_time_ica:.3f}s.\n")
        print("Begin the 3rd bad channel processing...")
    init_time_3rd = time.time()

    raw, bads_3rd = process_local_bad_channels(
        raw=raw, 
        chunk_sec=60.0, 
        detection_thresholds=[BAD_CHANNELS_THRESHOLDS["constant_noise"], BAD_CHANNELS_THRESHOLDS["abrupt_noise"]],
        interpolation_threshold=INTERPOLATION_THRESHOLD,
        exclude_channels=[]  # Empty list, check all channels
    )
    logger.log_bad_channels("3rd", bads_3rd)
    if verbose:
        print(f"The 3rd bad channels detection and interpolation finished, took {time.time()-init_time_3rd:.3f}")

    # Do the second re-referencing, using "average".
    raw.set_eeg_reference('average') # montage has already been set in apply_ica_denoise().
    logger.log_re_reference("average")
    if verbose:
        print("Data re-referenced, using 'average'")

    # Save the outputs to the expected places.
    logger.log_data_length(raw)

    save_as_fif(raw, output_directory, output_basename)
    # save_as_brainvision(raw, output_directory, output_basename)
    logger.save_log(output_dir=output_directory, filename=output_basename)
    
    if verbose:
        print(f"Subject-{subject_id}_session-{session} processing complete, saved to {output_directory}.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse the arguments for EEG data preprocess")

    parser.add_argument("--data_path", type=str, required=True, help="Path to the .bdf file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")

    parser.add_argument("--subject_id", type=int, choices=range(1, 16), help="Subject ID, indexing from 1 to 15")
    parser.add_argument("--session", type=int, choices=range(1, 12), help="Session number, indexing from 1 to 11")

    parser.add_argument("--verbose", action="store_true", help="If flagged, print the detailed processing logging")

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for single-process execution.
    """
    init_time: float = time.time() 
    args = parse_args()

    # Print the information from logging if verbose.
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(message)s" # Keep the format the same as "print".
    )

    # Trigger the isolated pipeline logic.
    process_single_trial(
        args.data_path, 
        args.output_dir, 
        args.subject_id, 
        args.session,
        verbose=args.verbose
        )
    
    if args.verbose:
        print(f"\nAll tasks done. Total execution time: {time.time() - init_time:.2f}s")


if __name__ == "__main__":
    main()