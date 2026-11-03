"""
The helper functions utilized by the preprocess pipeline.

Author: ddh

Date: July 2, 2026
"""

import os
import logging
import warnings
import re
from typing import Dict, List, Optional, Union, Tuple, Iterator, Any
from pathlib import Path

import mne 
import mne_icalabel
import numpy as np


def parse_sub_ses_from_path(file_path: Union[str, Path]) -> List[Tuple[Union[str, None], Union[str, None]]]:
    """
    Parse the "sub-xx" and "ses-xx" from the given path. Return them separately.

    Return None if they were not found.
    """
    file_path = Path(file_path)

    sub_pattern: re.Pattern[str] = re.compile(r"sub-\d{2}(?!\d)")  # (?!\d) is a "negative lookahead" asserting no more digits follow...
    ses_pattern: re.Pattern[str] = re.compile(r"ses-\d{2}(?!\d)")

    found_sub: Optional[str] = None
    found_ses: Optional[str] = None 

    path_parts_reversed: Iterator[str] = reversed(file_path.parts)  # Begin to traverse from filename.

    # Find the first met sub and ses.
    for part in path_parts_reversed:
        if found_sub is None:
            match_sub: Optional[re.Match[str]] = sub_pattern.search(part)
            if match_sub:
                found_sub = match_sub.group()

        if found_ses is None:
            match_ses: Optional[re.Match[str]] = ses_pattern.search(part)
            if match_ses:
                found_ses = match_ses.group()
        
        # Break the searching loop as long as all of them are found.
        if found_sub is not None and found_ses is not None:
            break

    return found_sub, found_ses


def detect_channel_noise(
    raw: mne.io.BaseRaw, 
    thresholds: List[Tuple[float, float]],
    ch_names: Optional[List[str]] = None,
    global_medians: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, str, str]]:
    """
    The function that detects channels noises using tiered median limits and median-based thresholds.

    Supports short-segments utilization, by passing pre-calculated median values from the larger whole
    length. 

    Will dynamically calculate the medians if not provided.

    Args:
        raw (mne.io.BaseRaw): The object that corresponds to the data to be detected.
        thresholds (List[Tuple[float, float]]): A list containing (multiplier, ratio).
        channel_names (Optional[List[str]]): Channel names to check. Default to use all EEG channels from the raw object.
        global_medians (Optional[np.ndarray]): A pre-calculated 1D array of median absolute values for each channel (e.g. from the entire recording).

    Returns:
        List[Tuple[int, str, str]]: A list containing (ch_idx, ch_name, brief_reason).  
                                    Duplicate channels may appear due to different reasons.
    ---
    """
    data: np.ndarray = raw.get_data(picks="eeg")
    total_samples: int = data.shape[1] 

    if ch_names is None:
        #     channel_names = [ # The EEG channels for 10-20 system. Total 59.
        #     'Fpz', 'Fp1', 'Fp2',
        #     'AF3', 'AF4', 'AF7', 'AF8',
        #     'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
        #     'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
        #     'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
        #     'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
        #     'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
        #     'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
        #     'Oz', 'O1', 'O2'
        # ]
        # print(f"No channel_names input was detected. Using default xxx system to index")
        eeg_indices: np.ndarray = mne.pick_types(raw.info, eeg=True, meg=False) # Pick the indices for EEG channels only.
        ch_names: List[str] = [raw.ch_names[idx] for idx in eeg_indices]

    if global_medians is None:
        logging.info("No global_medians provided, calculating local medians dynamically...")
        global_medians: np.ndarray = np.median(np.abs(data), axis=1)

    bad_channels: List[Tuple[int, str, str]] = []

    for ch_idx, ch_data in enumerate(data):
        # Check and tier the median value limit.
        if global_medians[ch_idx] > 1.0: # Unit is Volt here.
            bad_channels.append((ch_idx, ch_names[ch_idx], "abs_median_>1V_error"))
            continue # Jump to next loop directly.

        elif global_medians[ch_idx] > 1e-3: # The absolute value of EOC artifact can reach up to 1mV?
            bad_channels.append((ch_idx, ch_names[ch_idx], "abs_median_>1mV_error"))
            continue

        for multiplier, ratio in thresholds:
            high_values_count: int = np.sum(np.abs(ch_data) > global_medians[ch_idx] * multiplier)
            high_values_ratio: float = high_values_count / total_samples

            if high_values_ratio > ratio:
                bad_channels.append(
                    (ch_idx, ch_names[ch_idx], f"mult-{multiplier}_ratio-{ratio}")
                )

    return bad_channels


def detect_channel_flatline(
    raw: mne.io.BaseRaw,
    min_duration_sec: float = 1.0,
    zero_tolerance: float = 1e-8,
    ch_names: Optional[List[str]] = None,
) -> List[Tuple[int, str, str]]:
    """
    Detect the channels that exhibit continous flatline error using vectorized array dffing.
    
    Args:
        raw (mne.io.BaseRaw): The raw object corresponding to the EEG data.
        min_duration_sec (float): The minimum tolerance threshold in secs for a "continous flatline time".
        zero_tolerance (float): The absulote minimum threshold for a signal to be considered as "flat".
        ch_names (Optional[List[str]]): The selected channels to be detected. Default to the
                                        whole channels of the raw object.

    Returns:
        List[Tuple[int, str, str]]: Return the error messages of the detected channels.
            - Index of the channel.
            - Corresponding channel name.
            - The error message explaining the details of the (longest) flatline segment.
    """
    data: np.ndarray = raw.get_data(picks='eeg')
    sfreq = float(raw.info['sfreq'])
    min_samples = int(min_duration_sec * sfreq)

    if ch_names is None:
        eeg_indices: np.ndarray = mne.pick_types(raw.info, eeg=True, meg=False)
        ch_names = [raw.ch_names[idx] for idx in eeg_indices]

    bad_channels: List[Tuple[int, str, str]] = []

    # Detect the flatlines using vectorized method.
    for ch_idx, ch_data in enumerate(data):
        # Calulate absolute voltage diffs of the adjacent points.
        diffs = np.abs(np.diff(ch_data))  

        # Generate the "bool mask". To see if there is change.
        is_flat: np.ndarray = diffs <= zero_tolerance  

        # Padding the two-side terminals to enable boundry detection.
        padded_mask = np.concatenate(([False], is_flat, [False]))

        # Find the boundry points (segments essentially) of flatlines.
        runs = np.diff(padded_mask.astype(int))  # ".astype()" will forcefully convert the date into appointed format.
                                                 # Will convert bool into interger here. (True->1, False->0).
        starts = np.where(runs == 1)[0]  # "np.where()" has two types of usage. np.where(condition, x, y) and np.where(condition).
                                         # Will return all the indice of the "starting points" here. 
                                         # [0] is mandatory since np.where() will wrap a tuple shell outside...
        ends = np.where(runs == -1)[0]  # I finally realized that is this ace logic... The states of ALL points (in, keep, out) will
                                        # be traced/calculated accurately.

        # Record the information of the flatlines.
        longest_flat_samples = 0
        en_of_longest = 0
        st_of_longest = 0

        voltage_val = 0.0

        for st, en in zip(starts, ends):
            if (en - st) > longest_flat_samples:
                longest_flat_samples = int(en - st)
                en_of_longest = en
                st_of_longest = st

                voltage_val = float(ch_data[st]) # Correct and gogeous logic here.
                                                 # Now the st (index here, can be 0) is both the "segment idx" and "origianl LEFT endpoint idx".

        if longest_flat_samples >= min_samples:
            flat_secs: float = longest_flat_samples / sfreq
            st_time: float = raw.times[st_of_longest]
            en_time: float = raw.times[en_of_longest]

            error_str = f"flatline(longest)_{flat_secs:.2f}s_st{st_time:.2f}s-en{en_time:.2f}s_at_{voltage_val*1e6:.2e}uV"  # Use scientific notation.
                                                                                                                            # Keep two decimal digits.
            bad_channels.append(
                (ch_idx, ch_names[ch_idx], error_str)  
            )

    return bad_channels


def process_local_bad_channels(
    raw: mne.io.BaseRaw,
    chunk_sec: float,
    detection_thresholds: List[Tuple[float, float]],
    interpolation_threshold: int,
    exclude_channels: Optional[List[str]] = None
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """
    Split the raw data into fixed-length chunks, detect bad channels locally using
    global median anchors, then apply interpolation or drop the chunk via annotations.

    Will call "detect_channel_noise()" and "detect_channel_flatline()".

    Args:
        raw (mne.io.BaseRaw): The raw object to process.
        chunk_sec (float): Duration of each chunk in seconds (e.g., 60.0).
        detection_thresholds (List[Tuple[float, float]]): The detection thresholds for 
                                                          detect_bad_channels().
        interpolation_threshold (int): Maximum allowed bad channels before aborting interpolation.
        exclude_channels (List[str]): Channels to explicitly ignore during detection.

    Returns:
        Tuple[mne.io.BaseRaw, Dict[str, Any]]: 
            - The reassembled, cleaned Raw object.
            - A dictionary containing log info per chunk (e.g., {"segment_1": [...]}).
    """
    exclude_channels = exclude_channels or []
    duration = raw.times[-1]  # Get the time stamp of the final sample point, in seconds.
    
    # Calculate the global medians as anchors for all chunks.
    global_data = raw.get_data(picks='eeg')
    global_medians = np.median(np.abs(global_data), axis=1)
    
    processed_chunks = []
    log_dict: Dict[str, Any] = {}

    # Collect the bad segment info uniformly. Prevent duplicate annotating later.
    bad_onsets: List[float] = []
    bad_durations: List[float] = []
    bad_descriptions: List[str] = []

    # Define the boundaries for 1-minute chunks.
    starts: np.ndarray = np.arange(0, duration, chunk_sec)  # "arange"="array range". 
                                                            # (start, stop[not included], step).
    
    for i, st in enumerate(starts):
        # Prevent the last tiny segment from crashing if less than 1 second.
        end: float = min(st + chunk_sec, duration)
        if (end - st) < 1.0:
            break  # Exit the loop.
            
        # Safely crop a piece of continuous raw data.
        chunk = raw.copy().crop(tmin=st, tmax=end, include_tmax=False)
        
        # 1. Detect the amplitude noises.
        noise_bads: List[Tuple[int, str, str]] = detect_channel_noise(
            raw=chunk,
            thresholds=detection_thresholds,
            global_medians=global_medians
        )
        
        # 2. Detect flatlines.
        flat_bads: List[Tuple[int, str, str]] = detect_channel_flatline(
            raw=chunk,
            min_duration_sec=1.0,  # TODO to be discusses.
            zero_tolerance=1e-8
        )

        # Sort out the bad channels information.
        kept_noise_bads = [b for b in noise_bads if b[1] not in exclude_channels]
        all_bads = kept_noise_bads + flat_bads

        unique_bad_dict: Dict[str, Tuple[int, str, str]] = {}
        for b in all_bads:
            if b[1] not in unique_bad_dict:
                unique_bad_dict[b[1]] = b 
            else:
                # Concatenate reasons if both flatline and noise (can be multiple) are detected in one channel.
                existing = unique_bad_dict[b[1]]
                unique_bad_dict[b[1]] = (existing[0], existing[1], f"{existing[2]} | {b[2]}") # Bravo.

        unique_bad_list = list(unique_bad_dict.values()) # Containing all the bad channels details.
        bad_ch_names = list(unique_bad_dict.keys())

        # Decide whether to interpolate the bad chs or annotate them.
        if len(bad_ch_names) > interpolation_threshold:
            # Abort the interpolatioin and mark this seg as BAD for ICA.
            logging.warning(f"Chunk {i+1} minu has {len(bad_ch_names)} bad channels. Interpolation aborted.")

            bad_onsets.append(float(st))
            bad_durations.append(float(end - st))
            bad_descriptions.append("BAD_severe_noise_collapse")

            log_dict[f"minute_{i+1:03d}"] = {
                "status": "interpolation_aborted",
                "bad_count": len(bad_ch_names),
                "details": unique_bad_list
            }

        elif bad_ch_names:
            chunk.info['bads'] = bad_ch_names
            # chunk.interpolate_bads(reset_bads=True, exclude_channels=exclude_channels)  # Alternative.
            chunk.interpolate_bads(reset_bads=True)

            log_dict[f"minute_{i+1:03d}"] = {
                "status": "interpolated",
                "bad_count": len(bad_ch_names),
                "details": unique_bad_list
            }

        else:
            log_dict[f"minute_{i+1:03d}"] = {
                "status": "clean",
                "bad_count": 0
            }

        processed_chunks.append(chunk)

    # Reassemble all chunks back into one continuous raw object
    raw_assembled = mne.concatenate_raws(processed_chunks)

    # Attach the labels to the final data, prevent duplicate annotations.
    if bad_onsets:
        bad_annot = mne.Annotations(
            onset=bad_onsets,
            duration=bad_durations,
            description=bad_descriptions
        )
        raw_assembled.set_annotations(bad_annot)
    
    return raw_assembled, log_dict


def apply_ica_denoise(
    raw: mne.io.BaseRaw,
    prob_threshold: float = 0.8
) -> Tuple[mne.io.BaseRaw, List[Tuple[int, str, float]], List[Tuple[int, str, float]], int, int]:
    """
    Apply (PCA and) Independent Component Analysis to the raw object that
    contains EEG data to remove artifacts. 
    Then collect and return all components' details, total count and the removed 
    artifact component details(including index, name, confidence) and total count.

    Args:
        raw (mne.io.BaseRaw): The raw object contains data to be processed..
        prob_threshold (float): Between 0 and 1.
                            The confidence threshold for non-brain's artifacts' rejection.
                        
    Returns:
        Tuple[mne.io.BaseRaw, List[Tuple[int, str, float]], List[Tuple[int, str, float]], int, int]: 
                - The cleaned data object.
                - The list containing (index, name, confidence) for all the components.
                - The list containing (index, name, confidence) for each rejected component.
                - The total count of all components.
                - The total count of the rejected components.                                       
    ---
    """
    # Standardize spatial location and reference.
    raw.set_montage('standard_1020') # Set 3-dimensional coordinates to the channel (names),
                                    # using standard 10-20 system.

    raw.set_eeg_reference('average') # Re-referencing, eliminate the "common-mode noises".

    # # Calculate the number of PCA components to retain. # This part of logic might be verified...
    # num_components: int = int(np.floor(len(raw.ch_names) * components_ratio)) # "raw.ch_names" is shortcut to "raw.info.ch_names".

    # Initialize the ICA object.
    ica = mne.preprocessing.ICA(
        # n_components=num_components, # So it can implement both PCA and ICA.
        n_components=None, # Mandatory here, can't be set to 59 manually.
                           # After re-referencing, the rank will lower to 58. So there will be one 0 eigenvalue. 
                           # Then there wil be problem to do PCA whitening (devide by eigenvalues).
        random_state=97, # For reproducibility. "97" is in mne's documentation.
        max_iter='auto',
        method='infomax', # Use the extended infomax algorithm.
        fit_params=dict(extended=True) # Specify the infomax details, use "extended infomax". Best practice.
    )

    # Fit the ICA model.
    with warnings.catch_warnings(): 
        warnings.filterwarnings( # Use warnings module to capture the obsessive warnings sent by mne (due to ICA).
            "ignore",
            category=RuntimeWarning,
            message=r".*has not been high-pass filtered.*", # ".*" is RegEx of Python. "."=any_symples, "*"=repeat_any_times.
                                                            # The dependent re package is already inside warning package.
            module=r"mne.*"
        )
        ica.fit(raw) # Time consuming! 
                     # "raw" will not be modified/written here.
                     # The segments with "BAD" annotation will be ignored here.

    # Run ICLabel to classify the components.
    with warnings.catch_warnings(): 
        warnings.filterwarnings(
            "ignore",
            message=r"The provided raw instance is not filtered between 1 and 100 Hz.*",
            category=RuntimeWarning
        )
        component_dict = mne_icalabel.label_components(raw, ica, method='iclabel') # Pure Python dict.
                                                                                   # "brain" (or "eye blink", "other", etc.) may appear multiple times.
                                                                                   # Is this also time consuming? If so, how? TODO
                                                                                
    iclabel_labels: List[str] = component_dict['labels']
    iclabel_probs: List[float] = component_dict['y_pred_proba']

    # Identify and record the bad components.
    exclude_idxs: List[int] = []
    for idx, (label, prob) in enumerate(zip(iclabel_labels, iclabel_probs)):
        if label not in ["brain", "other"] and prob > prob_threshold:
            exclude_idxs.append(idx)

    # Copy raw data and reconstruct a cleaned data.
    raw_clean: mne.io.BaseRaw = ica.apply(raw.copy(), exclude=exclude_idxs)

    # Print results. No, "logging" results, compatible for multiprocessing.
    # print(f"{len(exclude_idxs)} artifact components were detected\n")
    logging.info(f"{len(exclude_idxs)} artifact components were detected\n")

    for idx in exclude_idxs:
        # print(f"Excluded: idx {idx:02d} | name {iclabel_labels[idx]:<10} | confidence {iclabel_probs[idx]:.4f}")
        logging.info(f"Excluded: idx {idx:02d} | name {iclabel_labels[idx]:<10} | confidence {iclabel_probs[idx]:.4f}")

    exclude_list = [
        (i, iclabel_labels[i], iclabel_probs[i]) for i in exclude_idxs
        ]
    full_list = [
        (i, iclabel_labels[i], iclabel_probs[i]) for i in range(len(iclabel_labels))
        ]
    return raw_clean, full_list, exclude_list, len(iclabel_labels), len(exclude_idxs)


def save_as_fif(raw: mne.io.BaseRaw, out_dir: Union[str, Path], filename: str) -> None:
    """Save raw data to MNE's standard .fif format."""
    os.makedirs(out_dir, exist_ok=True)

    out_path = Path(out_dir) / f"{Path(filename).stem}.fif"
    raw.save(out_path, overwrite=True)
    # print(f"Data saved to {out_path}")
    logging.info(f"Data saved to {out_path}")


def save_as_brainvision(raw: mne.io.BaseRaw, out_dir: Union[str, Path], filename: str) -> None:
    """Save raw data to BIDS-compliant BrainVision format (.vhdr, .vmrk, .eeg)."""
    os.makedirs(out_dir, exist_ok=True)

    out_path = Path(out_dir) / f"{Path(filename).stem}.fif"

    # MNE export functionality requires the generic string representation of the file.
    mne.export.export_raw(str(out_path), raw, fmt='brainvision', overwrite=True)
    # print(f"Data saved to BrainVision format at {out_dir}")
    logging.info(f"Data saved to BrainVision format at {out_dir}")