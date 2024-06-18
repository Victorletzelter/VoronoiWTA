import hashlib
import json
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft
from torch.utils.data import Dataset
from typing import Tuple
import h5py
import re

# File copied and adapted from (https://github.com/chrschy/adrenaline/blob/master/data_handlers/tut_sound_events.py) [A]
#
# [A] Schymura, C., Ochiai, T., Delcroix, M., Kinoshita, K., Nakatani, T., Araki, S., & Kolossa, D. (2021, January). Exploiting attention-based sequence-to-sequence architectures for sound event localization.
# In 2020 28th European Signal Processing Conference (EUSIPCO) (pp. 231-235). IEEE.


def classes_to_int(dataset, class_str, sigma_classes=[5 + 5 * i for i in range(11)]):
    if dataset == "ansim":
        classes = [
            "speech",
            "phone",
            "keyboard",
            "doorslam",
            "laughter",
            "keysDrop",
            "pageturn",
            "drawer",
            "cough",
            "clearthroat",
            "knock",
        ]
        idx = classes.index(class_str)
        return idx, sigma_classes[idx]


def extract_number(s):
    s = s.split("/")[-1]
    # Use regular expression to find sequences of digits separated by dashes
    matches = re.findall(r"\d+", s)
    # Return the second number in the sequence if it exists
    if len(matches) >= 2:
        return matches[1]  # Return the second number
    return None  # In case the string does not match the expected format


def noise_labels(events_in_chunk, sigma=10):
    """Add noise to the labels of the events in a chunk of data."""
    for row in events_in_chunk.itertuples():
        elevation = row.elevation
        distance = row.distance
        events_in_chunk.at[row.Index, "elevation"] += np.random.normal(
            loc=0.0, scale=(sigma) / (distance)
        )
        events_in_chunk.at[row.Index, "azimuth"] += np.random.normal(
            loc=0.0, scale=(sigma) / ((distance * np.cos(np.deg2rad(elevation))))
        )
    return events_in_chunk


def sph2cart(az, el, r):
    """
    Converts spherical coordinates given by azimuthal, elevation and radius to cartesian coordinates of x, y and z

    :param az: azimuth angle
    :param el: elevation angle
    :param r: radius
    :return: cartesian coordinate
    """
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def scaled_cross_product(a, b):
    ab = np.dot(a, b)
    if ab > 1 or ab < -1:
        return [999]

    acos_ab = np.arccos(ab)
    x = np.cross(a, b)
    if acos_ab == np.pi or acos_ab == 0 or sum(x) == 0:
        return [999]
    else:
        return x / np.sqrt(np.sum(x**2))


def cart2sph(x, y, z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)  # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))  # theta
    az = np.arctan2(y, x)  # phi
    return az, elev, r


def rotate_matrix_vec_ang(_rot_vec, theta):
    u_x_u = np.array(
        [
            [_rot_vec[0] ** 2, _rot_vec[0] * _rot_vec[1], _rot_vec[0] * _rot_vec[2]],
            [_rot_vec[1] * _rot_vec[0], _rot_vec[1] ** 2, _rot_vec[1] * _rot_vec[2]],
            [_rot_vec[2] * _rot_vec[0], _rot_vec[2] * _rot_vec[1], _rot_vec[2] ** 2],
        ]
    )

    u_x = np.array(
        [
            [0, -_rot_vec[2], _rot_vec[1]],
            [_rot_vec[2], 0, -_rot_vec[0]],
            [-_rot_vec[1], _rot_vec[0], 0],
        ]
    )
    return np.eye(3) * np.cos(theta) + np.sin(theta) * u_x + (1 - np.cos(theta)) * u_x_u


class TUTSoundEvents(Dataset):
    """
    This class enables using a subset of the Tampere University of Technology Sound Events datasets. For more
    detailed information about these datasets, please refer to

        https://github.com/sharathadavanne/seld-net

    The following subsets are currently supported by this class:

        - ANSIM: Ambisonic, Anechoic and Synthetic Impulse Response Dataset (https://doi.org/10.5281/zenodo.1237703)
        - RESIM: Ambisonic, Reverberant and Synthetic Impulse Response Dataset (https://doi.org/10.5281/zenodo.1237707)
        - REAL: Ambisonic, Reverberant and Real-life Impulse Response Dataset (https://doi.org/10.5281/zenodo.1237793)

    Please run the script download_data.sh first to download these subsets from the corresponding repositories.
    """

    def __init__(
        self,
        root: str,
        tmp_dir,  #: str = './tmp',
        split,  #: str = 'train',
        test_fold_idx,  #: int = 1,
        sequence_duration,  #: float = 30.,
        chunk_length,  #: float = 2.,
        frame_length,  #: float = 0.04,
        num_fft_bins,  #: int = 2048,
        max_num_sources,  #: int = 4,
        num_overlapping_sources,  #: int = None,
        noisy_version,  #: bool = False,
        sigma_classes,  #: list = [5+5*i for i in range(11)],
        offline,  #: bool = True,
        **kwargs
    ) -> None:
        """Class constructor.

        :param root: path to root directory of the desired subset
        :param split: choose between 'train' (default), 'valid' and 'test'
        :param test_fold_idx: cross-validation index used for testing; choose between 1, 2, and 3
        :param sequence_duration: fixed duration of each audio signal in seconds, which is set to 30s by default
        :param num_fft_bins: number of FFT bins
        """
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        self.tmp_dir = tmp_dir

        if split not in ["train", "valid", "test"]:
            raise RuntimeError(
                "Split must be specified as either train, valid or test."
            )

        if (test_fold_idx < 1) or (test_fold_idx > 3):
            raise RuntimeError(
                "The desired test fold index must be specified as either 1, 2 or 3."
            )

        self.split = split
        self.test_fold_idx = test_fold_idx
        self.sequence_duration = sequence_duration
        self.chunk_length = chunk_length
        self.num_chunks_per_sequence = int(self.sequence_duration / self.chunk_length)
        self.frame_length = frame_length
        self.num_fft_bins = num_fft_bins
        self.max_num_sources = max_num_sources
        self.noisy_version = noisy_version
        self.sigma_classes = sigma_classes
        self.offline = offline

        self.dataset_name = root.split("/")[-2].lower()

        if "ansim" in self.dataset_name:
            self.num_unique_classes = 11
        else:
            # raise error, dataset not supported
            raise NotImplementedError

        # Assemble table containing paths to all audio and annotation files.
        self.sequences = {}

        for audio_subfolder in os.listdir(root):
            if os.path.isdir(
                os.path.join(root, audio_subfolder)
            ) and audio_subfolder.startswith("wav"):
                annotation_subfolder = "desc" + audio_subfolder[3:-5]

                if num_overlapping_sources is not None:
                    if num_overlapping_sources != int(
                        annotation_subfolder[annotation_subfolder.find("ov") + 2]
                    ):
                        continue

                fold_idx = int(
                    annotation_subfolder[annotation_subfolder.find("split") + 5]
                )

                for file in os.listdir(os.path.join(root, audio_subfolder)):
                    file_prefix, extension = os.path.splitext(file)

                    if extension == ".wav":
                        path_to_audio_file = os.path.join(root, audio_subfolder, file)
                        path_to_annotation_file = os.path.join(
                            root, annotation_subfolder, file_prefix + ".csv"
                        )
                        is_train_file = file_prefix.startswith("train")

                        # Check all three possible cases where files will be added to the global file list
                        if (
                            (split == "train")
                            and (fold_idx != test_fold_idx)
                            and is_train_file
                        ):
                            self._append_sequence(
                                path_to_audio_file,
                                path_to_annotation_file,
                                is_train_file,
                                fold_idx,
                                num_overlapping_sources,
                            )
                        elif (
                            (split == "valid")
                            and (fold_idx != test_fold_idx)
                            and not is_train_file
                        ):
                            self._append_sequence(
                                path_to_audio_file,
                                path_to_annotation_file,
                                is_train_file,
                                fold_idx,
                                num_overlapping_sources,
                            )
                        elif (split == "test") and (fold_idx == test_fold_idx):
                            self._append_sequence(
                                path_to_audio_file,
                                path_to_annotation_file,
                                is_train_file,
                                fold_idx,
                                num_overlapping_sources,
                            )

    def _append_sequence(
        self,
        audio_file: str,
        annotation_file: str,
        is_train_file: bool,
        fold_idx: int,
        num_overlapping_sources: int,
    ) -> None:
        """Appends sequence (audio and annotation file) to global list of sequences.

        :param audio_file: path to audio file
        :param annotation_file: path to corresponding annotation file in *.csv-format
        :param is_train_file: flag indicating if file is used for training
        :param fold_idx: cross-validation fold index of current file
        :param num_overlapping_sources: number of overlapping sources in the dataset
        """
        for chunk_idx in range(self.num_chunks_per_sequence):
            sequence_idx = len(self.sequences)

            start_time = chunk_idx * self.chunk_length
            end_time = start_time + self.chunk_length

            self.sequences[sequence_idx] = {
                "audio_file": audio_file,
                "annotation_file": annotation_file,
                "is_train_file": is_train_file,
                "cv_fold_idx": fold_idx,
                "chunk_idx": chunk_idx,
                "start_time": start_time,
                "end_time": end_time,
                "num_overlapping_sources": num_overlapping_sources,
            }

    def _get_audio_features(
        self, audio_file: str, start_time: float = None, end_time: float = None
    ) -> np.ndarray:
        """Returns magnitude and phase of the multi-channel spectrogram for a given audio file.

        :param audio_file: path to audio file
        :param start_time: start time of the desired chunk in seconds
        :param end_time: end time of the desired chunk in seconds
        :return: magnitude, phase and sampling rate in Hz
        """
        sampling_rate, audio_data = wavfile.read(audio_file)
        num_samples, num_channels = audio_data.shape

        required_num_samples = int(sampling_rate * self.sequence_duration)

        # Perform zero-padding (if required) or truncate signal if it exceeds the desired duration.
        if num_samples < required_num_samples:
            audio_data = np.pad(
                audio_data,
                ((0, required_num_samples - num_samples), (0, 0)),
                mode="constant",
            )
        elif num_samples > required_num_samples:
            audio_data = audio_data[:required_num_samples, :]

        # Normalize and crop waveform
        start_time_samples = int(start_time * sampling_rate)
        end_time_samples = int(end_time * sampling_rate)

        waveform = audio_data[start_time_samples:end_time_samples, :]
        waveform = waveform / np.iinfo(waveform.dtype).max

        # Compute multi-channel STFT and remove first coefficient and last frame
        frame_length_samples = int(self.frame_length * sampling_rate)
        spectrogram = stft(
            waveform,
            fs=sampling_rate,
            nperseg=frame_length_samples,
            nfft=self.num_fft_bins,
            axis=0,
        )[-1]
        spectrogram = spectrogram[1:, :, :-1]
        spectrogram = np.transpose(spectrogram, [1, 2, 0])

        # Compose output tensor as concatenated magnitude and phase spectra
        audio_features = np.concatenate(
            (np.abs(spectrogram), np.angle(spectrogram)), axis=0
        )

        return audio_features.astype(np.float16)

    def _get_targets(
        self, annotation_file: str, chunk_start_time: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a polar map of directions-of-arrival (azimuth and elevation) from a given annotation file.

        :param annotation_file: path to annotation file
        :param chunk_start_time: start time of the desired chunk in seconds
        :return: two-dimensional DoA map
        """

        annotations = pd.read_csv(
            annotation_file,
            header=0,
            names=[
                "sound_event_recording",
                "start_time",
                "end_time",
                "elevation",
                "azimuth",
                "distance",
            ],
        )
        annotations = annotations.sort_values("start_time")

        event_start_time = annotations["start_time"].to_numpy()
        event_end_time = annotations["end_time"].to_numpy()

        num_frames_per_chunk = int(
            2 * self.chunk_length / self.frame_length
        )  # The number of "frames" per chunk (50% overlap)

        source_activity = np.zeros(
            (num_frames_per_chunk, self.max_num_sources), dtype=np.uint8
        )
        direction_of_arrival = np.zeros(
            (num_frames_per_chunk, self.max_num_sources, 2), dtype=np.float32
        )
        distances = np.zeros(
            (num_frames_per_chunk, self.max_num_sources), dtype=np.float32
        )

        direction_of_arrival_classes = np.zeros(
            (num_frames_per_chunk, self.max_num_sources, self.num_unique_classes, 2),
            dtype=np.float32,
        )
        source_activity_classes = np.zeros(
            (num_frames_per_chunk, self.max_num_sources, self.num_unique_classes),
            dtype=np.uint8,
        )

        for frame_idx in range(num_frames_per_chunk):
            frame_start_time = chunk_start_time + frame_idx * (
                self.frame_length / 2
            )  # Compute the start time of the frame
            frame_end_time = frame_start_time + (
                self.frame_length / 2
            )  # Compute the end time of the frame (frame with 50% overlap)

            event_mask = (
                event_start_time <= frame_start_time
            )  # The event has started before the frame
            event_mask = event_mask | (
                (event_start_time >= frame_start_time)
                & (event_start_time < frame_end_time)
            )  # The event has started before the frame or during the frame
            event_mask = event_mask & (
                event_end_time > frame_start_time
            )  # The event has not ended before the frame

            events_in_chunk = annotations[
                event_mask
            ]  # Get the events that are in the frame
            num_active_sources = len(
                events_in_chunk
            )  # Get the number of active sources in the frame (i.e., the number of events in the frame)

            if num_active_sources > 0:  # If there are active sources in the frame
                source_activity[frame_idx, :num_active_sources] = (
                    1  # Set the source activity to 1
                )

                if self.noisy_version is True and self.offline is True:
                    events_in_chunk = self.perclass_noise_labels(
                        events_in_chunk,
                        sigma_classes=self.sigma_classes,
                        dataset_name=self.dataset_name,
                    )

                direction_of_arrival[frame_idx, :num_active_sources, :] = np.deg2rad(
                    events_in_chunk[["azimuth", "elevation"]].to_numpy()
                )  # Extract the DOAs of the active sources
                distances[frame_idx, :num_active_sources] = events_in_chunk[
                    "distance"
                ].to_numpy()  # Extract the distances of the active sources

                ################### classes information extraction
                class_active_events = list(
                    events_in_chunk["sound_event_recording"].apply(self.separate_parts)
                )
                for source in range(num_active_sources):  # For each active source
                    source_activity_classes[
                        frame_idx,
                        source,
                        classes_to_int(
                            dataset=self.dataset_name,
                            class_str=class_active_events[source],
                        )[0],
                    ] = 1
                    mask_active_classes = (
                        source_activity_classes[frame_idx, source, :] == 1
                    )
                    direction_of_arrival_classes[
                        frame_idx, source, mask_active_classes, :
                    ] = np.deg2rad(
                        events_in_chunk[["azimuth", "elevation"]].to_numpy()[source, :]
                    )  # Extract the DOAs of the active sources
                ###################

        return (
            source_activity,
            direction_of_arrival,
            source_activity_classes,
            direction_of_arrival_classes,
        )

    def _get_parameter_hash(self) -> str:
        """Returns a hash value encoding the dataset parameter settings.

        :return: hash value
        """
        parameter_dict = {
            "chunk_length": self.chunk_length,
            "frame_length": self.frame_length,
            "num_fft_bins": self.num_fft_bins,
            "sequence_duration": self.sequence_duration,
        }

        return hashlib.md5(
            json.dumps(parameter_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        sequence = self.sequences[index]

        file_path, file_name = os.path.split(sequence["audio_file"])
        group_path, group_name = os.path.split(file_path)
        _, dataset_name = os.path.split(group_path)
        parameter_hash = self._get_parameter_hash()

        feature_file_name = file_name + "_" + str(sequence["chunk_idx"]) + "_f.h5"
        target_file_name = (
            file_name
            + "_"
            + str(sequence["chunk_idx"])
            + "_t"
            + str(self.max_num_sources)
            + ".h5"
        )

        path_to_feature_file = os.path.join(
            self.tmp_dir, dataset_name, group_name, parameter_hash
        )
        if not os.path.isdir(path_to_feature_file):
            try:
                os.makedirs(path_to_feature_file)
            except:
                pass

        if os.path.isfile(os.path.join(path_to_feature_file, feature_file_name)):
            with h5py.File(
                os.path.join(path_to_feature_file, feature_file_name), "r"
            ) as f:
                audio_features = f["audio_features"][:]
        else:
            audio_features = self._get_audio_features(
                sequence["audio_file"], sequence["start_time"], sequence["end_time"]
            )
            with h5py.File(
                os.path.join(path_to_feature_file, feature_file_name), "w"
            ) as f:
                f.create_dataset(
                    "audio_features", data=audio_features, compression="gzip"
                )

        if os.path.isfile(os.path.join(path_to_feature_file, target_file_name)):
            with h5py.File(
                os.path.join(path_to_feature_file, target_file_name), "r"
            ) as f:
                source_activity = f["source_activity"][:]
                direction_of_arrival = f["direction_of_arrival"][:]
                source_activity_classes = f["source_activity_classes"][:]
                direction_of_arrival_classes = f["direction_of_arrival_classes"][:]
        else:
            (
                source_activity,
                direction_of_arrival,
                source_activity_classes,
                direction_of_arrival_classes,
            ) = self._get_targets(sequence["annotation_file"], sequence["start_time"])
            with h5py.File(
                os.path.join(path_to_feature_file, target_file_name), "w"
            ) as f:
                f.create_dataset(
                    "source_activity", data=source_activity, compression="gzip"
                )
                f.create_dataset(
                    "direction_of_arrival",
                    data=direction_of_arrival,
                    compression="gzip",
                )
                f.create_dataset(
                    "source_activity_classes",
                    data=source_activity_classes,
                    compression="gzip",
                )
                f.create_dataset(
                    "direction_of_arrival_classes",
                    data=direction_of_arrival_classes,
                    compression="gzip",
                )

        return audio_features.astype(np.float32), (
            source_activity.astype(np.float32),
            direction_of_arrival.astype(np.float32),
            source_activity_classes.astype(np.float32),
            direction_of_arrival_classes.astype(np.float32),
        )

    # Function to separate text, numeric part, and file extension
    def separate_parts(self, sound_event_recording):
        if self.dataset_name == "ansim":
            index = 0
            while (
                index < len(sound_event_recording)
                and not sound_event_recording[index].isdigit()
            ):
                index += 1

            text_part = sound_event_recording[:index]
            number_part = sound_event_recording[index:]

            file_extension = ""
            if "." in number_part:
                number_part, file_extension = number_part.split(".", 1)

            return text_part

    def perclass_noise_labels(
        self, events_in_chunk, sigma_classes, dataset_name="ansim"
    ):
        """Add noise to the labels of the events in a chunk of data."""

        for row in events_in_chunk.itertuples():
            class_event = self.separate_parts(row.sound_event_recording)
            class_event, sigma_class = classes_to_int(
                dataset=dataset_name, class_str=class_event, sigma_classes=sigma_classes
            )

            events_in_chunk.at[row.Index, "elevation"] += np.random.normal(
                loc=0.0, scale=sigma_class
            )
            events_in_chunk.at[row.Index, "azimuth"] += np.random.normal(
                loc=0.0, scale=(sigma_class)
            )

        return events_in_chunk
