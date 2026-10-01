"""
This module provides a PyTorch dataset and dataloader class for loading
prostate volumes from H5 and CSV files.

The expected structure of an H5 file is:
    ['axt2']            : axial T2 volume
    ['adc']             : ADC volume
    ['b1500']           : b1500 volume
    ['dce']             : DCE volume (optional)
    ['axt2_mask']       : axial T2 mask
    .attrs['maxPIRADS'] : max PI-RADS score for the exam
    .attrs['lesion_t2'] : max PI-RADS score for the lesion as seen in T2
    .attrs['lesion_dwi']: max PI-RADS score for the lesion as seen in DWI
    .attrs['ManufacturerModelName']: scanner type
    .attrs['PatientAge']: patient age
    .attrs['ScannerStrength']: 1.5T or 3T
    .attrs['psa']: PSA value. (float) or 'N/A' if not available
    .attrs['prostate_volume']: prostate volume in cc. (float) or 'N/A' if not available
    .attrs['reader_ID']: reader ID of physician or 'N/A' if not available
"""

import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from src.data.processing import (
    preprocess_volume,
    normalize_volume,
)
from src.data.sampling import DualLabelBiasedSampler, get_sampler
from src.utils.data_enums import SeriesType, binarize_gleason_score


class ExamH5Dataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        data_dirs,
        series,
        model_type,
        augment,
        noise_sigma_range,
        downsample_factors,
        pirads_cutoff,
        mask_prostate,
        device,
        mode,
        normalize=True,
        target="pirads",
        axt2_key="axt2",
        dwi_suffices=None,
        dce_dirs=None,
        tabular_csv = None,
        load_volumes=True,
    ):
        """
        metadata_csv: path to csv containing labels and metadata for each
            accession number
        datadir: directory containing h5 files
        series: list of series to load from each h5 file
        dce_dir (str | Path, optional): directory containing DCE h5 files when
            they are stored separately from other series
        model_type: '3D' or '2D' model
        augment (str): augmentation mode 'none', 'noise', or 'downsample'
        noise_sigma_range (list[float]): min and max sigma for noise augmentation
        downsample_factors (list[int]): factors for k-space downsampling
        pirads_cutoff: the PIRADS score to above which the label is positive.
            pirads_cutoff=None will leave labels as PI-RADS Score.
        mask_prostate (bool): Indicates whether or not to mask the prostate
        device: device to load tensors on
        normalize (bool): Apply z-score normalization to each volume.
        tabular_csv (str | Path, optional): CSV of clinical features, one row
            per AccessionNumber (plus a 'split' column, which is dropped).
        load_volumes (bool): If False, skip locating and reading the H5/MRI
            volumes entirely and return empty volume dicts. For tabular-only
            models (e.g. mlpclinical.ClinicalMLPModel) that never use them.
        """
        super().__init__()
        self.series = series
        self.augment = augment
        self.noise_sigma_range = noise_sigma_range
        self.downsample_factors = downsample_factors
        self.model_type = model_type
        self.device = device
        self.pirads_cutoff = int(pirads_cutoff)
        self.mask_prostate = mask_prostate
        self.data_dirs = [Path(d) for d in data_dirs]
        self.mode = mode
        self.normalize = normalize
        self.target = target
        self.axt2_key = axt2_key
        self.dwi_suffices = dwi_suffices if dwi_suffices is not None else [None]
        self.num_variants = len(self.dwi_suffices)
        self.dce_dirs = [Path(d) for d in dce_dirs] if dce_dirs is not None else []
        self._dce_lookup = {}
        self.load_volumes = load_volumes
        self.tabular_features = {}
        # Width of the clinical feature vector, taken from the CSV itself so the
        # zero-filled fallback for exams missing from it always matches the real
        # rows (it was hardcoded to 11 while the CSV has 37 features, which made
        # batch collation crash whenever an exam had no clinical row).
        self.num_tabular_features = 0
        if tabular_csv is not None:
            tab_df = pd.read_csv(tabular_csv)
            tab_df = tab_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            feature_cols = tab_df.columns.drop(["AccessionNumber", "split", "PatientID", "csPCa", "MaxGradeGroup", "MaxGleasonScore", "lesion_has_epe"])
            self.num_tabular_features = len(feature_cols)
            for _, row in tab_df.iterrows():
                acc = int(row["AccessionNumber"])
                features = row[feature_cols].values.astype(float)
                self.tabular_features[acc] = features


        if not self.data_dirs:
            raise ValueError("At least one data directory must be provided.")
        if SeriesType.DCE in self.series and not self.dce_dirs:
            raise ValueError("DCE series requested but no DCE directories were provided.")

        # read in metadata for each accession number
        df_metadata = pd.read_csv(metadata_csv) #opens the CSV file and loads into dataframe.  Every row is one exam, every column is one piece of info about that exam

        df_metadata["pirads_target"] = df_metadata["maxPIRADS"].apply( #adds a new column called pirads target
            lambda x: 1 if x > self.pirads_cutoff else 0
        ) #if the exam's maxvalue for pirads, i.e maxpirads is greater than 3, pirads target mein we put 1, else we put 0


        df_metadata["gleason_target"] = df_metadata["MaxGleasonScore"].apply(
            lambda x: binarize_gleason_score(x)
        ) #add a new column gleason target, and binarises the value according to the function I assume which makes it 0 or 1

        if "ordinal_label" in df_metadata.columns:
            df_metadata["tstage_target"] = (
                df_metadata["ordinal_label"].apply(lambda x: 1 if x >= 2 else 0)
            ) #check if ordinal label exists. If it does, add a tstage_target column ,  1 if the ordinal label is 2 or higher (meaning the cancer has spread outside the prostate), 0 otherwise.

        if self.target == "gleason":
            pos = df_metadata["MaxGleasonScore"] > 6  #creates a variable called pos that holds T/F values based on if the gleason score is greater than 6 or not
            neg_gs6 = df_metadata["MaxGleasonScore"] == 6 #T/F if maxgleasonscore is 6 or not
            neg_pirads = (df_metadata["maxPIRADS"].isin([1, 2])) & (
                df_metadata["MaxGleasonScore"] == -1
            ) # if the maxpirads score is 1 or 2 and maxgleason score is -1, means no biopsy done, thats a negative pirads
            valid = pos | neg_gs6 | neg_pirads # | = or, a row is valid if its positive, or a gleason 6 negative or a pirads 1/2 negative. essentially we wanna know whats up, for sure
            df_metadata = df_metadata[valid].copy() #takes the full dataset and keeps only the valid rows.  .copy() makes a fresh copy so we're not accidentally editing the original.
            df_metadata.loc[pos, "target"] = 1 #everywhere where pos is True, we write 1 in the target
            df_metadata.loc[neg_gs6 | neg_pirads, "target"] = 0 #Goes to every row where either negative condition is True and writes 0 in the target column. These are your negative training examples.
        elif self.target == "tstage": #if goal is t stage and not gleason
            if "ordinal_label" not in df_metadata.columns:
                raise ValueError(
                    "CSV is missing 'ordinal_label' column required for tstage mode."
                )
            df_metadata["target"] = df_metadata["tstage_target"].astype(int) #just use the t/f we generated earlier
        elif self.target == "cspca": #if the target is to predict clinically significant prostate cancer

            # Clinically significant prostate cancer from biopsy-confirmed
            # labels only. The `csPCa` column (1 = ISUP grade group >= 2,
            # 0 = benign biopsy or grade group 1) is produced by
            # intern/make_labels.py and is <NA> for exams with no biopsy on
            # record. We drop the no-biopsy exams entirely (in every mode,
            # including eval) rather than presuming them negative.
            if "csPCa" not in df_metadata.columns:
                raise ValueError(
                    "CSV is missing 'csPCa' column required for cspca mode. "
                    "Generate it with intern/make_labels.py."
                )
            df_metadata = df_metadata[df_metadata["csPCa"].notna()].copy() #drop every row where cspca is NA
            df_metadata["target"] = df_metadata["csPCa"].astype(int) #create a target column by copying cspca where 1 means significant cancer and 0 means benighn
        else:
            df_metadata["target"] = df_metadata["pirads_target"] #if its not gleason, tstage or cspca we are predicting, just use pi-rads labels

        if SeriesType.DCE in self.series and self.dce_dirs:
            df_metadata["dce_path"] = df_metadata["AccessionNumber"].apply(
                lambda acc: self._find_existing_h5(acc, self.dce_dirs)
            )
            missing_mask = df_metadata["dce_path"].isna()
            if missing_mask.any():
                missing_accessions = df_metadata.loc[
                    missing_mask, "AccessionNumber"
                ].tolist()
                print(
                    "Skipping cases without DCE files:",
                    ", ".join(map(str, missing_accessions)),
                )
            df_metadata = df_metadata.loc[~missing_mask].copy()
            self._dce_lookup = dict(
                zip(df_metadata["AccessionNumber"], df_metadata["dce_path"])
            )
            df_metadata.drop(columns=["dce_path"], inplace=True)
            if df_metadata.empty:
                raise ValueError(
                    "No cases with available DCE files after filtering missing DCE entries"
                )

        self.df_metadata = df_metadata.reset_index(drop=True) #Saves the final cleaned dataframe and renumbers rows 0, 1, 2... cleanly after all the filtering.

        if tabular_csv is not None:
            n_missing = (
                ~self.df_metadata["AccessionNumber"].astype(int).isin(self.tabular_features)
            ).sum()
            if n_missing:
                print(
                    f"[ExamH5Dataset:{mode}] {n_missing}/{len(self.df_metadata)} exams "
                    f"have no row in {tabular_csv}; their clinical features will be "
                    f"all zeros."
                )

        self.pirads = self.df_metadata["maxPIRADS"].tolist() * self.num_variants
        self.pirads_labels = (
            self.df_metadata["pirads_target"].tolist() * self.num_variants
        )
        self.gleason_labels = (
            self.df_metadata["gleason_target"].tolist() * self.num_variants
        )

        self.class_weights = torch.from_numpy(
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(self.df_metadata["target"]),
                y=self.df_metadata["target"],
            )
        ).float()

    @staticmethod # Means this function belongs to the class but doesn't need self — it's just a utility function that could technically live anywhere.
    def _find_existing_h5(accession_number: int, search_dirs):
        """Return the first matching H5 path for an accession number.

        Args:
            accession_number (int): Accession number to search for.
            search_dirs (Iterable[Path]): Directories to scan.

        Returns:
            Path | None: The first path where ``<acc>.h5`` exists, or ``None`` if
            nothing is found.
        """

        acc_str = str(accession_number) #checks if accession is there or not
        for d in search_dirs:
            candidate = d / f"{acc_str}.h5"
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, index):
        #keep in mind that we are skipping the num_varients logic bc bpmri not mpmri

        case_index = index // self.num_variants #case index = index, index =1:14 PMClaude responded: index is just the row number in the dataset table — 0 means first row, 1 means second row, etc.index is just the row number in the dataset table — 0 means first row, 1 means second row, etc.
        variant_index = index % self.num_variants #varient index = index

        #convert the following into numbers

        accession_number = int(self.df_metadata.iloc[case_index]["AccessionNumber"])
        patient_id = int(self.df_metadata.iloc[case_index]["PatientID"])
        max_pirads_from_csv = int(self.df_metadata.iloc[case_index]["maxPIRADS"])
        gleason_target = int(self.df_metadata.iloc[case_index]["gleason_target"])
        max_gleason_score = int(self.df_metadata.iloc[case_index]["MaxGleasonScore"])
        target = int(self.df_metadata.iloc[case_index]["target"]) #Grab the label for this exam — 0 or 1. This is the ground truth the model is trying to predict.

        tabular_features = self.tabular_features.get(accession_number)
        if tabular_features is not None:
            tabular_tensor = torch.FloatTensor(tabular_features)
        else:
            tabular_tensor = torch.zeros(self.num_tabular_features, dtype=torch.float32)

        if not self.load_volumes:
            # Tabular-only model: don't locate, open or read any H5 file.
            return {
                "volume_data_dict": {},
                "unnorm_volume_data_dict": {},
                "label": target,
                "gleason_label": gleason_target,
                "maxPIRADS": max_pirads_from_csv,
                "MaxGleasonScore": max_gleason_score,
                "AccessionNumber": accession_number,
                "PatientID": patient_id,
                "location": "",
                "TabularFeatures": tabular_tensor,
            }

        path = self._find_existing_h5(accession_number, self.data_dirs)
        if path is None:
            raise FileNotFoundError(
                f"Could not find H5 file for accession {accession_number} in any of {self.data_dirs}"
            )
        feature_dict = {} #empty dictionary to store the final processed MRI volumes (T2, ADC, DWI).
        unnorm_feature_dict = {} #Same but for unnormalized volumes — stored separately in case you need the raw values later.
        dce_path = None
        if self.dce_dirs and (SeriesType.DCE in self.series):
            dce_path = self._dce_lookup.get(accession_number)
            if dce_path is None:
                dce_path = self._find_existing_h5(accession_number, self.dce_dirs)

        with h5py.File(path, "r") as f:
            dce_file = None
            if dce_path is not None:
                if not dce_path.exists():
                    raise FileNotFoundError(
                        f"Expected DCE file {dce_path} for accession {accession_number}"
                    )
                dce_file = h5py.File(dce_path, "r")
            sigma = None #no noise level
            if self.augment == "noise":
                sigma = random.uniform( #If augmentation is set to noise mode, pick a random noise level between the min and max values in the config. Your config has augment: none so this is skipped.
                    self.noise_sigma_range[0], self.noise_sigma_range[1]
                )

            vol_dict = {} #Empty dictionary that will store the raw loaded volumes before processing.
            variant_suffix = self.dwi_suffices[variant_index]
            if variant_suffix is None:
                adc_key = SeriesType.ADC.value["key"]
                b1500_key = SeriesType.B1500.value["key"]
            else:
                adc_key = f"adc_{variant_suffix}"
                b1500_key = f"b1500_{variant_suffix}"

            try:
                for s in self.series:
                    if s == SeriesType.AXT2:
                        key = self.axt2_key
                    elif s == SeriesType.ADC:
                        key = adc_key
                    elif s == SeriesType.B1500:
                        key = b1500_key
                    else:
                        key = s.value["key"]

                    file_handle = (
                        dce_file if (s == SeriesType.DCE and dce_file is not None) else f
                    ) #Pick which file to read from — DCE file if it's the DCE sequence, otherwise the regular H5 file f. For you always f.
                    source_path = dce_path if file_handle is dce_file else path #Just for error messages — track which file path we're reading from.
                    if key not in file_handle:
                        raise ValueError(f"Series {key} not in {source_path}")
                    vol_raw = file_handle[key][:] #Actually load the MRI volume from the H5 file into memory. [:] means load the whole thing.

                    vol_proc = preprocess_volume(
                        vol_raw,
                        s,
                        normalize=False,
                        augment=self.augment,
                        zero_pad=True,
                        sigma=sigma,
                        noise_sigma_range=self.noise_sigma_range,
                        downsample_factors=self.downsample_factors,
                    ) #Run preprocessing on the raw volume — zero padding, augmentation, etc. Returns a cleaned up numpy array.
                    vol_dict[s] = vol_proc
                    # store unnormalized volume for optional saving later
                    unnorm_feature_dict[s.value["key"]] = torch.unsqueeze(
                        torch.FloatTensor(vol_proc),
                        0,
                    ) #It stores the raw unnormalized version of the volume before normalization is applied. Useful for debugging or visualization — if you want to look at what the actual MRI scan looks like without any preprocessing applied, you have it saved separately. Not used during training itself.
            finally:
                if dce_file is not None:
                    dce_file.close()

            for s, vol in vol_dict.items(): #Loop through each MRI sequence and its volume. So first iteration: s=T2, vol=T2 scan data. Second: s=ADC, vol=ADC scan data. Third: s=DWI, vol=DWI scan data.
                if self.normalize:
                    vol = normalize_volume(vol) #Scale the pixel values so they're not wildly different across scans. Like if one scan has values 0-1000 and another has 0-5000, normalize makes them both roughly 0-1. Makes training more stable.

                vol = torch.FloatTensor(vol) #The volume right now is a numpy array — like a regular Python math object. Convert it to a PyTorch tensor — the format the model actually understands and can do GPU math on.

                vol = torch.unsqueeze(vol, 0) #The volume is a 3D cube (depth, height, width). The model expects a 4D input (channels, depth, height, width). This adds the channels dimension — like how a photo has 3 color channels (RGB), an MRI has 1 channel (grayscale). So it goes from shape (20, 64, 64) to (1, 20, 64, 64).

                feature_dict[s.value["key"]] = vol #Store the final processed volume in the dictionary under its sequence name. At the end you have a dictionary like {"axt2": tensor, "adc": tensor, "b1500": tensor} — all three sequences ready to feed into the model.


        return {
            "volume_data_dict": feature_dict,
            "unnorm_volume_data_dict": unnorm_feature_dict,
            "label": target,
            "gleason_label": gleason_target,
            "maxPIRADS": max_pirads_from_csv,
            "MaxGleasonScore": max_gleason_score,
            "AccessionNumber": accession_number,
            "PatientID": patient_id,
            "location": str(path),
            "TabularFeatures": tabular_tensor
            
        }
    def __len__(self):
        # Total size of the dataset: number of exams (rows) times num_variants
        # (1 unless dwi_suffices lists several DWI variants). The DataLoader uses
        # this to know when an epoch is done. (Previously this sat at module
        # level, outside the class, so len(dataset) raised TypeError.)
        return self.df_metadata.shape[0] * self.num_variants


def load_data(
    data_csv,
    data_dirs,
    dce_dirs=None,
    series=[SeriesType.AXT2, SeriesType.ADC, SeriesType.B1500],
    model_type="3D",
    augment="none",
    noise_sigma_range=(0.0, 0.15),
    downsample_factors=None,
    pirads_cutoff=None,
    mask_prostate=False,
    device="cpu",
    batch_size=4,
    num_workers=4,
    shuffle=False,
    weighted_sample=False,
    multitask_sampler=False,
    mode="train",
    normalize=True,
    target="pirads",
    axt2_key="axt2",
    dwi_suffices=None,
    tabular_csv = None
):
    """
    data_csv: path to csv containing labels and metadata for each accession
        number
    data_dirs: directories containing h5 files. Files in these directories
        should be a set containing the accession numbers in data_csv
    dce_dirs: optional list of directories containing DCE h5 files when they
        reside outside of ``data_dirs``
    series: list of series to include in the dataset. Default is
        [SeriesType.AXT2, SeriesType.ADC, SeriesType.B1500]
    model_type: '3D' or '2D' model
    augment (str): augmentation mode 'none', 'standard', 'noise', or 'downsample'
    noise_sigma_range (tuple): min and max sigma for noise augmentation
    downsample_factors (list[int]): factors for k-space downsampling
    pirads_cutoff: the PIRADS score to above which the label is positive.
        pirads_cutoff=None will leave labels as PI-RADS Score. Default is None.
    mask_prostate (bool): Indicates whether or not to mask the prostate
    device: device to load tensors on, "cpu" or "cuda:0"
    batch_size: batch size for data loader
    num_workers: number of workers for data loader
    shuffle: boolean for whether to shuffle data loader
    weighted_sample: boolean for whether to use weighted sampling
    normalize (bool): Apply z-score normalization to each volume
    target (str): "pirads" to train on PIRADS > cutoff, "gleason" to train on
        Gleason score > 6, or "tstage" to train on extraprostatic extension
        where positive labels correspond to ``ordinal_label >= 2``. When
        ``target='gleason'`` the dataset is filtered so only cases with valid
        Gleason labels remain.

    """
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs] #Just makes sure data_dirs is always a list even if someone passed a single path string.

    if dce_dirs is None:
        dce_dirs = []
    elif isinstance(dce_dirs, (str, Path)):
        dce_dirs = [dce_dirs]

    dataset = ExamH5Dataset(
        data_csv,
        data_dirs,
        series,
        model_type,
        augment,
        noise_sigma_range,
        downsample_factors,
        pirads_cutoff,
        mask_prostate,
        device=device,
        mode=mode,
        normalize=normalize,
        target=target,
        axt2_key=axt2_key,
        dwi_suffices=dwi_suffices,
        dce_dirs=dce_dirs,
        tabular_csv = tabular_csv
    )

    if weighted_sample: #Use a weighted sampler — oversamples the minority class (positive cancer cases) so the model sees them more often. Helps with class imbalance.
        sampler = get_sampler(dataset, pirads_cutoff) #Creates the weighted sampler object that knows how to oversample positives.
        data_loader = DataLoader( #Pass the sampler to the dataloader. shuffle=False because the sampler is already handling the ordering — you don't want both.
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            sampler=sampler,
            generator=torch.Generator(device="cuda"), #Random number generator on the GPU for reproducibility.
            pin_memory=True,
        )
    else:
        if multitask_sampler:
            sampler = DualLabelBiasedSampler(dataset, batch_size=batch_size)
            data_loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                generator=torch.Generator(device="cuda"),
                pin_memory=True,
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                generator=torch.Generator(device="cuda"),
                pin_memory=True,
            )

    return data_loader