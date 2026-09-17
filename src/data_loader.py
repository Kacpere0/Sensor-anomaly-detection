from pathlib import Path
import pandas as pd

DATA_ROOT = Path("../data/RoboticArm")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

class SensorRecord:
    def __init__(self, df_acc, df_gyro, df_mic, metadata):
        self.acc = df_acc
        self.gyro = df_gyro
        self.mic = df_mic
        self.metadata = metadata

    def __repr__(self):
        meta_keys = [
            "segment_id",
            "split_label",
            "anomaly_label",
            "domain_shift_op",
            "domain_shift_env"
        ]
        meta_str = "\n".join(
            f"  {key}: {self.metadata.get(key, 'Brak')}"
            for key in meta_keys
            if key in self.metadata
        )
        return (
            f"{self.metadata.get('segment_id', 'Nieznany')} ===\n\n"
            f"{meta_str}\n\n"
            f"  acc:  {self.acc.shape[0]} rows, {self.acc.shape[1]} columns\n"
            f"  gyro: {self.gyro.shape[0]} rows, {self.gyro.shape[1]} columns\n"
            f"  mic:  {self.mic.shape[0]} rows, {self.mic.shape[1]} columns"
        )


def load_sensor(path, columns):
    return pd.read_parquet(path)[columns]


def build_dataset(metadata, data_dir):  
    dataset = []
    
    for _, row in metadata.iterrows():
        record = SensorRecord(
            df_acc=load_sensor(Path(data_dir) / row["ism330dhcx_acc"], ["Time", "A_x [g]", "A_y [g]", "A_z [g]"]),
            df_gyro=load_sensor(Path(data_dir) / row["ism330dhcx_gyro"], ["Time", "G_x [mdps]", "G_y [mdps]", "G_z [mdps]"]),
            df_mic=load_sensor(Path(data_dir) / row["imp23absu_mic"], ["Time", "MIC [Waveform]"]),
            metadata=row.to_dict()
        )
        dataset.append(record)
        
    return dataset

def build_complete_dataset():
    """
    first part of the tuple is train dataset, the second is test dataset
    """
    train_metadata = pd.concat([
        pd.read_csv(TRAIN_DIR / f"attributes_{sub}.csv")
        for sub in ["normal_source_train", "normal_target_train"]
    ], ignore_index=True)

    test_metadata = pd.concat([
        pd.read_csv(TEST_DIR / "attributes_normal_source_test.csv"),
        pd.read_csv(TEST_DIR / "attributes_anomaly_source_test.csv"),
        pd.read_csv(TEST_DIR / "attributes_normal_target_test.csv"),
        pd.read_csv(TEST_DIR / "attributes_anomaly_target_test.csv")
    ], ignore_index=True)

    train_dataset = build_dataset(train_metadata, TRAIN_DIR)
    test_dataset = build_dataset(test_metadata, TEST_DIR)

    return (train_dataset, test_dataset)