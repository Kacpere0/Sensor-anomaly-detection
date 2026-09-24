import pandas as pd
from scipy.signal import butter, filtfilt
from data_loader import build_complete_dataset

WINDOW_SIZE = 0.5
OVERLAP = 0.75
STEP_SIZE = WINDOW_SIZE * (1 - OVERLAP)


def lowpass_filter_signal(df, cutoff=150, fs=6700, order=4):
    """Aplikuje filtr dolnoprzepustowy zero-phase na kolumny sygnałowe."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    df_filtered = df.copy()
    
    columns_to_filter = [col for col in df.columns if col != "Time"]
    for col in columns_to_filter:
        df_filtered[col] = filtfilt(b, a, df[col])
        
    return df_filtered


def create_windows(df, window_size, step_size, start_time=None, end_time=None):

    if start_time is None:
        start_time = df["Time"].iloc[0]

    if end_time is None:
        end_time = df["Time"].iloc[-1]

    windows = []
    current_start = start_time

    while current_start + window_size <= end_time:
        current_end = current_start + window_size

        window = df[
            (df["Time"] >= current_start) &
            (df["Time"] < current_end)
        ].copy()

        windows.append(window)
        current_start += step_size

    return windows


def create_record_windows(record):

    start_time = max(
        record.acc["Time"].iloc[0],
        record.gyro["Time"].iloc[0],
        record.mic["Time"].iloc[0]
    )

    end_time = min(
        record.acc["Time"].iloc[-1],
        record.gyro["Time"].iloc[-1],
        record.mic["Time"].iloc[-1]
    )
    
    filtered_gyro = lowpass_filter_signal(record.gyro, cutoff=150, fs=6700, order=4)

    acc_windows = create_windows(
        record.acc,
        WINDOW_SIZE,
        STEP_SIZE,
        start_time,
        end_time
    )

    gyro_windows = create_windows(
        filtered_gyro,
        WINDOW_SIZE,
        STEP_SIZE,
        start_time,
        end_time
    )

    mic_windows = create_windows(
        record.mic,
        WINDOW_SIZE,
        STEP_SIZE,
        start_time,
        end_time
    )

    return acc_windows, gyro_windows, mic_windows


if __name__ == "__main__":
    train_dataset, test_dataset = build_complete_dataset()
    if train_dataset:
        record = train_dataset[0]
        acc_windows, gyro_windows, mic_windows = create_record_windows(record)