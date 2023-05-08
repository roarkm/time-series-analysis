import pandas as pd

from src.data import RAW_DATA_PATH
from src.features.build_features import (add_dayofweek_feature,
                                         add_holiday_feature)


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Create a DataFrame from csv file.
    Adds a date column by re-windowing the timestamp to a period of one day.
    """
    df = pd.read_csv(data_path,
                     header=None,
                     names=['seq_guid', 'timestamp', 'value'],
                     dtype={'seq_guid': str, 'timestamp': str, 'value': int})
    df['date'] = (
        pd.to_datetime(df.timestamp, format='%Y-%m-%dT%H:%M:%S.%fZ')
        .dt.normalize()
    )
    return df


def merge_ambiguous_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop (date, seq_guid) collisions and add new row with daily total value.
    """
    idx = (
        df.duplicated(subset=['seq_guid', 'date'], keep=False)
        ^ df.duplicated(subset=['seq_guid', 'date', 'value'], keep=False)
    )
    collisions = df[idx]
    for (_, _), new_df in collisions.groupby(['seq_guid', 'date']):
        # drop ambiguous rows
        df = df.drop(new_df.index)

        # create new row with daily total
        row = new_df.iloc[0].copy()
        row.value = new_df.value.sum()

        # add new row back in to df
        df = pd.concat([df, row.to_frame().T])
    return df


def prune_duplicate_rows(df: pd.DataFrame) -> None:
    """
    Removes duplicate rows.
    """
    df.drop_duplicates(subset=['seq_guid', 'date', 'value'], inplace=True)


def add_index(df: pd.DataFrame) -> None:
    """
    Sets multi-index to order by seq_guid then date.
    """
    df.set_index(['seq_guid', 'date'], inplace=True)
    df.sort_index(inplace=True)


def load_dataset_and_build_features() -> pd.DataFrame:
    df = load_raw_data(RAW_DATA_PATH)
    df = merge_ambiguous_rows(df)
    prune_duplicate_rows(df)
    add_dayofweek_feature(df)
    add_holiday_feature(df)
    add_index(df)
    return df


def get_sequence(df: pd.DataFrame, guid: str) -> pd.DataFrame:
    """Returns a DataFrame indexed by date (not  a multiindex)
    """
    if isinstance(df.index, pd.MultiIndex):
        seq = df.loc[guid]
    else:
        seq = (df[df.seq_guid == guid]
               .drop('seq_guid', axis=1)
               .set_index('date'))
    return seq


if __name__ == '__main__':
    df = load_dataset_and_build_features()
    print(df.head())
