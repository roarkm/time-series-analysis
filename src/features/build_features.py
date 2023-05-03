import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np


def add_weekday_features(df: pd.DataFrame) -> None:
    df['weekday'] = (
        df.date.dt.dayofweek
        # .map({0: 'M', 1: 'T', 2: 'W', 3: 'Th', 4: 'F', 5: 'Sa', 6: 'Su'})
    )


def add_holiday_features(df: pd.DataFrame) -> None:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.date.min(),
                            end=df.date.max())
    df['fed_holiday'] = df.date.isin(holidays).astype(np.uint8)
