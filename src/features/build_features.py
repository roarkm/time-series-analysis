import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np


def add_dayofweek_feature(df: pd.DataFrame) -> None:
    df['day_of_week'] = (
        df.date.dt.dayofweek
        # .map({0: 'M', 1: 'T', 2: 'W', 3: 'Th', 4: 'F', 5: 'Sa', 6: 'Su'})
    )
    df['weekend'] = (
        df.day_of_week
        .map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1})
    )


def add_holiday_feature(df: pd.DataFrame) -> None:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.date.min(),
                            end=df.date.max())
    df['fed_holiday'] = df.date.isin(holidays).astype(np.uint8)
