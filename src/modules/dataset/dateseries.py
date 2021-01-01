from __future__ import annotations

import numpy
import pandas
import datetime
from functools import cached_property
from dataclasses import dataclass


def make_window(df, wsz, wshift) -> numpy.ndarray:
    seqsets = []
    for seq in generate_window(df, wsz, wshift):
        seqsets.append(seq)
    return numpy.array(seqsets)


def generate_window(df, wsz, wshift) -> numpy.ndarray:
    n = len(df)
    for idx in range(0, n - wsz + 1, wshift):
        seq = df.values[idx : idx + wsz]
        yield seq


def to_date(ti: numpy.array):
    _ti = pandas.to_datetime(ti)
    if str(ti.dtype)[-2:] == "64":
        return _ti

    ti_ser = pandas.Series(_ti, name="time_index")

    def _adjust_date(ts):
        dte = ts.to_pydatetime()
        # dte += datetime.timedelta(minutes=3)  # more precisely
        dte += datetime.timedelta(hours=1)
        return datetime.datetime(
            year=dte.year, month=dte.month, day=dte.day, hour=0, minute=0, second=0
        )

    return ti_ser.apply(_adjust_date)


def next_date(s: pandas.Series) -> datetime.datetime:
    e = to_date(s.tail(1)).reset_index(drop=True)[0]
    nxt = e.to_pydatetime() + datetime.timedelta(days=1)
    return nxt


@dataclass
class CategorySeries(object):
    data: numpy.ndarray
    cols: list
    list_n_labels: list[int]

    def __init__(self, df: pandas.DataFrame, wsz: int = 7, wshift: int = 1) -> None:
        super().__init__()
        self.df = df
        self.data = make_window(df, self.wsz, self.wshift)
        self.cols = df.columns.copy()
        self.n_labels = [len(set(df[col])) + 1 for col in self.cols]


class DatasetDateSeries(object):
    def __init__(
        self,
        start: str = "2016-01-01",
        end: str = "2018-12-31",
        freq: str = "D",
        wsz: int = 7,
        wshift: int = 1,
        to_onehot=True,
    ) -> None:
        self.start = start  # start of date range
        self.end = end  # end of date range
        self.freq = freq  # date frequancy, default "D" day unit
        self.wsz = wsz  # window size
        self.wshift = wshift  # size of shifting window
        self.to_onehot = to_onehot  # to encode to onehot or not

        # variable to be initialized
        self._df = None
        self.ti_win = None
        self.kn_win = None

        self._make_datedata()._make_window()

    def _make_datedata(self) -> DatasetDateSeries:
        df = pandas.DataFrame([])
        index_date = pandas.date_range(self.start, self.end, freq=self.freq)
        df["month"] = index_date.month
        df["weekth"] = index_date.isocalendar().week.values.astype(numpy.int32)
        df["weekday"] = index_date.weekday
        if self.to_onehot:
            df.month = df.month.astype("category")
            df.weekth = df.month.astype("category")
            df.weekday = df.month.astype("category")
            df = pandas.get_dummies(df, ["month", "weekth", "weekday"])
        df["date_index"] = index_date.astype(numpy.int32)
        df.index = index_date
        self._df = df
        return self

    def _make_window(self) -> DatasetDateSeries:
        df = self.df.copy()
        self.ti_win = make_window(df["date_index"], self.wsz, self.wshift)
        self.ti_win = self.ti_win[:, :, numpy.newaxis]
        df.drop("date_index", axis=1, inplace=True)
        # self.kn_win = CategorySeries(df, self.wsz, self.wshift)
        self.kn_win = make_window(df, self.wsz, self.wshift)
        return self

    @cached_property
    def date_index(self) -> pandas.Series:
        return to_date(self._df.date_index)

    @property
    def df(self) -> pandas.DataFrame:
        return self._df

    @cached_property
    def next_date(self) -> datetime.datetime:
        return next_date(self.df.date_index)

    @cached_property
    def next_index(self, by: str = "2019-12-31") -> pandas.Series:
        s = next_date(self.df.date_index)
        return pandas.date_range(s, by, freq=self.freq)
