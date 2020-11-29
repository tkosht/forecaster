from __future__ import annotations
import numpy
import pandas
import datetime
from functools import cached_property


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
    return pandas.to_datetime(ti)


def _to_date(ti: numpy.array):
    _ti = pandas.to_datetime(ti)
    ti_ser = pandas.Series(_ti, name="time_index")

    def _adjust_date(ts):
        dte = ts.to_pydatetime()
        dte += datetime.timedelta(minutes=3)
        return datetime.datetime(
            year=dte.year, month=dte.month, day=dte.day, hour=0, minute=0, second=0
        )

    return ti_ser.apply(_adjust_date)


def next_date(s: pandas.Series):
    # e = to_date(s.tail(1)).reset_index(drop=True)[0]
    e = pandas.to_datetime(s.tail(1)).reset_index(drop=True)[0]
    nxt = e.to_pydatetime() + datetime.timedelta(days=1)
    return nxt


class DatasetDateData(object):
    def __init__(
        self, start="2016-01-01", end="2018-12-31", freq="D", wsz=7, wshift=1
    ) -> None:
        self.start = start  # start of date range
        self.end = end  # end of date range
        self.freq = freq  # date frequancy
        self.wsz = wsz  # window size
        self.wshift = wshift  # size of shifting window

        # variable to be initialized
        self._df = None
        self.seqti = None
        self.seqdata = None

        self._make_datedata()._make_window()

    def _make_datedata(
        self, start="2016-01-01", end="2018-12-31", freq="D"
    ) -> DatasetDateData:
        _df = pandas.DataFrame([])
        index_date = pandas.date_range(start, end, freq=freq)
        _df["month"] = index_date.month
        _df["weekth"] = index_date.isocalendar().week
        _df["weekday"] = index_date.weekday
        _df.month = _df.month.astype("category")
        _df.weekth = _df.month.astype("category")
        _df.weekday = _df.month.astype("category")
        df = pandas.get_dummies(_df, ["month", "weekth", "weekday"])
        df.index = index_date
        df["date_index"] = index_date.astype(numpy.int32)
        self._df = df
        return self

    def _make_window(self) -> DatasetDateData:
        df = self.df.copy()
        self.seqti = make_window(df["date_index"], self.wsz, self.wshift)
        self.seqti = self.seqti[:, :, numpy.newaxis]
        df.drop("date_index", axis=1, inplace=True)
        self.seqdata = make_window(df, self.wsz, self.wshift)
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
    def next_index(self, by="2019-12-31") -> pandas.Series:
        s = next_date(self.df.date_index)
        return pandas.date_range(s, by, freq=self.freq)
