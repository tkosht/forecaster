import numpy
import pandas


class Score(object):
    def __init__(self, a: numpy.ndarray, p: numpy.ndarray):
        self.a = numpy.array(a)  # actual/observation values
        self.p = numpy.array(p)  # predicted/forecast values
        assert len(self.a) == len(self.p)

    def mae(self) -> numpy.float:
        return numpy.abs((self.a - self.p)).mean()

    def mse(self) -> numpy.float:
        return ((self.a - self.p) ** 2).mean()

    def rmse(self) -> numpy.float:
        return self.mse() ** 0.5

    def mape(self) -> numpy.float:
        a = self._adjusted_values(self.a)
        return numpy.abs(((self.a - self.p) / a)).mean()

    def mspe(self) -> numpy.float:
        a = self._adjusted_values(self.a)
        return (((self.a - self.p) / a) ** 2).mean()

    def rmspe(self) -> numpy.float:
        return self.mspe() ** 0.5

    def _adjusted_values(self, v) -> numpy.ndarray:
        v = v.copy()
        c = 1.0
        if (v != 0).any():
            c = v[v != 0].mean()
        v[v == 0] = c
        return v

    def sqmrpa(self) -> numpy.float:  # square measurement sqmr of prediction/actual
        a = self._adjusted_values(self.a)
        return self.p.sum() / a.sum()

    def sqmrap(self) -> numpy.float:  # square measurement sqmr of actual/prediction
        p = self._adjusted_values(self.p)
        return self.a.sum() / p.sum()

    def gmpa(self) -> numpy.float:
        a = self._adjusted_values(self.a)
        r = numpy.abs(self.p / a)
        if (r == 0).all():
            return 0.0
        return r.prod() ** (1 / len(r))

    def gmap(self) -> numpy.float:
        p = self._adjusted_values(self.p)
        r = numpy.abs(self.a / p)
        if (r == 0).all():
            return 0.0
        return r.prod() ** (1 / len(r))

    def rsq(self) -> numpy.float:  # R square
        srs = ((self.a - self.p) ** 2).sum()  # sum of residuals squared
        srm = ((self.a - self.a.mean()) ** 2).sum()  # sum of residual from mean
        if srm == 0:
            srm = 1.0
        return 1 - srs / srm

    @property
    def dic(self) -> dict:
        return dict(
            mae=self.mae(),
            mse=self.mse(),
            rmse=self.rmse(),
            mape=self.mape(),
            mspe=self.mspe(),
            rmspe=self.rmspe(),
            sqmrpa=self.sqmrpa(),
            sqmrap=self.sqmrap(),
            gmpa=self.gmpa(),
            gmap=self.gmap(),
            rsq=self.rsq(),
        )

    @property
    def df(self) -> pandas.DataFrame:
        s = self.dic
        df = pandas.DataFrame(s, index=["scores"]).T
        return df

    def to_csv(self, tsv_file, sep=","):
        self.df.to_csv(tsv_file, sep=sep, header=False)

    def to_json(self, json_file):
        self.df.to_json(json_file)

    @property
    def json(self) -> str:
        return self.df.to_json()


if __name__ == "__main__":
    y = numpy.array([1.1, 1.2, 1.3])
    p = numpy.array([1.0, 0.9, 1.4])
    score = Score(y, p)
    print(score.df)
    print(score.json)
    score.to_csv("score.tsv")
    score.to_json("score.json")

    y = numpy.array([0.1, 0.2, -0.1])
    p = numpy.array([0.0, 0.0, 0.0])
    score = Score(y, p)
    print(score.df)

    y = numpy.array([0.0, 0.0, 0.0])
    p = numpy.array([0.1, 0.2, -0.1])
    score = Score(y, p)
    print(score.df)
