import pytest
import numpy
import pandas
import torch
from src.modules.dataset.dateset import DateTensors, DatesetToy
from src.modules.dataset.dateseries import to_date

base = torch.arange(0, 60).view(12, 1, -1)
base2 = numpy.arange(0, 60).reshape(12, 1, -1)

fake = torch.arange(0, 60).view(12, 1, -1) + 1
fake2 = numpy.arange(0, 60).reshape(12, 1, -1) + 1


class TestDateTensors(object):
    @pytest.fixture
    def date_tensors(self):
        ti = tv = tc = kn = tg = base.clone()
        dtr = DateTensors(ti, tv, tc, kn, tg)
        return dtr

    @pytest.fixture
    def date_numpys(self):
        ti = tv = tc = kn = tg = base.clone()
        dtr = DateTensors(ti, tv, tc, kn, tg)
        return dtr

    def test_tensors_next(self, date_tensors):
        def _batch_check(batch_size):
            tsr = next(date_tensors(bsz=batch_size))
            assert (tsr.ti[:batch_size] == base[:batch_size]).all()
            assert (tsr.tv[:batch_size] == base[:batch_size]).all()
            assert (tsr.tc[:batch_size] == base[:batch_size]).all()
            assert (tsr.kn[:batch_size] == base[:batch_size]).all()
            assert (tsr.tg[:batch_size] == base[:batch_size]).all()

        _batch_check(batch_size=1)
        _batch_check(batch_size=3)

    def test_tensors_loop(self, date_tensors):
        batch_size = 4
        for idx, bch in enumerate(date_tensors(bsz=batch_size)):
            assert len(bch) == batch_size

    def test_tensors_zip(self, date_tensors):
        batch_size = 4
        ti = tv = tc = kn = tg = fake.clone()
        date_tensors_1 = DateTensors(ti, tv, tc, kn, tg)
        for idx, bch in enumerate(
            zip(date_tensors(bsz=batch_size), date_tensors_1(bsz=batch_size))
        ):
            assert len(bch) == 2
            assert len(bch[0]) == batch_size
            assert len(bch[1]) == batch_size

    def test_tensors_eq(self, date_tensors):
        ti = tv = tc = kn = tg = base.clone()
        date_tensors_1 = DateTensors(ti, tv, tc, kn, tg)
        date_tensors_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_tensors_1 == date_tensors_2

        date_tensors_1 = DateTensors(fake, tv, tc, kn, tg)
        date_tensors_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_tensors_1 != date_tensors_2

        date_tensors_1 = DateTensors(ti, fake, tc, kn, tg)
        date_tensors_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_tensors_1 != date_tensors_2

        date_tensors_1 = DateTensors(ti, tv, fake, kn, tg)
        date_tensors_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_tensors_1 != date_tensors_2

        date_tensors_1 = DateTensors(ti, tv, tc, fake, tg)
        date_tensors_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_tensors_1 != date_tensors_2

        date_tensors_1 = DateTensors(ti, tv, tc, kn, fake)
        date_tensors_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_tensors_1 != date_tensors_2

        date_tensors_1 = DateTensors(ti, None, tc, kn, tg)
        date_tensors_2 = DateTensors(ti, None, tc, kn, tg)
        assert date_tensors_1 == date_tensors_2

        date_tensors_1 = DateTensors(ti, tv, None, kn, tg)
        date_tensors_2 = DateTensors(ti, tv, None, kn, tg)
        assert date_tensors_1 == date_tensors_2

        date_tensors_1 = DateTensors(ti, tv, tc, None, tg)
        date_tensors_2 = DateTensors(ti, tv, tc, None, tg)
        assert date_tensors_1 == date_tensors_2

        date_tensors_1 = DateTensors(ti, tv, tc, kn, None)
        date_tensors_2 = DateTensors(ti, tv, tc, kn, None)
        assert date_tensors_1 == date_tensors_2

    def test_numpys_next(self, date_numpys):
        def _batch_check(batch_size):
            nmp = next(date_numpys(bsz=batch_size))
            assert (nmp.ti[:batch_size] == base[:batch_size]).all()
            assert (nmp.tv[:batch_size] == base[:batch_size]).all()
            assert (nmp.tc[:batch_size] == base[:batch_size]).all()
            assert (nmp.kn[:batch_size] == base[:batch_size]).all()
            assert (nmp.tg[:batch_size] == base[:batch_size]).all()

        _batch_check(batch_size=1)
        _batch_check(batch_size=3)

    def test_numpys_loop(self, date_numpys):
        batch_size = 4
        for idx, bch in enumerate(date_numpys(bsz=batch_size)):
            assert len(bch) == batch_size

    def test_numpys_eq(self, date_tensors):
        ti = tv = tc = kn = tg = base2.copy()
        date_numpys_1 = DateTensors(ti, tv, tc, kn, tg)
        date_numpys_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_numpys_1 == date_numpys_2

        date_numpys_1 = DateTensors(fake2, tv, tc, kn, tg)
        date_numpys_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_numpys_1 != date_numpys_2

        date_numpys_1 = DateTensors(ti, fake2, tc, kn, tg)
        date_numpys_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_numpys_1 != date_numpys_2

        date_numpys_1 = DateTensors(ti, tv, fake2, kn, tg)
        date_numpys_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_numpys_1 != date_numpys_2

        date_numpys_1 = DateTensors(ti, tv, tc, fake2, tg)
        date_numpys_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_numpys_1 != date_numpys_2

        date_numpys_1 = DateTensors(ti, tv, tc, kn, fake2)
        date_numpys_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_numpys_1 != date_numpys_2

        date_numpys_1 = DateTensors(ti, None, tc, kn, tg)
        date_numpys_2 = DateTensors(ti, None, tc, kn, tg)
        assert date_numpys_1 == date_numpys_2

        date_numpys_1 = DateTensors(ti, tv, None, kn, tg)
        date_numpys_2 = DateTensors(ti, tv, None, kn, tg)
        assert date_numpys_1 == date_numpys_2

        date_numpys_1 = DateTensors(ti, tv, tc, None, tg)
        date_numpys_2 = DateTensors(ti, tv, tc, None, tg)
        assert date_numpys_1 == date_numpys_2

        date_numpys_1 = DateTensors(ti, tv, tc, kn, None)
        date_numpys_2 = DateTensors(ti, tv, tc, kn, None)
        assert date_numpys_1 == date_numpys_2

        date_numpys_1 = DateTensors(ti, tv, tc, kn, None)
        date_numpys_2 = DateTensors(ti, tv, tc, kn, tg)
        assert date_numpys_1 != date_numpys_2

    def test_numpys_zip(self, date_numpys):
        batch_size = 4
        ti = tv = tc = kn = tg = fake2.copy()
        date_numpys_1 = DateTensors(ti, tv, tc, kn, tg)
        for idx, bch in enumerate(
            zip(date_numpys(bsz=batch_size), date_numpys_1(bsz=batch_size))
        ):
            assert len(bch) == 2
            assert len(bch[0]) == batch_size
            assert len(bch[1]) == batch_size


class TestDatesetToy(object):
    @pytest.fixture
    def toydata(self):
        W, Dout = 14, 1
        toydataset = DatesetToy(Dout, W, "cycle", device=torch.device("cuda:0"))
        return toydataset

    def test_cyclic(self):
        data_type = "cyclic"
        W, Dout = 14, 1
        toydataset = DatesetToy(Dout, W, data_type, device=torch.device("cuda:0"))
        trainset = toydataset.create_trainset(s="2016-01-01", e="2017-12-31", freq="D")
        assert toydataset.trainset == trainset

        testset = toydataset.create_testset(by="2019-12-31")
        assert toydataset.testset == testset

        restored = to_date(trainset.ti[:, 0, 0].cpu().numpy())
        original = pandas.date_range("2016-01-01", "2017-12-31", freq="D")
        assert (toydataset.date_series.date_index == original).all()
        assert (toydataset.date_series.date_index[: -W + 1] == restored).all()

    def test_trend(self):
        data_type = "trend"
        W, Dout = 14, 1
        toydataset = DatesetToy(Dout, W, data_type, device=torch.device("cuda:0"))
        trainset = toydataset.create_trainset(s="2016-01-01", e="2017-12-31", freq="D")
        assert toydataset.trainset == trainset

        testset = toydataset.create_testset(by="2019-12-31")
        assert toydataset.testset == testset

        restored = to_date(trainset.ti[:, 0, 0].cpu().numpy())
        original = pandas.date_range("2016-01-01", "2017-12-31", freq="D")
        assert (toydataset.date_series.date_index == original).all()
        assert (toydataset.date_series.date_index[: -W + 1] == restored).all()
