import datetime
from typing import List, Iterable
from pathlib import Path
import csv

from sample import Sample, TestingKnownSample, TrainingKnownSample, InvalidSampleError
from hyperparameter import Hyperparameter


class TrainingData:
    """A set of training data and testing data with methods to load and test the samples."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: List[Sample] = []
        self.testing: List[Sample] = []
        self.tuning: List[Hyperparameter] = []

    def load(self, raw_data_iter: Iterable[dict[str, str]]) -> None:
        """Load and partition the raw data"""
        bad_count = 0
        for n, row in enumerate(raw_data_iter):
            try:
                if n % 5 == 0:
                    test = TestingKnownSample.from_dict(row)
                    self.testing.append(test)
                else:
                    train = TrainingKnownSample.from_dict(row)
                    self.training.append(train)
            except InvalidSampleError as ex:
                print(f"Row {n+1}: {ex}")
                bad_count += 1
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter, sample: Sample) -> None:
        """Test this hyperparameter."""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, sample: Sample) -> Sample:
        """Classify this sample."""
        return sample.classify(parameter.classify(sample))


class SampleReader:
    """
    See iris.names for attribute ordering in bezdekIris.data file
    """

    target_class = Sample
    header = ['sepal_length', 'sepal_width',
              'petal_length', 'petal_width', 'class']

    def __init__(self, source: Path) -> None:
        self.source = source

    def sample_iter(self) -> Iterable[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length=float(row['sepal_length']),
                        sepal_width=float(row['sepal_width']),
                        petal_length=float(row['petal_length']),
                        petal_width=float(row['petal_width']),
                        species=row['class'])
                except ValueError as ex:
                    raise BadSampleRow(
                        f"invalid data in {row!r}") from ex

                yield sample


class BadSampleRow(ValueError):
    pass
