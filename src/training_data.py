import datetime
from typing import List, Iterable

from sample import Sample
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

    def load(self, raw_data_source: Iterable[dict[str, str]]) -> None:
        """Load and partition the raw data"""
        for n, row in enumerate(raw_data_source):
            # sample = Sample(
            #     sepal_length=float(row['sepal_length']),
            #     sepal_width=float(row['sepal_width']),
            #     petal_length=float(row['petal_length']),
            #     petal_width=float(row['petal_width']),
            #     species=row['species']
            # )

            # if n % 5:
            #     self.training.append(sample)
            # else:
            #     self.testing.append(sample)
            pass

        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter, sample: Sample) -> None:
        """Test this hyperparameter."""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, sample: Sample) -> Sample:
        """Classify this sample."""
        return sample.classify(parameter.classify(sample))
