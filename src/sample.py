from typing import Optional, cast, Set


class Domain(Set[str]):
    def validate(self, value: str) -> None:
        if value not in self:
            raise ValueError(f"{value!r} is not a valid value")


species = Domain(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


class Sample:
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: Optional[str] = None
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species
        self.classification: Optional[str] = None

    def __repr__(self) -> str:
        if self.species is None:
            known_unknown = "UnknownSample"
        else:
            known_unknown = "KnownSample"

        if self.classification is None:
            classification = ""
        else:
            classification = f", {self.classification}"

        return (
            f'{known_unknown}('
            f'sepal_length={self.sepal_length}, '
            f'sepal_width={self.sepal_width}, '
            f'petal_length={self.petal_length}, '
            f'petal_width={self.petal_width}'
            f'species={self.species!r}'
            f'{classification})'
            f')'
        )

    def classify(self, classification: str) -> None:
        self.classification = classification

    def matches(self) -> bool:
        return self.species == self.classification


class InvalidSampleError(ValueError):
    """Source date file has invalid data representation"""


class OutlierError(ValueError):
    """Value lies outside of the expected range"""


class KnownSample(Sample):
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: str
    ) -> None:
        super().__init__(sepal_length, sepal_width, petal_length, petal_width, species)
        self.species = species

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "KnownSample":
        try:
            return cls(
                sepal_length=float(row['sepal_length']),
                sepal_width=float(row['sepal_width']),
                petal_length=float(row['petal_length']),
                petal_width=float(row['petal_width']),
                species=species.validate(row['species'])
            )
        except ValueError as ex:
            raise InvalidSampleError(f"invalid data in {row!r}")

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'sepal_length={self.sepal_length}, '
            f'sepal_width={self.sepal_width}, '
            f'petal_length={self.petal_length}, '
            f'petal_width={self.petal_width}'
            f'species={self.species!r}'
            f')'
        )


class TrainingKnownSample(KnownSample):
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TrainingKnownSample":
        return cast(TrainingKnownSample, super().from_dict(row))


class TestingKnownSample(KnownSample):
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TestingKnownSample":
        return cast(TestingKnownSample, super().from_dict(row))
