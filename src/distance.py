from math import hypot

from sample import Sample


class Distance:
    """Definition of a distance computation"""

    def distance(self, s1: Sample, s2: Sample) -> float:
        """Compute the distance between two samples"""
        pass


class EuclideanDistance(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width,
        )


class ManhattanDistance(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            abs(s1.sepal_length - s2.sepal_length),
            abs(s1.sepal_width - s2.sepal_width),
            abs(s1.petal_length - s2.petal_length),
            abs(s1.petal_width - s2.petal_width),
        )


class ChebyshevDistance(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            abs(s1.sepal_length - s2.sepal_length),
            abs(s1.sepal_width - s2.sepal_width),
            abs(s1.petal_length - s2.petal_length),
            abs(s1.petal_width - s2.petal_width),
        )


class SorensonDistance(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            abs(s1.sepal_length - s2.sepal_length),
            abs(s1.sepal_width - s2.sepal_width),
            abs(s1.petal_length - s2.petal_length),
            abs(s1.petal_width - s2.petal_width),
        ) / sum(
            s1.sepal_length + s2.sepal_length,
            s1.sepal_width + s2.sepal_width,
            s1.petal_length + s2.petal_length,
            s1.petal_width + s2.petal_width,
        )
