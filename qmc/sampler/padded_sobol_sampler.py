class PaddedSobolSampler:
    def __init__(self, max_dim=64, replicate=0):
        ...

    def start(self, px, py, sample_index):
        """Call once per path, before tracing begins."""
        ...

    def next_1d(self) -> float:
        """Consume one dimension, return value in [0,1)."""
        ...

    def next_2d(self) -> tuple[float, float]:
        """Consume two dimensions, return (u,v) in [0,1)^2."""
        ...