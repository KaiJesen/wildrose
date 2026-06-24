"""K 线数据集（兼容入口）。"""

from transformer_kit.segment_dataset import (
    BarSeriesBundle,
    BarWindowDataset,
    PatternSequenceDataset,
    SequenceSampleIndex,
    VariableSegmentDataset,
    build_sequence_sample_indices,
    make_synthetic_ohlcv,
    prepare_bar_series,
    time_ordered_split,
)

__all__ = [
    "BarSeriesBundle",
    "BarWindowDataset",
    "PatternSequenceDataset",
    "SequenceSampleIndex",
    "VariableSegmentDataset",
    "build_sequence_sample_indices",
    "make_synthetic_ohlcv",
    "prepare_bar_series",
    "time_ordered_split",
]
