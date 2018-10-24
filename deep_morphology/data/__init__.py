#! /usr/bin/env python

from deep_morphology.data.seq2seq_data import Seq2seqDataset, UnlabeledSeq2seqDataset
from deep_morphology.data.seq2seq_data import InflectionDataset, UnlabeledInflectionDataset
from deep_morphology.data.seq2seq_data import GlobalPaddingInflectionDataset, UnlabeledGlobalPaddingInflectionDataset
from deep_morphology.data.reinflection_data import ReinflectionDataset, UnlabeledReinflectionDataset
from deep_morphology.data.tagging_data import TaggingDataset, UnlabeledTaggingDataset
from deep_morphology.data.classification_data import ClassificationDataset, UnlabeledClassificationDataset
from deep_morphology.data.classification_data import NoSpaceClassificationDataset, UnlabeledNoSpaceClassificationDataset
from deep_morphology.data.seq2seq_data import GlobalPaddingSeq2seqDataset, UnlabeledGlobalPaddingSeq2seqDataset
