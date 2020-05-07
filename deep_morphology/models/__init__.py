from deep_morphology.models.base import BaseModel
from deep_morphology.models.cnn_seq2seq import CNNSeq2seq
from deep_morphology.models.hard_monotonic_attention import HardMonotonicAttentionSeq2seq
from deep_morphology.models.luong_attention import LuongAttentionSeq2seq
from deep_morphology.models.lstm_tagger import LSTMTagger
from deep_morphology.models.seq2seq import Seq2seq, AttentionOnlySeq2seq, VanillaSeq2seq
from deep_morphology.models.sequence_classifier import SequenceClassifier, MidSequenceClassifier, CNNSequenceClassifier, LSTMPermuteProber, RandomLSTMProber
from deep_morphology.models.sequence_classifier import PairSequenceClassifier, PairCNNSequenceClassifier
from deep_morphology.models.sopa_classifier import SopaClassifier, MultiLayerSopaClassifier
from deep_morphology.models.sopa_seq2seq import SopaSeq2seq
from deep_morphology.models.reinflection_seq2seq import ReinflectionSeq2seq
from deep_morphology.models.test_packed_sequence import TestPackedSeq2seq
from deep_morphology.models.contextual_embedding_classifier import BERTClassifier
#from deep_morphology.models.contextual_embedding_classifier import ELMOClassifier
#from deep_morphology.models.contextual_embedding_classifier import ELMOPairClassifier
from deep_morphology.models.contextual_embedding_classifier import BERTPairClassifier
from deep_morphology.models.contextual_embedding_classifier import EmbeddingClassifier
from deep_morphology.models.contextual_embedding_classifier import EmbeddingPairClassifier
from deep_morphology.models.contextual_embedding_classifier import SentenceRepresentationProber
from deep_morphology.models.contextual_embedding_classifier import TransformerForSequenceClassification
from deep_morphology.models.contextual_embedding_classifier import SentenceTokenPairRepresentationProber
#from deep_morphology.models.elmo_tagger import ELMOTagger
from deep_morphology.models.bert_tagger import BERTTagger
