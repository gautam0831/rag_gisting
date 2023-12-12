"""Combined Alpaca and Self-Instruct dataset."""


import json

import datasets
from datasets.splits import NamedSplit

from datasets import load_dataset

import random


logger = datasets.logging.get_logger(__name__)


class WikipediaConfig(datasets.BuilderConfig):
    def __init__(
        self,
        *args,
        train_file=None,
        validation_seen_file=None,
        validation_unseen_file=None,
        validation_human_file=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_file: str = train_file
        self.validation_seen_file: str = validation_seen_file
        self.validation_unseen_file: str = validation_unseen_file
        self.validation_human_file: str = validation_human_file


class Wikipedia(datasets.GeneratorBasedBuilder):
    """Wikipedia Dataset."""

    VERSION = datasets.Version("1.0.1")
    BUILDER_CONFIG_CLASS = WikipediaConfig
    BUILDER_CONFIGS = [
        WikipediaConfig(
            name="wikipedia",
            train_file="./data/alpaca_plus/alpaca_plus_train.json",
            validation_seen_file="./data/alpaca_plus/alpaca_plus_validation_seen.json",
            validation_unseen_file="./data/alpaca_plus/alpaca_plus_validation_unseen.json",  # noqa
            validation_human_file="./data/alpaca_plus/alpaca_plus_validation_human.json",  # noqa
            description="Default config for Alpaca",
        ),
    ]
    DEFAULT_CONFIG_NAME = "wikipedia"

    dataset = load_dataset("trivia_qa", "rc.wikipedia")
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']
    
    def _info(self):
        return datasets.DatasetInfo(
            description="This is the Trivia QA dataset sourced from Hugging Face.",
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "split": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        del dl_manager
        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name="validation",
                gen_kwargs={
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_seen"),
                gen_kwargs={
                    "split": "validation_seen",
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_human"),
                gen_kwargs={
                    "split": "validation_human",
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_unseen"),
                gen_kwargs={
                    "split": "validation_unseen",
                },
            ),
        ]

    def _generate_examples(
        self,
        split: str,
    ):
        """Yields examples."""
        if "validation" in split:
            split = "validation"
        for i, example in enumerate(self.dataset[split]):
            yield i, {
                "context": example['entity_pages']['wiki_context'][0][:200],
                "input": example["question"],
                "output": example['answer']['normalized_value'],
                "source": "wikipedia",
                "split": split,
            }


