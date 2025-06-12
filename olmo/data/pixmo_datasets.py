import logging
import re
import shutil
from os.path import join, exists
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import glob

import datasets
import numpy as np

from olmo.data.dataset import DATA_HOME, Dataset
from olmo.data.download_urls import download_pixmo_urls, filter_and_group_data

if DATA_HOME is not None:
    PIXMO_DATASETS = join(DATA_HOME, "pixmo_datasets")
else:
    PIXMO_DATASETS = None
"""Where to save local version of the data after URLs filtering"""

VERIFY = True
"""Verify SSL certificates when downloading"""


NO_POINT_PREFIX = [
    "No pointing: ",
    "No pointing: ",
    "no pointing:\n",
    "No pointing:\n",
    "Not pointing:\n",
    "No Points: ",
    "No Points: ",
    "NO POINTING\n",
    "No pontiing\n",
    "No Points:\n ",
    "No pointing\n",
    "Do not point. ",
    "Refrain from pointing. ",
    "Avoid generating points . ",
    "For this question, do not use points. ",
    "Refrain from using points:\n",
    "Don't include points in your response. ",
    "Don't point. ",
    "Don't use points. ",
    "Please don't use points.\n\n",
    "Please don't use points.\n\n",
    "Respond without using points. ",
    "Respond without pointing:\n",
    "Do not generate ponits: ",
    "Do not point. ",
    "Do not point\n",
    "no pointing\n\n",
    "Answer without points: ",
    "Answer this question without pointing: ",
    "Answer without poiints. ",
    "answer without points: ",
    "answer with text only, do not points\n"
]
"""No-pointing requests templates, used for preprocessing"""


def save_local_dataset(dataset: datasets.Dataset, name: str, n_procs, n_val=None):
    if len(dataset) == 0:
        raise ValueError("Given an empty dataset")
    if n_val:
        split = dataset.train_test_split(test_size=n_val, seed=96817)
        dataset = datasets.DatasetDict(train=split["train"], validation=split["test"])
    logging.info("Preparing local dataset...")
    if exists(name):
        logging.info(f"{name} already exists, it will be removed")
        shutil.rmtree(name)
    dataset.save_to_disk(name, num_proc=n_procs)
    logging.info("Done")


class PixMoCount(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=1024, cache_only=False):
        local_name = join(PIXMO_DATASETS, "count")
        if exists(local_name):
            return
        all_data = datasets.DatasetDict()
        for split in ["validation", "test", "train"]:
            ds = datasets.load_dataset("allenai/pixmo-count", split=split)
            url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=False)
            ds = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
            ds = ds.add_column("image", [url_to_filename[x] for x in ds["image_url"]])
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, sample=None, counting=False, keep_in_memory=False):
        self.dataset = datasets.load_from_disk(join(PIXMO_DATASETS, "count"), keep_in_memory=keep_in_memory)[split]
        self.counting = counting
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        example = self.dataset[item]
        out = dict(
            style="point_count" if self.counting else "pointing",
            image=example["image"],
            label=example["label"],
            metadata=dict(
                image_url=example["image_url"],
                count=example["count"],
            )
        )
        if self.split == "train":
            points = example["points"]
            out["points"] = np.stack([points["x"], points["y"]], -1, dtype=np.float32)
        return out


class PixMoDocs(Dataset):
    V1_STYLE = {
        "pixmo_docs_other": "scifi_document",
        "pixmo_docs_charts": "scifi_charts",
        "pixmo_docs_diagrams": "scifi_diagram",
        "pixmo_docs_tables": "scifi_table"
    }

    @classmethod
    def download(cls, n_procs=1):
        for name in ["other", "charts", "diagrams", "tables"]:
            datasets.load_dataset_builder("allenai/pixmo-docs", name=name).download_and_prepare()

    def __init__(self, doc_type, split, sample=None, keep_in_memory=False, v1_style=False):
        assert doc_type in ["other", "charts", "diagrams", "tables"]
        assert split in ["train", "validation", "test"]
        self.doc_type = doc_type
        self.v1_style = v1_style
        self.dataset = datasets.load_dataset(
            "allenai/pixmo-docs", name=doc_type, split=split, keep_in_memory=keep_in_memory)

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        style = f"pixmo_docs_{self.doc_type}"
        if self.v1_style:
            style = self.V1_STYLE[style]
        example = self.dataset[item]
        qas = example["questions"]
        return dict(
            image=example["image"],
            message_list=[
                dict(question=q, answer=a, style=style) for q, a in
                zip(qas["question"], qas["answer"])
            ],
            metadata=dict(
                image_id=example["image_id"]
            )
        )


class PixMoPoints(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False, hold_out_pointing_eval=True):
        collection_method = ["pointing", "counting"]
        local_names = [join(PIXMO_DATASETS, f"points-{name}") for name in collection_method]
        if all(exists(x) for x in local_names):
            return
        ds = datasets.load_dataset("allenai/pixmo-points", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        if hold_out_pointing_eval:
            eval_ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
            for url in eval_ds["image_url"]:
                if url in filenames:
                    del filenames[url]
        for method, local_name in zip(collection_method, local_names):
            logging.info(f"Building subset {method}")
            ds_for_method = ds.filter(lambda x: x == method, input_columns="collection_method")
            filtered_dataset = filter_and_group_data(ds_for_method, filenames, check_sha)
            name = "high_frequency" if method == "counting" else "basic"
            save_local_dataset(filtered_dataset, local_name, n_procs=n_procs, n_val=n_val)

    def __init__(self, split, kind="both", counting=False, keep_in_memory=False):
        if kind not in ["high_frequency", "basic", "both"]:
            raise ValueError(kind)
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        mode = "pointing" if counting else "point_count"
        self.split = split
        self.kind = kind
        self.mode = mode
        if kind == "both":
            data1 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-counting"), keep_in_memory=keep_in_memory)[split]
            data2 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-pointing"), keep_in_memory=keep_in_memory)[split]
            self.data = datasets.concatenate_datasets([data1, data2])
        elif kind == "basic":
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-pointing"), keep_in_memory=keep_in_memory)[split]
        else:
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-counting"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = []
        for label, points in zip(ex["label"], ex["points"]):
            messages.append(dict(
                label=label,
                points=np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1),
                point_scale=100,
                style=self.mode
            ))
        # print("DEBUG: Messages:", messages)
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_url=ex["image_url"],
            )
        )


class PixMoPointExplanations(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=1024, cache_only=False):
        local_name = join(PIXMO_DATASETS, "point-explanations")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-point-explanations", split="train")
        ds = ds.filter(lambda x: x is not None, input_columns=["parsed_response"])
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, split_groups=True, keep_in_memory=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.split_groups = split_groups
        data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "point-explanations"),
            keep_in_memory=keep_in_memory)[split]
        out = []
        for ex in data:
            molmo_ex = dict(
                image=ex["image"],
                metadata=dict(
                    image_url=ex["image_url"],
                )
            )
            msg_list = []
            for q, res, alt, inline, points in zip(
                ex["question"], ex["parsed_response"],
                ex["alt_text"], ex["inline_text"], ex["points"]
            ):
                msg_list.append(dict(
                    question=q,
                    answer=res,
                    answer_annotations=[dict(
                        points=p, inline_text=i, alt_text=a
                    ) for p, i, a in zip(points, inline, alt)],
                    style="point_qa"
                ))
            if self.split_groups and len(msg_list) > 1:
                n = len(msg_list) // 2 + len(msg_list) % 2
                out.append(dict(molmo_ex, message_list=msg_list[:n]))
                out.append(dict(molmo_ex, message_list=msg_list[n:]))
            else:
                out.append(dict(molmo_ex, message_list=msg_list))
        self.data = out

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        return dict(self.data[item])


class PixMoCapQa(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False):
        local_name = join(PIXMO_DATASETS, "cap-qa")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-cap-qa", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, prefix_how_many=True, keep_in_memory=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cap-qa"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = self.data[item]
        messages = [dict(messages=msg, style="synthetic_qa") for msg in example["messages"]]

        ex = dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(
                image_url=example["image_url"],
            )
        )

        if self.prefix_how_many:
            for conv in ex["message_list"]:
                messages = conv["messages"]
                for user_question_ix in range(0, len(messages), 2):
                    if re.fullmatch("how many.*", messages[user_question_ix].lower()):
                        prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                        messages[user_question_ix] = prefix + messages[0]
        return ex


class PixMoCap(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False, sample=None):
        local_name = join(PIXMO_DATASETS, "cap")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-cap", split="train")
        if sample:
            ds = ds.take(sample)
        url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        logging.info("Preparing data...")
        filtered_dataset = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
        filtered_dataset = filtered_dataset.add_column(
            "image", [url_to_filename[x] for x in filtered_dataset["image_url"]])
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, mode, prefix_how_many=True, keep_in_memory=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        if mode not in ["transcripts", "captions", "transcript_and_caption", "transcript1_and_caption"]:
            raise ValueError(mode)
        self.split = split
        self.mode = mode
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cap"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = []
        caption = ex.pop("caption")
        transcripts = ex.pop("transcripts")
        if self.mode in ["captions", "transcript_and_caption", "transcript1_and_caption"]:
            messages.append(dict(text=caption, style="long_caption"))
        if self.mode in ["transcript_and_caption", "transcript1_and_caption"]:
            if self.mode == "transcript_and_caption":
                ix = rng.randint(0, len(transcripts))
            else:
                ix = 0
            messages.append(dict(text=transcripts[ix], style="transcript"))
        if self.mode == "transcripts":
            messages += [dict(text=tr, style="transcript") for tr in transcripts]
        out = dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_url=ex.pop("image_url"),
            )
        )
        return out


class PixMoAskModelAnything(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False):
        local_name = join(PIXMO_DATASETS, "ask-model-anything")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-ask-model-anything", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, prefix_how_many=True, keep_in_memory=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "ask-model-anything"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = self.data[item]
        messages = []
        for q, a in zip(example["question"], example["answer"]):
            messages.append(dict(
                question=q,
                answer=a,
                style="user_qa"
            ))

        ex = dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(
                image_url=example["image_url"],
            )
        )

        if self.prefix_how_many:
            for conv in ex["message_list"]:
                if re.fullmatch("how many.*", conv["question"].lower()):
                    prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                    conv["question"] = prefix + conv["question"]
        return ex


class PixMoPointsEval(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, cache_only=False):
        local_name = join(PIXMO_DATASETS, "pixmo-points-eval")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
        url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        ds = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
        ds = ds.add_column("image", [url_to_filename[x] for x in ds["image_url"]])
        save_local_dataset(ds, local_name, n_procs)

    def __init__(self, keep_in_memory=False):
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "pixmo-points-eval"), keep_in_memory=keep_in_memory)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        points = ex["points"]
        messages = []
        points = np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1)
        return dict(
            image=ex["image"],
            label=ex["label"],
            points=points,
            point_scale=100,
            style="pointing",
            metadata=dict(
                points=points,
                masks=np.array(ex["masks"], dtype=bool),
                image_url=ex["image_url"],
            )
        )


class TokenImageDataset(Dataset):
    """Dataset for training on token images with BOS token input."""
    
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=1024, cache_only=False):
        """Download and prepare token images dataset."""
        local_name = join(os.environ.get('MOLMO_DATA_DIR', ''), "token_images")
        if exists(local_name):
            return
            
        # Create directory if it doesn't exist
        Path(local_name).mkdir(parents=True, exist_ok=True)
        
        # Generate token images if they don't exist
        from scripts.generate_token_images import main as generate_token_images
        generate_token_images()
    
    def __init__(self, split, config=None, keep_in_memory=False, **kwargs):
        """Initialize dataset.
        
        Args:
            split: Dataset split ('train' or 'validation')
            config: Full TrainConfig object containing all configuration
            keep_in_memory: Whether to keep dataset in memory
            **kwargs: Additional arguments that will be set as attributes
        """
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
            
        self.split = split
        self.config = config
        
        # Get values from extra_args if provided, otherwise use defaults
        extra_args = getattr(config.data, 'extra_args', {}) if config else {}
        extra_args = extra_args or {}  # Convert None to empty dict
        self.training_dataset_size_debug = extra_args.get('training_dataset_size_debug', -1)
        self.seed = config.data.seed if config else 42
        self.prompt_type = extra_args.get('prompt_type', 'empty')
        
        # Update with any provided kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Ensure dataset is downloaded
        self.download()
        
        # Get list of image files
        local_name = join(os.environ.get('MOLMO_DATA_DIR', ''), "token_images")
        self.image_files = sorted(glob.glob(join(local_name, "token_*.png")))
        if not self.image_files:
            raise ValueError(f"No token images found in {local_name}")
            
        # Create RNG and shuffle files
        rng = np.random.RandomState(self.seed)
        self.image_files = np.array(self.image_files)  # Convert to numpy array for easier indexing
        rng.shuffle(self.image_files)
        
        # Split into train/validation
        split_idx = int(len(self.image_files) * 0.9)  # 90% for training
        self.image_files = self.image_files[:split_idx] if split == "train" else self.image_files[split_idx:]

        # Limit dataset size if in debug mode and it's the training split
        if self.training_dataset_size_debug > 0 and split == "train":
            rng.shuffle(self.image_files)  # Shuffle again to get random subset
            self.image_files = self.image_files[:self.training_dataset_size_debug]
            logging.info(f"Debug mode: Limited training dataset to {self.training_dataset_size_debug} examples")
        
        # Load tokenizer for decoding token IDs
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
    
    def __len__(self):
        return len(self.image_files)
    
    def get(self, item, rng):
        """Get a dataset item.
        
        Args:
            item: Index of item to get
            rng: Random number generator for reproducibility
            
        Returns:
            Dictionary containing:
                image: Path to image file
                message_list: List of messages with token information
                metadata: Additional metadata
        """
        image_path = self.image_files[item]
        
        # Get token ID from filename
        token_id = int(Path(image_path).stem.split('_')[1])
        
        # Decode token ID to get actual token text, ensuring we only get the token itself
        # Use encode then decode to ensure we only get the single token
        token_text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        # assert self.tokenizer.encode(token_text, add_special_tokens=False) == [token_id]

        # check cycle-consistency of the tokenizer
        # print(f"CYCLE-CONSISTENT TOKEN ID: {token_id}")
        # print(f"CYCLE-CONSISTENT TOKEN TEXT: {token_text}")
        # print(f"CYCLE-CONSISTENT TOKEN IDS FROM TEXT: {self.tokenizer.encode(token_text)}")
        if self.prompt_type == 'empty':
            prompt = ''
        elif self.prompt_type == 'caption':
            prompt = 'Output the token shown in the image:'
        elif self.prompt_type == 'copy':
            prompt = 'Copy the token shown in the image:'

        return {
            'image': image_path,
            'message_list': [{
                'style': 'none',
                'prompt': prompt,
                'answer': token_id,
                'token_id': token_id
            }],
            'metadata': {
                'image_path': image_path,
                'token_id': token_id,
                'token_text': token_text
            }
        }
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        batch_entry = {}
        
        # Keep image paths as a list
        batch_entry['images'] = [entry['image'] for entry in batch]
        
        # Keep message lists
        batch_entry['message_list'] = [entry['message_list'] for entry in batch]
        
        return batch_entry

import os
import glob
import logging
from pathlib import Path
from os.path import join, exists

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ColorImageDataset(Dataset):
    """Dataset for training on color images with BOS token input."""
    
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=1024, cache_only=False):
        """Download and prepare color images dataset."""
        local_name = join(os.environ.get('MOLMO_DATA_DIR', ''), "color_images")
        if exists(local_name):
            return
        
        Path(local_name).mkdir(parents=True, exist_ok=True)
        from scripts.generate_color_images import main as generate_color_images
        generate_color_images()

    def __init__(self, split, config=None, keep_in_memory=False, **kwargs):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        
        self.split = split
        self.config = config
        extra_args = getattr(config.data, 'extra_args', {}) if config else {}
        extra_args = extra_args or {}
        self.training_dataset_size_debug = extra_args.get('training_dataset_size_debug', -1)
        self.seed = config.data.seed if config else 42
        self.prompt_type = extra_args.get('prompt_type', 'empty')
        
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.download()

        local_name = join(os.environ.get('MOLMO_DATA_DIR', ''), "color_images")
        self.image_files = sorted(glob.glob(join(local_name, "color_*.png")))
        if not self.image_files:
            raise ValueError(f"No color images found in {local_name}")

        rng = np.random.RandomState(self.seed)
        self.image_files = np.array(self.image_files)
        rng.shuffle(self.image_files)

        split_idx = int(len(self.image_files) * 0.9)
        self.image_files = self.image_files[:split_idx] if split == "train" else self.image_files[split_idx:]

        if self.training_dataset_size_debug > 0 and split == "train":
            rng.shuffle(self.image_files)
            self.image_files = self.image_files[:self.training_dataset_size_debug]
            logging.info(f"Debug mode: Limited training dataset to {self.training_dataset_size_debug} examples")

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
        print(f"LENGTH OF DATASET: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def get(self, item, rng):
        image_path = self.image_files[item]
        filename = Path(image_path).stem  # e.g., color_123_red
        _, token_id_str, *color_parts = filename.split('_')
        token_id = int(token_id_str)
        color_name = "_".join(color_parts)
        print(f"COLOR NAME: {color_name}")

        token_text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        if self.prompt_type == 'empty':
            prompt = ''
        elif self.prompt_type == 'caption':
            prompt = 'Output the color shown in the image:'
        elif self.prompt_type == 'copy':
            prompt = 'Copy the color name shown in the image:'

        return {
            'image': image_path,
            'message_list': [{
                'style': 'none',
                'prompt': prompt,
                'answer': token_id,
                'token_id': token_id
            }],
            'metadata': {
                'image_path': image_path,
                'token_id': token_id,
                'token_text': token_text,
                'color_name': color_name
            }
        }

    def collate_fn(self, batch):
        batch_entry = {
            'images': [entry['image'] for entry in batch],
            'message_list': [entry['message_list'] for entry in batch],
        }
        return batch_entry