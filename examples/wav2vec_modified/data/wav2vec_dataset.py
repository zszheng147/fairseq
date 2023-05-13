# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import io

from typing import Any, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset
# from fairseq.data.audio import RawAudioBackupDataset
from fairseq.data.data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


logger = logging.getLogger(__name__)


def load_fbank(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    fnames, sizes = [], []
    with open(manifest_path) as f:
        root_dir = f.readline().strip() 
        for line in f.readlines():   
            items = line.strip().split("\t") 
            sz = int(items[1])   
            if min_keep is not None and sz < min_keep / 160:  
                n_short += 1
            elif max_keep is not None and sz > max_keep / 160: 
                n_long += 1
            else:
                fnames.append(items[0])  
                sizes.append(sz)
    if min_keep==None:
        logger.info(
            (
                f"max_keep={max_keep}, min_keep={min_keep}, "
                f"loaded {len(fnames)}, skipped {n_short} short and {n_long} long, "
                f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
            )
        )
    else:
        logger.info(
            (
                f"max_keep={max_keep}, min_keep={min_keep/160}, "
                f"loaded {len(fnames)}, skipped {n_short} short and {n_long} long, "
                f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
            )
        )  
    return root_dir, fnames, sizes

class Wav2vecFbankDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        max_sample_size = None,
        min_sample_size = 0,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad = False,
        normalize = False,
        num_buckets=0,
        compute_mask_indices=False,
        text_compression_level=TextCompressionLevel.none,
        stats_npz_path=None,
        **mask_compute_kwargs,
    ):
        
        self.root_dir, self.fnames, self.sizes = load_fbank(  #change
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )  

        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.pad = pad
        self.normalize = normalize
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.compute_mask_indices = compute_mask_indices
        if self.compute_mask_indices:
            self.mask_compute_kwargs = mask_compute_kwargs
            self._features_size_map = {}
            self._C = mask_compute_kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])
        
        self.text_compressor = TextCompressor(level=text_compression_level)

        skipped = 0
        self.skipped_indices = set()

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.set_bucket_info(num_buckets)

        # add global cmvn
        #原始路径 /mnt/lustre/sjtu/home/xc915/data/Gigaspeech/gigaspeech-s-rawwav/global_cmvn.npy “两个是一样的”
        assert stats_npz_path is not None, "stats_npz_path is None"
        stats = np.load(stats_npz_path, allow_pickle=True).tolist()
        self.mean, self.std = stats["mean"], stats["std"]  #mean(80,), std(80,)
    
    def __getitem__(self, index):
        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        # fn = self.text_compressor.decompress(fn)
        path_or_fp = os.path.join(self.root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)      

        feats = np.load(path_or_fp, allow_pickle=True)  # Seq_len * 80
        feats = torch.from_numpy(feats).float()  # Seq_len * 80

        # This is important! 
        feats = np.subtract(feats, self.mean) 
        feats = np.divide(feats, self.std)  

        return {"id": index, "source": feats}

    def __len__(self):
        return len(self.sizes)
    
    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def _compute_mask_indices(self, dims, padding_mask):
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks=2,
                no_overlap=self.mask_compute_kwargs["no_mask_overlap"],
                min_space=self.mask_compute_kwargs["mask_min_space"],
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap=self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space=self.mask_compute_kwargs["mask_channel_min_space"],
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices


        # import soundfile as sf

        # fn = self.fnames[index]
        # fn = fn if isinstance(self.fnames, list) else fn.as_py()
        # fn = self.text_compressor.decompress(fn)
        # path_or_fp = os.path.join(self.root_dir, fn)
        # _path, slice_ptr = parse_path(path_or_fp)
        # if len(slice_ptr) == 2:
        #     byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        #     assert is_sf_audio_data(byte_data)
        #     path_or_fp = io.BytesIO(byte_data)

        # wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        # feats = torch.from_numpy(wav).float()
        # feats = self.postprocess(feats, curr_sample_rate)
        # return {"id": index, "source": feats}

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size, 80)
        # 80 dim for Fbank features

        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )

        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:, :-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped,
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )