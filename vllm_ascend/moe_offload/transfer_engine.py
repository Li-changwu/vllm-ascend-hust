#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass
from typing import Sequence

from vllm_ascend.moe_offload.host_store import ExpertWeightBundle
from vllm_ascend.moe_offload.layout import LayoutValidator
from vllm_ascend.moe_offload.slot_bank import ExpertSlot, SlotState


DEFAULT_EXPERT_TRANSFER_BATCH_SIZE = 8


@dataclass(frozen=True)
class ExpertTransfer:
    bundle: ExpertWeightBundle
    slot: ExpertSlot


@dataclass
class ExpertTransferHandle:
    ready_event: object | None = None
    keepalive: tuple[object, ...] = ()
    children: tuple["ExpertTransferHandle", ...] = ()

    def wait(self) -> None:
        if self.is_empty():
            return

        import torch

        current_stream = torch.npu.current_stream()
        for child in self.children:
            child.wait()
        if self.ready_event is not None:
            current_stream.wait_event(self.ready_event)
        self.keepalive = ()

    def is_empty(self) -> bool:
        return (
            self.ready_event is None
            and not self.keepalive
            and not self.children
        )


class TransferEngine:
    def __init__(self, *, batch_size: int = DEFAULT_EXPERT_TRANSFER_BATCH_SIZE) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = int(batch_size)
        self._transfer_stream = None

    def load_sync(self, bundle: ExpertWeightBundle, slot: ExpertSlot) -> None:
        LayoutValidator.validate_copy_compatible(bundle, slot.as_bundle())
        slot.w13.copy_(bundle.w13)
        slot.w2.copy_(bundle.w2)
        slot.state = SlotState.READY

    def load_batch_sync(self, transfers: Sequence[ExpertTransfer]) -> None:
        for chunk in _chunks(transfers, self.batch_size):
            self._load_chunk_sync(chunk)

    def load_batch_async(self, transfers: Sequence[ExpertTransfer]) -> ExpertTransferHandle:
        if not transfers:
            return ExpertTransferHandle()
        if not _all_npu_slots(transfers):
            self.load_batch_sync(transfers)
            return ExpertTransferHandle()

        import torch

        stream = self._get_transfer_stream()
        keepalive: list[object] = []
        with torch.npu.stream(stream):
            for chunk in _chunks(transfers, self.batch_size):
                keepalive.extend(self._load_chunk_sync(chunk))
            ready_event = stream.record_event()
        return ExpertTransferHandle(
            ready_event=ready_event,
            keepalive=tuple(keepalive),
        )

    def _load_chunk_sync(self, transfers: Sequence[ExpertTransfer]) -> tuple[object, ...]:
        if not transfers:
            return ()
        for transfer in transfers:
            LayoutValidator.validate_copy_compatible(
                transfer.bundle,
                transfer.slot.as_bundle(),
            )

        bundles = [transfer.bundle for transfer in transfers]
        slots = [transfer.slot for transfer in transfers]
        w13_source, w13_keepalive = _bundle_contiguous_view(
            [bundle.w13 for bundle in bundles],
            expected_batch=len(bundles),
        )
        w2_source, w2_keepalive = _bundle_contiguous_view(
            [bundle.w2 for bundle in bundles],
            expected_batch=len(bundles),
        )
        w13_target = _contiguous_slot_view(
            [slot.w13 for slot in slots],
            expected_batch=len(slots),
        )
        w2_target = _contiguous_slot_view(
            [slot.w2 for slot in slots],
            expected_batch=len(slots),
        )

        if (
            w13_source is None
            or w2_source is None
            or w13_target is None
            or w2_target is None
        ):
            for transfer in transfers:
                self.load_sync(transfer.bundle, transfer.slot)
            return ()

        w13_target.copy_(w13_source)
        w2_target.copy_(w2_source)
        for slot in slots:
            slot.state = SlotState.READY
        return (*w13_keepalive, *w2_keepalive)

    def _get_transfer_stream(self):
        if self._transfer_stream is None:
            import torch

            self._transfer_stream = torch.npu.Stream()
        return self._transfer_stream


def _chunks(values: Sequence[ExpertTransfer], size: int) -> list[Sequence[ExpertTransfer]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _bundle_contiguous_view(tensors, *, expected_batch: int):
    view = _contiguous_slot_view(tensors, expected_batch=expected_batch)
    if view is not None:
        return view, ()

    import torch

    stacked = torch.stack(list(tensors), dim=0)
    return stacked, (stacked,)


def _contiguous_slot_view(tensors, *, expected_batch: int):
    if not tensors:
        return None
    first = tensors[0]
    if len(tensors) != expected_batch:
        return None
    if any(tensor.device != first.device for tensor in tensors):
        return None
    if any(tensor.dtype != first.dtype for tensor in tensors):
        return None
    if any(tuple(tensor.shape) != tuple(first.shape) for tensor in tensors):
        return None
    if any(not tensor.is_contiguous() for tensor in tensors):
        return None

    storage_offset = first.storage_offset()
    expert_numel = first.numel()
    expected_offsets = [
        storage_offset + index * expert_numel
        for index in range(expected_batch)
    ]
    actual_offsets = [tensor.storage_offset() for tensor in tensors]
    if actual_offsets != expected_offsets:
        return None

    return first.as_strided(
        (expected_batch, *first.shape),
        (expert_numel, *first.stride()),
    )


def _all_npu_slots(transfers: Sequence[ExpertTransfer]) -> bool:
    return all(
        transfer.slot.w13.device.type == "npu"
        and transfer.slot.w2.device.type == "npu"
        for transfer in transfers
    )
