# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.utils import next_power_of_2
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class PlainDraftModelSpeculator(DraftModelSpeculator):
    """Speculative decoding using a separate smaller draft LM.

    Unlike Eagle, the draft model runs fully independently of the target model.
    Step 0 builds an expanded buffer (accepted + correction token + rejected
    slots) via a Triton kernel; steps 1..k-1 are single-token decode steps.
    """

    # Plain draft model needs one extra slot per request for correction.
    num_extra_query_per_req = 1

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # draft_max_seq_len is read by the parent's attention metadata builder.
        # Plain draft model doesn't do per-batch adjustment; cap at max.
        self.draft_max_seq_len = self.max_model_len

        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        # GPU-side arange for decode query_start_loc initialisation.
        self.arange_gpu = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        # Scalar step counter used as column index into draft_logits.
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)

        _expanded_max = self.max_num_tokens + self.max_num_reqs
        self.expanded_input_ids = torch.zeros(
            _expanded_max, dtype=torch.int32, device=device
        )
        self.expanded_positions = torch.zeros(
            _expanded_max, dtype=torch.int64, device=device
        )
        self.expanded_slot_mappings: torch.Tensor
        self.supports_mm_inputs = False

        # FULL CUDA graph manager for the uniform multi-step decode loop
        # (steps 1..num_speculative_steps - 1). Draft prefill (step 0) is
        # variable-length and stays eager; see capture().
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize the manager for the draft decode steps (one token per
        # request per step, which enables FULL graph replay). The manager is
        # created unconditionally so dispatch_cg_and_sync_dp always has a
        # target; capture() skips the actual capture when there is no
        # multi-step decode loop.
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )

    def capture(self) -> None:
        if self.num_speculative_steps <= 1 or self.decode_cudagraph_manager is None:
            # No multi-step decode loop to graph (or no FULL decode support).
            return
        logger.info("Capturing decode CUDA graphs for plain draft model...")
        # Reset indices to zeros to prevent stale values from prior dummy runs
        # from causing out-of-bounds indexing during capture.
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()
        self.decode_cudagraph_manager.capture(
            self._generate_draft,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing plain-draft decode CUDA graphs",
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self.expanded_slot_mappings = torch.empty(
            block_tables.num_kv_cache_groups,
            self.max_num_tokens + self.max_num_reqs,
            dtype=torch.int64,
            device=self.device,
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        spec = self.speculative_config
        assert spec is not None
        draft_vllm_config = replace(
            self.vllm_config,
            model_config=self.draft_model_config,
            quant_config=None,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
        )
        with set_model_tag("draft_model"):
            return get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )

    @torch.inference_mode()
    def _run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self.model(  # type: ignore[misc]
                input_ids=input_ids,
                positions=positions,
            )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Run one uniform draft decode step: forward + sample + advance.

        This is the single step captured into the FULL decode CUDA graph.
        The per-step state advancement (writing ``draft_tokens[:, step]`` and
        moving input_ids/positions/seq_lens forward) is performed by a kernel
        keyed on the GPU ``current_draft_step`` scalar, so one captured step
        correctly chains when replayed ``num_speculative_steps - 1`` times.
        Slot mappings and attention metadata are computed eagerly per step
        into persistent buffers, which the replayed graph reads.
        """
        self._prepare_eplb_forward(num_reqs)

        # Run the draft model forward pass.
        hidden_states = self._run_model(
            self.input_buffers.input_ids[:num_tokens_padded],
            self.input_buffers.positions[:num_tokens_padded],
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        # Sample the draft token for this step.
        sampled = self.sample_draft(
            hidden_states[:num_reqs],
            self.input_buffers.positions[:num_reqs],
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

        # Advance the inputs for the next step (in-graph for FULL replay).
        advance_decode_inputs(
            sampled,
            self.current_draft_step,
            self.input_buffers,
            self.draft_tokens,
            self.draft_tokens.stride(0),
            self.max_model_len,
            self.num_speculative_steps,
        )

    def _accepted_last_indices(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        qsl = input_batch.query_start_loc
        adjusted_lens = qsl[1 : num_reqs + 1] - qsl[:num_reqs] - num_rejected[:num_reqs]
        return qsl[:num_reqs] + adjusted_lens - 1

    def _prefill(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        if skip_attn:
            src_tokens = input_batch.num_tokens
            self.last_token_indices[:num_reqs] = self._accepted_last_indices(
                input_batch, num_rejected, num_reqs
            )
            hidden_states = self._run_model(
                input_batch.input_ids[:src_tokens],
                input_batch.positions[:src_tokens],
                None,
                None,
                num_tokens_across_dp,
            )
            positions_buf = input_batch.positions
        else:
            block_tables = self.block_tables
            kv_cache_config = self.kv_cache_config
            assert block_tables is not None
            assert kv_cache_config is not None

            total_expanded = prepare_prefill_inputs(
                input_buffers=self.input_buffers,
                input_batch=input_batch,
                last_sampled=last_sampled,
                num_rejected=num_rejected,
                expanded_input_ids=self.expanded_input_ids,
                expanded_positions=self.expanded_positions,
                last_token_indices=self.last_token_indices,
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
            )

            prefill_slot_mappings = block_tables.compute_slot_mappings(
                input_batch.idx_mapping,
                self.input_buffers.query_start_loc[: num_reqs + 1],
                self.expanded_positions[:total_expanded],
                total_expanded,
                out=self.expanded_slot_mappings,
            )
            qsl_np = input_batch.query_start_loc_np
            query_start_loc_cpu_expanded = (
                torch.from_numpy(qsl_np[: num_reqs + 1]).int()
                + torch.from_numpy(self.arange_np[: num_reqs + 1]).int()
            )
            prefill_attn_md = self._build_attn_metadata(
                batch_desc=BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=total_expanded,
                    num_reqs=num_reqs,
                ),
                num_reqs=num_reqs,
                query_start_loc_np=query_start_loc_cpu_expanded.numpy(),
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                step=0,
                slot_mappings=prefill_slot_mappings,
            )
            prefill_slot_maps_by_layer = build_slot_mappings_by_layer(
                prefill_slot_mappings, kv_cache_config
            )
            hidden_states = self._run_model(
                self.expanded_input_ids[:total_expanded],
                self.expanded_positions[:total_expanded],
                prefill_attn_md,
                prefill_slot_maps_by_layer,
                num_tokens_across_dp,
            )
            positions_buf = self.expanded_positions

        last_indices = self.last_token_indices[:num_reqs]
        self.current_draft_step.fill_(0)
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            hidden_states[last_indices],
            positions_buf[last_indices],
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

    def _multi_step_decode(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        """Generate draft tokens for steps 1..num_speculative_steps - 1.

        The eager setup advances the persistent input buffers into the step-1
        state (input_ids <- draft_tokens[:, 0], positions+1, seq_lens+1). Each
        loop iteration then computes slot mappings and attention metadata
        eagerly (into the same persistent buffers the graph reads) and runs the
        step, which is captured as a FULL CUDA graph and replayed, or run
        eagerly when ``batch_desc.cg_mode`` is not FULL. The per-step advance to
        the next step happens inside the captured step (``_generate_draft``).
        """
        if skip_attn:
            last_positions = input_batch.positions[
                self._accepted_last_indices(input_batch, num_rejected, num_reqs)
            ]
        else:
            last_positions = self.expanded_positions[self.last_token_indices[:num_reqs]]

        # Advance the persistent buffers into the step-1 state.
        self.input_buffers.input_ids[:num_reqs].copy_(
            self.draft_tokens[:num_reqs, 0].int()
        )
        torch.clamp(
            last_positions + 1,
            max=self.max_model_len - 1,
            out=self.input_buffers.positions[:num_reqs],
        )
        # Step s forward uses seq_lens = target_seq_lens - num_rejected + 1 + s,
        # so the step-1 state is target_seq_lens - num_rejected + 2.
        torch.clamp(
            input_batch.seq_lens[:num_reqs] - num_rejected[:num_reqs].int() + 2,
            max=self.max_model_len,
            out=self.input_buffers.seq_lens[:num_reqs],
        )
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            self.arange_gpu[: num_reqs + 1]
        )
        # Reset the padded region for FULL CUDA graph replay. A captured graph
        # reads the full [max_num_reqs] buffers, and the values written during
        # capture (make_dummy) or a prior larger batch would otherwise make
        # padded requests appear to hold tokens. Flatten query_start_loc to the
        # last real value and zero the padded seq_lens, matching the autoreg
        # speculator's prepare_decode_inputs.
        if num_reqs < self.max_num_reqs:
            self.input_buffers.query_start_loc[num_reqs + 1 :].fill_(num_reqs)
            self.input_buffers.seq_lens[num_reqs:].zero_()

        idx_mapping = input_batch.idx_mapping
        num_tokens = batch_desc.num_tokens

        for step in range(1, self.num_speculative_steps):
            decode_attn_md = None
            step_slot_maps_by_layer = None
            if not skip_attn:
                block_tables = self.block_tables
                kv_cache_config = self.kv_cache_config
                assert block_tables is not None
                assert kv_cache_config is not None
                q_start = self.input_buffers.query_start_loc[: num_reqs + 1]
                positions = self.input_buffers.positions[:num_reqs]
                step_slot_mappings = block_tables.compute_slot_mappings(
                    idx_mapping, q_start, positions, num_tokens
                )
                step_slot_maps_by_layer = build_slot_mappings_by_layer(
                    step_slot_mappings, kv_cache_config
                )
                decode_attn_md = self._build_uniform_attn_metadata(
                    batch_desc=batch_desc,
                    num_reqs=num_reqs,
                    num_query_per_req=1,
                    seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                    # Include the correction token inserted during prefill.
                    step=step + 1,
                )

            self.current_draft_step.fill_(step)

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    num_tokens,
                    decode_attn_md,
                    step_slot_maps_by_layer,
                    num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,  # unused
        aux_hidden_states: list[torch.Tensor] | None,  # unused
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        assert self.model is not None

        num_reqs = input_batch.num_reqs
        skip_attn = dummy_run and skip_attn_for_dummy_run
        num_tokens_across_dp = (
            dp_sync.num_tokens_across_dp if dp_sync is not None else None
        )

        # Copy per-request temperature/seeds/idx_mapping into pre-allocated
        # buffers so sample_draft can read them without extra slicing.
        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        self._prefill(
            input_batch,
            num_rejected,
            last_sampled,
            num_reqs,
            skip_attn,
            num_tokens_across_dp,
        )

        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        # Each request produces exactly 1 token per draft decode step, which
        # enables FULL graph replay of the uniform multi-step decode loop.
        decode_batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs, num_query_per_req=1)
            if dp_sync is not None
            else (None, num_reqs)
        )
        decode_batch_desc, decode_batch_sync = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync=decode_batch_sync,
        )
        decode_num_tokens_across_dp = (
            decode_batch_sync.num_tokens_across_dp
            if decode_batch_sync is not None
            else None
        )

        self._multi_step_decode(
            input_batch,
            num_rejected,
            num_reqs,
            skip_attn,
            decode_batch_desc,
            decode_num_tokens_across_dp,
        )

        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_prefill_inputs_kernel(
    target_input_ids_ptr,  # [src_tokens] int32
    target_positions_ptr,  # [src_tokens] int64
    last_sampled_ptr,  # [max_num_reqs] int32
    idx_mapping_ptr,  # [num_reqs] int32
    out_input_ids_ptr,  # [src_tokens + num_reqs] int32
    out_positions_ptr,  # [src_tokens + num_reqs] int64
    last_token_indices_ptr,  # [max_num_reqs] int64
    out_query_start_loc_ptr,  # [max_num_reqs + 1] int32
    out_seq_lens_ptr,  # [max_num_reqs] int32
    query_start_loc_ptr,  # [num_reqs + 1] int32
    seq_lens_ptr,  # [num_reqs] int32
    num_rejected_ptr,  # [num_reqs] int32
    max_num_reqs,
    max_model_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-request step-0 input preparation for the plain draft-model speculator.

    Output layout for request i (out_start = query_start_loc[i] + i):
        [out_start,              out_start + num_valid)      accepted tokens
        [out_start + num_valid]                              correction token
        (out_start + num_valid, out_start + total_out)      rejected slots

    where num_valid  = query_lens[i] - num_rejected[i]
          total_out  = query_lens[i] + 1
    """
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    q_start = tl.load(query_start_loc_ptr + req_idx)
    q_next = tl.load(query_start_loc_ptr + req_idx + 1)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)

    num_valid = q_next - q_start - num_rejected
    correction_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    start_pos = tl.load(target_positions_ptr + q_start)
    out_start = q_start + req_idx
    total_out = q_next - q_start + 1

    for i in range(0, total_out, BLOCK_SIZE):
        j = i + tl.arange(0, BLOCK_SIZE)
        in_bounds = j < total_out

        is_valid = j < num_valid
        is_correction = j == num_valid
        src_idx = q_start + j
        token_ids = tl.load(target_input_ids_ptr + src_idx, mask=is_valid, other=0)
        positions = tl.minimum(start_pos + j, max_model_len - 1)
        token_ids = tl.where(is_correction, correction_token, token_ids)

        out_idx = out_start + j
        tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
        tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)

    tl.store(last_token_indices_ptr + req_idx, out_start + num_valid)
    tl.store(out_query_start_loc_ptr + req_idx, out_start)
    # seqlen_k for the expanded prefill attention:
    #   pre_existing = seq_len - query_len (draft KV before this round)
    #   expanded_query_len = query_len + 1 (adds correction token)
    #   seqlen_k = pre_existing + expanded_query_len = seq_len + 1
    # num_rejected does NOT change seqlen_k — the expanded query always has
    # query_len+1 slots. Rejected positions write temporary KV entries that
    # are overwritten with valid tokens in a later round.
    new_seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(out_seq_lens_ptr + req_idx, new_seq_len)

    if req_idx == num_reqs - 1:
        total_expanded = out_start + total_out
        tl.store(out_query_start_loc_ptr + num_reqs, total_expanded)
        for i in range(num_reqs + 1, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block <= max_num_reqs
            tl.store(out_query_start_loc_ptr + block, total_expanded, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_prefill_inputs(
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    last_sampled: torch.Tensor,  # [max_num_reqs] int64
    num_rejected: torch.Tensor,  # [num_reqs]     int64
    expanded_input_ids: torch.Tensor,  # [max_tokens + max_reqs] int32
    expanded_positions: torch.Tensor,  # [max_tokens + max_reqs] int64
    last_token_indices: torch.Tensor,  # [max_num_reqs] int64
    max_num_reqs: int,
    max_model_len: int,
) -> int:
    """Call _prepare_prefill_inputs_kernel and return total_expanded tokens.

    Side-effects (kernel writes):
      - expanded_input_ids, expanded_positions
      - last_token_indices
      - input_buffers.query_start_loc  (expanded: original[i] + i)
      - input_buffers.seq_lens         (= target_seq_lens - num_rejected + 1)
    """
    num_reqs = input_batch.num_reqs
    src_tokens = input_batch.num_tokens
    qsl_np = input_batch.query_start_loc_np
    query_lens = qsl_np[1 : num_reqs + 1] - qsl_np[:num_reqs]
    max_total_out = int(query_lens.max()) + 2  # +1 correction, +1 alignment
    BLOCK_SIZE = min(512, next_power_of_2(max_total_out))

    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_batch.input_ids,
        input_batch.positions,
        last_sampled.int(),
        input_batch.idx_mapping,
        expanded_input_ids,
        expanded_positions,
        last_token_indices,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        num_rejected.int(),
        max_num_reqs,
        max_model_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return src_tokens + num_reqs


@triton.jit
def _advance_decode_inputs_kernel(
    sampled_tokens_ptr,  # [num_reqs] int32
    input_ids_ptr,  # [max_num_tokens] int32
    positions_ptr,  # [max_num_tokens] int64
    seq_lens_ptr,  # [max_num_reqs] int32
    draft_tokens_ptr,  # [max_num_reqs, num_speculative_steps] int64
    draft_tokens_stride,
    current_draft_step_ptr,  # scalar int64
    max_model_len,
    num_speculative_steps,
):
    """Advance the draft decode inputs by one step (runs inside the graph).

    Called after sampling the token for the current step (identified by the
    GPU ``current_draft_step`` scalar). Writes the sampled token into
    ``draft_tokens[req, step]`` and, unless this is the final step, moves the
    persistent input buffers forward so the next step's forward can run.
    """
    req_idx = tl.program_id(0)

    draft_token = tl.load(sampled_tokens_ptr + req_idx).to(tl.int64)
    step = tl.load(current_draft_step_ptr).to(tl.int64)
    tl.store(
        draft_tokens_ptr + req_idx * draft_tokens_stride + step,
        draft_token,
    )

    if step >= num_speculative_steps - 1:
        # Final step: nothing left to advance.
        return

    # Feed the sampled token into the next forward pass.
    tl.store(input_ids_ptr + req_idx, draft_token.to(tl.int32))

    # Advance position and seq_lens for the next step, clamping to the max
    # model length to prevent out-of-range access.
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def advance_decode_inputs(
    sampled_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    draft_tokens: torch.Tensor,
    draft_tokens_stride: int,
    max_model_len: int,
    num_speculative_steps: int,
) -> None:
    """Launch ``_advance_decode_inputs_kernel`` for every request."""
    num_reqs = sampled_tokens.shape[0]
    _advance_decode_inputs_kernel[(num_reqs,)](
        sampled_tokens,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
        draft_tokens,
        draft_tokens_stride,
        current_draft_step,
        max_model_len,
        num_speculative_steps,
    )
