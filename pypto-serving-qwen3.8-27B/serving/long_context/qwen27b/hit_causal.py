"""Layer-major prefill with packed projections and original state/attention arithmetic."""
import pypto.language as pl
from serving.long_context.model import linear_ROWS, linear_HIDDEN, linear_OUTPUT, linear_INPUT, linear_OUTPUT_TILE, linear_K_TILE, rmsnorm_ROWS, rmsnorm_HIDDEN, rmsnorm_CHUNK, gated_norm_ROWS, gated_norm_DIM, gated_norm_ROW_TILE, causal_conv_TOKENS, causal_conv_CHANNELS, causal_conv_KERNEL, causal_conv_CHANNEL_TILE, delta_prepare_TOKENS, delta_prepare_KEY_HEADS, delta_prepare_VALUE_HEADS, delta_prepare_DIM, delta_prepare_CHANNELS, delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS, delta_window16_ROWS, delta_window16_KEY_HEADS, delta_window16_HEADS, delta_window16_DIM, delta_window16_VALUE_TILE, delta_window16_NORM_TILE, delta_window16_COLUMN_REP, delta_window16_VALUE_WIDTH, attention3_prepare_ROWS, attention3_prepare_HIDDEN, attention3_prepare_HEAD_DIM, attention3_prepare_Q_HEADS, attention3_prepare_KV_HEADS, attention3_prepare_Q_PROJECTION, attention3_prepare_KV_WIDTH, attention3_prepare_Q_WIDTH, attention3_prepare_ROTARY, attention512_core_ROWS, attention512_core_CONTEXT, attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM, attention512_core_Q_HEADS, attention512_core_KV_HEADS, attention512_core_Q_WIDTH, residual_ROWS, residual_HIDDEN, residual_TILE, swiglu_ROWS, swiglu_INTERMEDIATE, swiglu_CHANNEL_TILE, layer0_stem_ROWS, layer0_stem_HIDDEN, layer0_stem_QKV, layer0_stem_Z, layer0_stem_GATE_PAD, ROWS, BATCH, CONTEXT, BLOCK_SIZE, POOL_BLOCKS, HIDDEN, INTERMEDIATE, QKV, VALUE_WIDTH, DELTA_HEADS, DELTA_DIM, FULL_HEAD_DIM, FULL_Q_WIDTH, FULL_Q_PROJECTION, FULL_KV_WIDTH, ROTARY, VOCAB, tp_rank_ROWS, tp_rank_BATCH, tp_rank_CONTEXT, tp_rank_BLOCK_SIZE, tp_rank_POOL_BLOCKS, tp_rank_HIDDEN, tp_rank_INTERMEDIATE, tp_rank_QKV, tp_rank_VALUE_WIDTH, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_FULL_HEAD_DIM, tp_rank_FULL_Q_WIDTH, tp_rank_FULL_Q_PROJECTION, tp_rank_FULL_KV_WIDTH, tp_rank_ROTARY, tp_rank_VOCAB, tp_rank_TP_VOCAB, causal_conv_causal_conv_silu_inline, delta_window16_expand_qkv_window16_inline, delta_prepare_decay_beta_inline, delta_window16_recurrence_window16_inline, attention3_prepare_normalize_head_inline, attention3_prepare_rotary_head_inline
STATE_BATCH = 1
BATCH = 1
ROWS = 512
PACKED = BATCH * ROWS
delta_prepare_PAD_ROWS = 128
delta_window16_ROWS = ROWS
attention3_prepare_ROWS = 16
ATTENTION_TILE_ROWS = 16
PV_TILE_ROWS = 32
attention512_core_SCORE_KEY_BLOCK = 128
attention512_core_ROWS = ROWS
delta_window16_VALUE_TILE = 128
FUSED_ROWS = 16

@pl.jit.inline
def prefill_project_inline_prefix512(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(4 * output_tiles, name_hint='prefill_nz_projection_active4'):
        task = pl.tile.get_block_idx()
        row = task % 4 * 128
        col = task // 4 * 128
        accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
            weight_tile = pl.slice(weight, [256, 128], [k_start, col])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
        output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_project_ffn_up_paired_m128(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(2 * output_tiles, name_hint='prefill_ffn_up_paired_m128'):
        task = pl.tile.get_block_idx()
        row = task % 2 * 256
        col = task // 2 * 128
        accumulator_first = pl.create_tensor([128, 128], dtype=pl.FP32)
        accumulator_second = pl.create_tensor([128, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_first = pl.slice(hidden, [128, 256], [row, k_start])
            hidden_second = pl.slice(hidden, [128, 256], [row + 128, k_start])
            weight_tile = pl.slice(weight, [256, 128], [k_start, col])
            accumulator_first = pl.matmul_acc(accumulator_first, hidden_first, weight_tile, b_trans=False, init_cond=chunk == 0)
            accumulator_second = pl.matmul_acc(accumulator_second, hidden_second, weight_tile, b_trans=False, init_cond=chunk == 0)
        output[row:row + 128, col:col + 128] = pl.cast(accumulator_first, pl.BF16, mode='rint')
        output[row + 128:row + 256, col:col + 128] = pl.cast(accumulator_second, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_norm_inline_prefix512(hidden: pl.Tensor, gamma: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(hidden, 0)
    with pl.spmd(packed_rows // rmsnorm_ROWS, name_hint='qwen_prefill_rmsnorm'):
        row_start = pl.tile.get_block_idx() * rmsnorm_ROWS
        square_sum = pl.full([1, rmsnorm_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, rmsnorm_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / rmsnorm_HIDDEN), 1e-06), high_precision=True), [rmsnorm_ROWS, 1])
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            weight = pl.slice(gamma, [1, rmsnorm_CHUNK], [0, column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            output[row_start:row_start + rmsnorm_ROWS, column:column + rmsnorm_CHUNK] = pl.cast(normalized, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_last_logits_inline_prefix512(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    selected_hidden = pl.create_tensor([16, HIDDEN], dtype=pl.BF16)
    with pl.spmd(16, name_hint='prefill_gather_last_hidden'):
        batch = pl.tile.get_block_idx()
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if count > 0:
                source_row = batch * ROWS + count - 1
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.slice(hidden, [1, HIDDEN], [source_row, 0])
            else:
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
        else:
            selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
    output_tiles = pl.tensor.dim(weight, 0) // 128
    input_columns = pl.tensor.dim(weight, 1)
    with pl.spmd(output_tiles, name_hint='prefill_last_vocab_projection'):
        column = pl.tile.get_block_idx() * 128
        accumulator = pl.create_tensor([16, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(selected_hidden, [16, 256], [0, k_start])
            weight_tile = pl.slice(weight, [128, 256], [column, k_start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=True, init_cond=chunk == 0)
        projected = pl.cast(accumulator, pl.BF16, mode='rint')
        for batch in pl.range(BATCH):
            if batch < active_count:
                count = pl.read(token_counts, [batch])
                if count > 0:
                    destination_row = batch * ROWS + count - 1
                    output[destination_row:destination_row + 1, column:column + 128] = pl.slice(projected, [1, 128], [batch, 0])
    return output

@pl.jit.inline
def prefill_batch_prefix512_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32], rk_bank: pl.Tensor, rk_mode: pl.Scalar[pl.INT32], rk_start: pl.Scalar[pl.INT32], rk_clean: pl.Scalar[pl.INT32], rk_end: pl.Scalar[pl.INT32], rk_windows: pl.Tensor[[48, 144], pl.INT32], rk_core: pl.Tensor, rk_state: pl.Tensor, rk_scores: pl.Tensor, rk_cutoffs: pl.Tensor[[64, 2048], pl.FP32]):
    mask = pl.create_tensor([ROWS, 384], dtype=pl.FP32)
    with pl.spmd(ROWS // 16 * 2, name_hint='p32_boundary_mask'):
        task = pl.tile.get_block_idx()
        start_row = task // 2 * 16
        panel_slot = task % 2
        pos = pl.read(cache_offsets, [0])
        count = pl.read(token_counts, [0])
        base_page = (pos + pl.min(start_row, count - 1)) // 128
        first_key = (base_page + panel_slot) * 128
        page_keys = pl.cast(pl.tensor.ci(pl.cast(first_key, pl.INT32), [1, 128], pl.INT32), pl.FP32)
        query_ids = pl.tensor.ci(pl.cast(pos + start_row, pl.INT32), [1, 16], pl.INT32)
        last_ids = pl.minimum(query_ids, pl.cast(pos + count - 1, pl.INT32))
        last_wide = pl.col_expand_add(pl.full([8, 16], dtype=pl.FP32, value=0.0), pl.cast(last_ids, pl.FP32))
        last = pl.row_max(pl.transpose(last_wide, 0, 1))
        logical_keys = pl.col_expand_add(pl.full([16, 128], dtype=pl.FP32, value=0.0), page_keys)
        future = pl.minimum(pl.maximum(pl.row_expand_sub(logical_keys, last), 0.0), 1.0)
        bias = pl.sub(pl.full([16, 128], dtype=pl.FP32, value=0.0), pl.mul(future, 10000.0))
        mask[start_row:start_row + 16, (panel_slot + 1) * 128:(panel_slot + 1) * 128 + 128] = bias
        if panel_slot == 0:
            mask[start_row:start_row + 16, 0:128] = pl.full([16, 128], dtype=pl.FP32, value=0.0)
    running = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
    with pl.spmd(512, name_hint='qwen_prefill_embedding'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if token_row < count:
                token = pl.read(input_ids, [batch, token_row])
                embedded = pl.slice(embedding, [1, HIDDEN], [token, 0])
            else:
                embedded = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
            running[row:row + 1, 0:HIDDEN] = embedded
            traces[batch:batch + 1, 0:1, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(embedded, [1, 1, 1, HIDDEN])
        else:
            running[row:row + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
    norm = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
    first_gamma: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, HIDDEN], [0, 0])
    norm = prefill_norm_inline_prefix512(running, first_gamma, norm)
    for layer in pl.range(64):
        post_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        residual, post_norm, offset, count, keep, mlp_output, sparse_mode, do_pack, candidates, prefix = redknot_ffn_initialize(cache_offsets, token_counts, layer, rk_mode, rk_end, rk_start)
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[HIDDEN, QKV], pl.BF16, pl.NZ] = delta_qkv_weight[linear]
            z_weight_layer: pl.Tensor[[HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[linear]
            a_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[linear]
            b_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[linear]
            conv_weight_layer: pl.Tensor[[4, QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[linear]
            qkv = pl.create_tensor([512, QKV], dtype=pl.BF16)
            z = pl.create_tensor([512, VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([512, 128], dtype=pl.BF16)
            b = pl.create_tensor([512, 128], dtype=pl.BF16)
            qkv = prefill_project_ffn_up_paired_m128(norm, qkv_weight_layer, qkv, active_count)
            z = prefill_project_ffn_up_paired_m128(norm, z_weight_layer, z, active_count)
            a = prefill_project_inline_prefix512(norm, a_weight_layer, a, active_count)
            b = prefill_project_inline_prefix512(norm, b_weight_layer, b, active_count)
            gated = pl.create_tensor([512, VALUE_WIDTH], dtype=pl.BF16)
            prefill_delta_slots_inline_prefix512(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated, linear, token_counts, active_count, rk_windows, rk_core, rk_state, rk_start, rk_end, rk_mode, pl.read(cache_offsets, [0]))
            attention = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_ffn_up_paired_m128(gated, delta_out_layer, attention, active_count)
            residual, post_norm, keep, prefix = redknot_ffn_prepare(layer, attention, keep, rk_scores, rk_cutoffs, offset, count, rk_start, rk_end, rk_mode, do_pack, prefix, running, post_gamma_layer, residual, post_norm)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q = pl.create_tensor([512, FULL_Q_PROJECTION], dtype=pl.BF16)
            k = pl.create_tensor([512, FULL_KV_WIDTH], dtype=pl.BF16)
            v = pl.create_tensor([512, FULL_KV_WIDTH], dtype=pl.BF16)
            q = prefill_project_ffn_up_paired_m128(norm, q_weight_full, q, active_count)
            k = prefill_project_inline_prefix512(norm, k_weight_full, k, active_count)
            v = prefill_project_inline_prefix512(norm, v_weight_full, v, active_count)
            gated = pl.create_tensor([512, FULL_Q_WIDTH], dtype=pl.BF16)
            prefill_attention_slots_inline_prefix512(q, k, v, q_gamma_full, k_gamma_full, cos, sin, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, token_counts, gated, full, active_count, rk_bank, rk_mode, rk_start, rk_clean, rk_end)
            attention = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_ffn_up_paired_m128(gated, out_weight_full, attention, active_count)
            residual, post_norm, keep, prefix = redknot_ffn_prepare(layer, attention, keep, rk_scores, rk_cutoffs, offset, count, rk_start, rk_end, rk_mode, do_pack, prefix, running, post_gamma_layer, residual, post_norm)
        running, norm, traces = redknot_ffn_execute(do_pack, post_norm, mlp_gate_layer, prefix, mlp_up_layer, mlp_down_layer, mlp_output, active_count, sparse_mode, keep, residual, input_gamma, final_gamma, running, norm, traces, layer)
    normalized = norm
    logits_rows = pl.reshape(logits, [PACKED, VOCAB])
    prefill_last_logits_inline_prefix512(normalized, lm_head, logits_rows, token_counts, active_count)
    with pl.spmd(512, name_hint='qwen_prefill_store_final_norm'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            traces[batch:batch + 1, 65:66, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(pl.slice(normalized, [1, HIDDEN], [row, 0]), [1, 1, 1, HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.inline
def prefill_batch_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32], rk_bank: pl.Tensor, rk_mode: pl.Scalar[pl.INT32], rk_start: pl.Scalar[pl.INT32], rk_clean: pl.Scalar[pl.INT32], rk_end: pl.Scalar[pl.INT32], rk_windows: pl.Tensor[[48, 144], pl.INT32], rk_core: pl.Tensor, rk_state: pl.Tensor, rk_scores: pl.Tensor, rk_cutoffs: pl.Tensor[[64, 2048], pl.FP32]):
    prefix_group = (active_count + 1) // 2
    if prefix_group == 1:
        prefill_batch_prefix512_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, rk_bank, rk_mode, rk_start, rk_clean, rk_end, rk_windows, rk_core, rk_state, rk_scores, rk_cutoffs)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def rank_prefill_shared(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[STATE_BATCH, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[STATE_BATCH, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.InOut[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32], rk_bank: pl.InOut[pl.Tensor[[32768, 4096], pl.BF16]], rk_mode: pl.Scalar[pl.INT32], rk_start: pl.Scalar[pl.INT32], rk_clean: pl.Scalar[pl.INT32], rk_end: pl.Scalar[pl.INT32], rk_windows: pl.Tensor[[48, 144], pl.INT32], rk_core: pl.InOut[pl.Tensor[[1550336, 128], pl.BF16]], rk_state: pl.InOut[pl.Tensor[[149376, 128], pl.FP32]], rk_scores: pl.InOut[pl.Tensor[[64, 2048], pl.FP32]], rk_cutoffs: pl.Tensor[[64, 2048], pl.FP32]):
    return prefill_batch_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, rk_bank, rk_mode, rk_start, rk_clean, rk_end, rk_windows, rk_core, rk_state, rk_scores, rk_cutoffs)

@pl.jit.host
def prefill_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64, HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ], mlp_up_weight: pl.Tensor[[64, HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ], mlp_down_weight: pl.Tensor[[64, INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ], delta_qkv_weight: pl.Tensor[[48, HIDDEN, QKV], pl.BF16, pl.NZ], delta_z_weight: pl.Tensor[[48, HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ], delta_a_weight: pl.Tensor[[48, HIDDEN, 128], pl.BF16, pl.NZ], delta_b_weight: pl.Tensor[[48, HIDDEN, 128], pl.BF16, pl.NZ], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48, VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ], full_q_weight: pl.Tensor[[16, HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ], full_k_weight: pl.Tensor[[16, HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ], full_v_weight: pl.Tensor[[16, HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16, FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[STATE_BATCH, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[STATE_BATCH, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.InOut[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32], rk_bank: pl.InOut[pl.Tensor[[32768, 4096], pl.BF16]], rk_mode: pl.Scalar[pl.INT32], rk_start: pl.Scalar[pl.INT32], rk_clean: pl.Scalar[pl.INT32], rk_end: pl.Scalar[pl.INT32], rk_windows: pl.Tensor[[48, 144], pl.INT32], rk_core: pl.InOut[pl.Tensor[[1550336, 128], pl.BF16]], rk_state: pl.InOut[pl.Tensor[[149376, 128], pl.FP32]], rk_scores: pl.InOut[pl.Tensor[[64, 2048], pl.FP32]], rk_cutoffs: pl.Tensor[[64, 2048], pl.FP32]):
    rank_prefill_shared(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, rk_bank, rk_mode, rk_start, rk_clean, rk_end, rk_windows, rk_core, rk_state, rk_scores, rk_cutoffs, device=0)
