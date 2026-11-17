"""Layer-major prefill with packed projections and original state/attention arithmetic."""
import pypto.language as pl
from serving.long_context.model import linear_ROWS, linear_HIDDEN, linear_OUTPUT, linear_INPUT, linear_OUTPUT_TILE, linear_K_TILE, rmsnorm_ROWS, rmsnorm_HIDDEN, rmsnorm_CHUNK, gated_norm_ROWS, gated_norm_DIM, gated_norm_ROW_TILE, causal_conv_TOKENS, causal_conv_CHANNELS, causal_conv_KERNEL, causal_conv_CHANNEL_TILE, delta_prepare_TOKENS, delta_prepare_KEY_HEADS, delta_prepare_VALUE_HEADS, delta_prepare_DIM, delta_prepare_CHANNELS, delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS, delta_window16_ROWS, delta_window16_KEY_HEADS, delta_window16_HEADS, delta_window16_DIM, delta_window16_VALUE_TILE, delta_window16_NORM_TILE, delta_window16_COLUMN_REP, delta_window16_VALUE_WIDTH, attention3_prepare_ROWS, attention3_prepare_HIDDEN, attention3_prepare_HEAD_DIM, attention3_prepare_Q_HEADS, attention3_prepare_KV_HEADS, attention3_prepare_Q_PROJECTION, attention3_prepare_KV_WIDTH, attention3_prepare_Q_WIDTH, attention3_prepare_ROTARY, attention512_core_ROWS, attention512_core_CONTEXT, attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM, attention512_core_Q_HEADS, attention512_core_KV_HEADS, attention512_core_Q_WIDTH, residual_ROWS, residual_HIDDEN, residual_TILE, swiglu_ROWS, swiglu_INTERMEDIATE, swiglu_CHANNEL_TILE, layer0_stem_ROWS, layer0_stem_HIDDEN, layer0_stem_QKV, layer0_stem_Z, layer0_stem_GATE_PAD, ROWS, BATCH, CONTEXT, BLOCK_SIZE, POOL_BLOCKS, HIDDEN, INTERMEDIATE, QKV, VALUE_WIDTH, DELTA_HEADS, DELTA_DIM, FULL_HEAD_DIM, FULL_Q_WIDTH, FULL_Q_PROJECTION, FULL_KV_WIDTH, ROTARY, VOCAB, tp_rank_ROWS, tp_rank_BATCH, tp_rank_CONTEXT, tp_rank_BLOCK_SIZE, tp_rank_POOL_BLOCKS, tp_rank_HIDDEN, tp_rank_INTERMEDIATE, tp_rank_QKV, tp_rank_VALUE_WIDTH, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_FULL_HEAD_DIM, tp_rank_FULL_Q_WIDTH, tp_rank_FULL_Q_PROJECTION, tp_rank_FULL_KV_WIDTH, tp_rank_ROTARY, tp_rank_VOCAB, tp_rank_TP_VOCAB, causal_conv_causal_conv_silu_inline, delta_window16_expand_qkv_window16_inline, delta_prepare_decay_beta_inline, delta_window16_recurrence_window16_inline, attention3_prepare_normalize_head_inline, attention3_prepare_rotary_head_inline
PACKED = BATCH * ROWS
delta_window16_VALUE_TILE = 64

@pl.jit.inline
def prefill_project_inline(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(output_tiles, name_hint='prefill_nz_projection'):
        col = pl.tile.get_block_idx() * 128
        accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(hidden, [128, 256], [0, k_start])
            weight_tile = pl.slice(weight, [256, 128], [k_start, col])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
        output[0:128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_norm_inline(hidden: pl.Tensor, gamma: pl.Tensor, output: pl.Tensor):
    with pl.spmd(BATCH, name_hint='qwen_prefill_rmsnorm'):
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
def prefill_residual_inline(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    with pl.spmd(PACKED, name_hint='qwen_prefill_residual'):
        row = pl.tile.get_block_idx()
        a = pl.cast(pl.slice(left, [1, HIDDEN], [row, 0]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, HIDDEN], [row, 0]), pl.FP32)
        output[row:row + 1, 0:HIDDEN] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_swiglu_inline(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    with pl.spmd(BATCH * (INTERMEDIATE // 1024), name_hint='qwen_prefill_swiglu'):
        task = pl.tile.get_block_idx()
        row = task // (INTERMEDIATE // 1024) * ROWS
        column = task % (INTERMEDIATE // 1024) * 1024
        gate_tile = pl.cast(pl.slice(gate, [ROWS, 1024], [row, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        up_tile = pl.cast(pl.slice(up, [ROWS, 1024], [row, column]), pl.FP32)
        output[row:row + ROWS, column:column + 1024] = pl.cast(pl.mul(pl.cast(activated, pl.FP32), up_tile), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_batch_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32], rk_bank: pl.InOut[pl.Tensor[[32768, 4096], pl.BF16]], rk_mode: pl.Scalar[pl.INT32], rk_start: pl.Scalar[pl.INT32], rk_clean: pl.Scalar[pl.INT32], rk_end: pl.Scalar[pl.INT32]):
    running = pl.create_tensor([PACKED, HIDDEN], dtype=pl.BF16)
    with pl.spmd(PACKED, name_hint='qwen_prefill_embedding'):
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
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        norm = pl.create_tensor([PACKED, HIDDEN], dtype=pl.BF16)
        norm = prefill_norm_inline(running, input_gamma_layer, norm)
        residual = pl.create_tensor([PACKED, HIDDEN], dtype=pl.BF16)
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
            qkv = pl.create_tensor([PACKED, QKV], dtype=pl.BF16)
            z = pl.create_tensor([PACKED, VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([PACKED, 128], dtype=pl.BF16)
            b = pl.create_tensor([PACKED, 128], dtype=pl.BF16)
            qkv = prefill_project_inline(norm, qkv_weight_layer, qkv)
            z = prefill_project_inline(norm, z_weight_layer, z)
            a = prefill_project_inline(norm, a_weight_layer, a)
            b = prefill_project_inline(norm, b_weight_layer, b)
            gated = pl.create_tensor([PACKED, VALUE_WIDTH], dtype=pl.BF16)
            prefill_delta_slots_inline(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated, linear, token_counts, active_count)
            attention = pl.create_tensor([PACKED, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline(gated, delta_out_layer, attention)
            residual = prefill_residual_inline(running, attention, residual)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q = pl.create_tensor([PACKED, FULL_Q_PROJECTION], dtype=pl.BF16)
            k = pl.create_tensor([PACKED, FULL_KV_WIDTH], dtype=pl.BF16)
            v = pl.create_tensor([PACKED, FULL_KV_WIDTH], dtype=pl.BF16)
            q = prefill_project_inline(norm, q_weight_full, q)
            k = prefill_project_inline(norm, k_weight_full, k)
            v = prefill_project_inline(norm, v_weight_full, v)
            gated = pl.create_tensor([PACKED, FULL_Q_WIDTH], dtype=pl.BF16)
            prefill_attention_slots_inline(q, k, v, q_gamma_full, k_gamma_full, cos, sin, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, token_counts, gated, full, active_count, rk_bank, rk_mode, rk_start, rk_clean, rk_end)
            attention = pl.create_tensor([PACKED, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline(gated, out_weight_full, attention)
            residual = prefill_residual_inline(running, attention, residual)
        post_norm = pl.create_tensor([PACKED, HIDDEN], dtype=pl.BF16)
        post_norm = prefill_norm_inline(residual, post_gamma_layer, post_norm)
        mlp_gate = pl.create_tensor([PACKED, INTERMEDIATE], dtype=pl.BF16)
        mlp_up = pl.create_tensor([PACKED, INTERMEDIATE], dtype=pl.BF16)
        mlp_gate = prefill_project_inline(post_norm, mlp_gate_layer, mlp_gate)
        mlp_up = prefill_project_inline(post_norm, mlp_up_layer, mlp_up)
        middle = pl.create_tensor([PACKED, INTERMEDIATE], dtype=pl.BF16)
        middle = prefill_swiglu_inline(mlp_gate, mlp_up, middle)
        mlp_output = pl.create_tensor([PACKED, HIDDEN], dtype=pl.BF16)
        mlp_output = prefill_project_inline(middle, mlp_down_layer, mlp_output)
        with pl.spmd(PACKED // 2, name_hint='qwen_prefill_residual_store'):
            row = pl.tile.get_block_idx() * 2
            batch = row // ROWS
            token_row = row % ROWS
            io_a = pl.cast(pl.slice(residual, [2, HIDDEN], [row, 0]), pl.FP32)
            io_b = pl.cast(pl.slice(mlp_output, [2, HIDDEN], [row, 0]), pl.FP32)
            next_hidden = pl.cast(pl.add(io_a, io_b), pl.BF16, mode='rint')
            running[row:row + 2, 0:HIDDEN] = next_hidden
            if batch < active_count:
                traces[batch:batch + 1, layer + 1:layer + 2, token_row:token_row + 2, 0:HIDDEN] = pl.reshape(next_hidden, [1, 1, 2, HIDDEN])
    normalized = pl.create_tensor([PACKED, HIDDEN], dtype=pl.BF16)
    normalized = prefill_norm_inline(running, final_gamma, normalized)
    logits_rows = pl.reshape(logits, [PACKED, VOCAB])
    prefill_project_inline_nd(normalized, lm_head, logits_rows)
    with pl.spmd(PACKED, name_hint='qwen_prefill_store_final_norm'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            traces[batch:batch + 1, 65:66, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(pl.slice(normalized, [1, HIDDEN], [row, 0]), [1, 1, 1, HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def rank_prefill_shared(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[1, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[1, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32], rk_bank: pl.InOut[pl.Tensor[[32768, 4096], pl.BF16]], rk_mode: pl.Scalar[pl.INT32], rk_start: pl.Scalar[pl.INT32], rk_clean: pl.Scalar[pl.INT32], rk_end: pl.Scalar[pl.INT32]):
    return prefill_batch_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, rk_bank, rk_mode, rk_start, rk_clean, rk_end)

@pl.jit.host
def prefill_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64, HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ], mlp_up_weight: pl.Tensor[[64, HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ], mlp_down_weight: pl.Tensor[[64, INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ], delta_qkv_weight: pl.Tensor[[48, HIDDEN, QKV], pl.BF16, pl.NZ], delta_z_weight: pl.Tensor[[48, HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ], delta_a_weight: pl.Tensor[[48, HIDDEN, 128], pl.BF16, pl.NZ], delta_b_weight: pl.Tensor[[48, HIDDEN, 128], pl.BF16, pl.NZ], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48, VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ], full_q_weight: pl.Tensor[[16, HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ], full_k_weight: pl.Tensor[[16, HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ], full_v_weight: pl.Tensor[[16, HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16, FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[1, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[1, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32], rk_bank: pl.InOut[pl.Tensor[[32768, 4096], pl.BF16]], rk_mode: pl.Scalar[pl.INT32], rk_start: pl.Scalar[pl.INT32], rk_clean: pl.Scalar[pl.INT32], rk_end: pl.Scalar[pl.INT32]):
    rank_prefill_shared(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, rk_bank, rk_mode, rk_start, rk_clean, rk_end, device=0)

@pl.jit.inline
def prefill_project_inline_nd(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor):
    output_tiles = pl.tensor.dim(weight, 0) // 128
    input_columns = pl.tensor.dim(weight, 1)
    with pl.spmd(128 // 128 * output_tiles, name_hint='prefill_packed_projection'):
        task = pl.tile.get_block_idx()
        row = task // output_tiles * 128
        col = task % output_tiles * 128
        accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
            weight_tile = pl.slice(weight, [128, 256], [col, k_start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=True, init_cond=chunk == 0)
        output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output
