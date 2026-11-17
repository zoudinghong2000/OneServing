"""ROWS=8 decode specialization for the Qwen3.8-27B PyPTO kernels."""
import pypto.language as pl
import pypto.language.distributed as pld
linear_ROWS = 8
linear_HIDDEN = 5120
linear_OUTPUT = pl.dynamic('QWEN_LINEAR_OUTPUT')
linear_INPUT = pl.dynamic('QWEN_LINEAR_INPUT')
linear_OUTPUT_TILE = 128
linear_K_TILE = 512
rmsnorm_ROWS = 8
rmsnorm_HIDDEN = 5120
rmsnorm_CHUNK = 128
gated_norm_ROWS = 240
gated_norm_DIM = 128
gated_norm_ROW_TILE = 16
causal_conv_TOKENS = pl.dynamic('QWEN_CONV_TOKENS')
causal_conv_CHANNELS = 10240
causal_conv_KERNEL = 4
causal_conv_CHANNEL_TILE = 128
delta_prepare_TOKENS = 5
delta_prepare_KEY_HEADS = 16
delta_prepare_VALUE_HEADS = 48
delta_prepare_DIM = 128
delta_prepare_CHANNELS = (delta_prepare_KEY_HEADS * 2 + delta_prepare_VALUE_HEADS) * delta_prepare_DIM
delta_prepare_PAD_ROWS = 8
delta_prepare_PAD_HEADS = 64
delta_window16_ROWS = 8
delta_window16_KEY_HEADS = 16
delta_window16_HEADS = 48
delta_window16_DIM = 128
delta_window16_VALUE_TILE = 32
delta_window16_NORM_TILE = 16
delta_window16_COLUMN_REP = 16
delta_window16_VALUE_WIDTH = delta_window16_HEADS * delta_window16_DIM
attention3_prepare_ROWS = 8
attention3_prepare_HIDDEN = 5120
attention3_prepare_HEAD_DIM = 256
attention3_prepare_Q_HEADS = 24
attention3_prepare_KV_HEADS = 4
attention3_prepare_Q_PROJECTION = attention3_prepare_Q_HEADS * attention3_prepare_HEAD_DIM * 2
attention3_prepare_KV_WIDTH = attention3_prepare_KV_HEADS * attention3_prepare_HEAD_DIM
attention3_prepare_Q_WIDTH = attention3_prepare_Q_HEADS * attention3_prepare_HEAD_DIM
attention3_prepare_ROTARY = 64
attention512_core_ROWS = 8
attention512_core_CONTEXT = 32768
attention512_core_KEY_BLOCK = 16
attention512_core_HEAD_DIM = 256
attention512_core_Q_HEADS = 24
attention512_core_KV_HEADS = 4
attention512_core_Q_WIDTH = attention512_core_Q_HEADS * attention512_core_HEAD_DIM
residual_ROWS = 8
residual_HIDDEN = 5120
residual_TILE = 5120
swiglu_ROWS = 8
swiglu_INTERMEDIATE = 17408
swiglu_CHANNEL_TILE = 128
layer0_stem_ROWS = 8
layer0_stem_HIDDEN = 5120
layer0_stem_QKV = 10240
layer0_stem_Z = 6144
layer0_stem_GATE_PAD = 128
ROWS = 8
BATCH = 8
CONTEXT = 32768
BLOCK_SIZE = 16
POOL_BLOCKS = 2049
HIDDEN = 5120
INTERMEDIATE = 17408
QKV = 10240
VALUE_WIDTH = 6144
DELTA_HEADS = 48
DELTA_DIM = 128
FULL_HEAD_DIM = 256
FULL_Q_WIDTH = 6144
FULL_Q_PROJECTION = 12288
FULL_KV_WIDTH = 1024
ROTARY = 64
VOCAB = 248320
tp_rank_ROWS = 8
tp_rank_BATCH = 8
tp_rank_CONTEXT = 32768
tp_rank_BLOCK_SIZE = 16
tp_rank_POOL_BLOCKS = 2049
tp_rank_HIDDEN = 5120
tp_rank_INTERMEDIATE = 17408
tp_rank_QKV = 10240
tp_rank_VALUE_WIDTH = 6144
tp_rank_DELTA_HEADS = 48
tp_rank_DELTA_DIM = 128
tp_rank_FULL_HEAD_DIM = 256
tp_rank_FULL_Q_WIDTH = 6144
tp_rank_FULL_Q_PROJECTION = 12288
tp_rank_FULL_KV_WIDTH = 1024
tp_rank_ROTARY = 64
tp_rank_VOCAB = 248320
tp_rank_TP_VOCAB = tp_rank_VOCAB // 2

@pl.program
class CandidateExchange:

    @pl.function(type=pl.FunctionType.InCore)
    def exchange(self, local: pl.Tensor[[1, 8], pl.FP32], result: pl.Out[pl.Tensor[[1, 16], pl.FP32]], data: pl.InOut[pld.DistributedTensor[[2, 8], pl.FP32]], signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]]) -> pl.Tensor[[1, 16], pl.FP32]:
        data = pld.tensor.allgather(local, data, signal)
        for rank in pl.range(2):
            chunk = pl.load(data, [rank, 0], [1, 8])
            result = pl.store(chunk, [0, rank * 8], result)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def rank_orch(self, local: pl.Tensor[[1, 8], pl.FP32], result: pl.Out[pl.Tensor[[1, 16], pl.FP32]], data: pl.InOut[pld.DistributedTensor[[2, 8], pl.FP32]], signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]]) -> pl.Tensor[[1, 16], pl.FP32]:
        return self.exchange(local, result, data, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self, candidates: pl.Tensor[[2, 1, 8], pl.FP32], gathered: pl.Out[pl.Tensor[[2, 1, 16], pl.FP32]]) -> pl.Tensor[[2, 1, 16], pl.FP32]:
        data_buffer = pld.alloc_window_buffer(2 * 8 * pl.FP32.get_byte())
        signal_buffer = pld.alloc_window_buffer(2 * pl.INT32.get_byte())
        for rank in pl.range(pld.world_size()):
            data = pld.window(data_buffer, [2, 8], dtype=pl.FP32)
            signal = pld.window(signal_buffer, [2, 1], dtype=pl.INT32)
            self.rank_orch(candidates[rank], gathered[rank], data, signal, device=rank)
        return gathered
from serving.long_context.sampling import BATCH as combined_sampler_BATCH, VOCAB as combined_sampler_VOCAB, CHUNKS as combined_sampler_CHUNKS, FULL_CHUNKS as combined_sampler_FULL_CHUNKS, CHUNK as combined_sampler_CHUNK, PARTIAL_COLUMNS as combined_sampler_PARTIAL_COLUMNS, SENTINEL as combined_sampler_SENTINEL, NEGATIVE_FINITE as combined_sampler_NEGATIVE_FINITE, greedy_partial as combined_sampler_greedy_partial, greedy_partial_tail as combined_sampler_greedy_partial_tail

@pl.jit.inline
def linear_hidden_projection_inline(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
    output_columns = pl.tensor.dim(result, 1)
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(output_columns // 256, name_hint='qwen_nz_hidden_projection'):
        start = pl.tile.get_block_idx() * 256
        accumulator = pl.create_tensor([linear_ROWS, 256], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(hidden, [linear_ROWS, 256], [0, k_start])
            weight_tile = pl.slice(weight, [256, 256], [k_start, start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
        result[0:linear_ROWS, start:start + 256] = pl.cast(accumulator, pl.BF16, mode='rint')
    return result

@pl.jit.inline
def rmsnorm_rmsnorm_inline(hidden: pl.Tensor[[rmsnorm_ROWS, rmsnorm_HIDDEN], pl.BF16], gamma: pl.Tensor[[1, rmsnorm_HIDDEN], pl.FP32], output: pl.Tensor[[rmsnorm_ROWS, rmsnorm_HIDDEN], pl.BF16]):
    with pl.spmd(1, name_hint='qwen_rmsnorm'):
        row_start = pl.tile.get_block_idx() * 8
        square_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        for load_group in pl.pipeline(0, rmsnorm_HIDDEN // 512, stage=2):
            column = load_group * 512
            source_wide = pl.cast(pl.slice(hidden, [8, 512], [row_start, column]), pl.FP32)
            square_wide = pl.mul(source_wide, source_wide)
            for reduction_part in pl.unroll(4):
                square_part = pl.slice(square_wide, [8, 128], [0, reduction_part * 128])
                square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(square_part), [1, 8]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / rmsnorm_HIDDEN), 1e-06), high_precision=True), [8, 1])
        for output_group in pl.pipeline(0, rmsnorm_HIDDEN // 512, stage=2):
            output_column = output_group * 512
            output_source = pl.cast(pl.slice(hidden, [8, 512], [row_start, output_column]), pl.FP32)
            output_weight = pl.slice(gamma, [1, 512], [0, output_column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(output_source, inverse), output_weight)
            output[row_start:row_start + 8, output_column:output_column + 512] = pl.cast(normalized, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def delta_prepare_decay_beta_inline(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor):
    with pl.spmd(1, name_hint='qwen_delta_decay_beta'):
        row = pl.tile.get_block_idx() * delta_prepare_PAD_ROWS
        a_tile = pl.cast(pl.slice(a, [delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], [row, 0]), pl.FP32)
        b_tile = pl.cast(pl.slice(b, [delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], [row, 0]), pl.FP32)
        a_parameter = pl.cast(pl.slice(a_log, [1, delta_prepare_PAD_HEADS], [0, 0]), pl.FP32)
        dt_parameter = pl.cast(pl.slice(dt_bias, [1, delta_prepare_PAD_HEADS], [0, 0]), pl.FP32)
        softplus_input = pl.col_expand_add(a_tile, dt_parameter)
        softplus = pl.log(pl.add(pl.exp(softplus_input), 1.0))
        decay = pl.mul(pl.col_expand_mul(softplus, pl.exp(a_parameter)), -1.0)
        beta_value = pl.div(pl.full([delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(b_tile)), 1.0))
        log_decay[row:row + delta_prepare_PAD_ROWS, 0:delta_prepare_PAD_HEADS] = decay
        beta[row:row + delta_prepare_PAD_ROWS, 0:delta_prepare_PAD_HEADS] = pl.cast(beta_value, pl.BF16, mode='rint')
    return (log_decay, beta)

@pl.jit.inline
def attention3_prepare_normalize_head_inline(head_values: pl.Tensor, gamma: pl.Tensor, result: pl.Tensor):
    source = pl.cast(head_values, pl.FP32)
    square = pl.mul(source, source)
    inverse = pl.rsqrt(pl.add(pl.mul(pl.row_sum(square), 1.0 / attention3_prepare_HEAD_DIM), 1e-06), high_precision=True)
    normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), gamma)
    result[0:attention3_prepare_ROWS, 0:attention3_prepare_HEAD_DIM] = pl.cast(normalized, pl.BF16, mode='rint')
    return result

@pl.jit.inline
def attention3_prepare_rotary_head_inline(normalized: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], pl.BF16], cos: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_ROTARY], pl.BF16], sin: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_ROTARY], pl.BF16], result: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], pl.BF16]):
    first = pl.cast(pl.slice(normalized, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    second = pl.cast(pl.slice(normalized, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, attention3_prepare_ROTARY // 2]), pl.FP32)
    cos_first = pl.cast(pl.slice(cos, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    cos_second = pl.cast(pl.slice(cos, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, attention3_prepare_ROTARY // 2]), pl.FP32)
    sin_first = pl.cast(pl.slice(sin, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    sin_second = pl.cast(pl.slice(sin, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, attention3_prepare_ROTARY // 2]), pl.FP32)
    rotated_first = pl.cast(pl.sub(pl.mul(first, cos_first), pl.mul(second, sin_first)), pl.BF16, mode='rint')
    rotated_second = pl.cast(pl.add(pl.mul(second, cos_second), pl.mul(first, sin_second)), pl.BF16, mode='rint')
    rest = pl.slice(normalized, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM - attention3_prepare_ROTARY], [0, attention3_prepare_ROTARY])
    result[0:attention3_prepare_ROWS, 0:attention3_prepare_HEAD_DIM] = pl.concat(pl.concat(rotated_first, rotated_second), rest)
    return result

@pl.jit.inline
def attention3_prepare_attention3_prepare_inline(hidden: pl.Tensor, q_weight: pl.Tensor, k_weight: pl.Tensor, v_weight: pl.Tensor, q_gamma: pl.Tensor, k_gamma: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, q_norm: pl.Tensor, k_norm: pl.Tensor, q_rot: pl.Tensor, k_rot: pl.Tensor, value: pl.Tensor, gate: pl.Tensor):
    q_projected = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_Q_PROJECTION], dtype=pl.BF16)
    k_projected = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_KV_WIDTH], dtype=pl.BF16)
    q_projected = linear_hidden_projection_inline(hidden, q_weight, q_projected)
    k_projected = linear_hidden_projection_inline(hidden, k_weight, k_projected)
    value = linear_hidden_projection_inline(hidden, v_weight, value)
    with pl.spmd(attention3_prepare_Q_HEADS, name_hint='qwen_attention_query_norm'):
        head = pl.tile.get_block_idx()
        projection_start = head * attention3_prepare_HEAD_DIM * 2
        result_start = head * attention3_prepare_HEAD_DIM
        raw = pl.slice(q_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [0, projection_start])
        normalized = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = attention3_prepare_normalize_head_inline(raw, q_gamma, normalized)
        q_norm[0:attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = normalized
        gate[0:attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = pl.slice(q_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [0, projection_start + attention3_prepare_HEAD_DIM])
        rotated = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        rotated = attention3_prepare_rotary_head_inline(normalized, cos, sin, rotated)
        q_rot[0:attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = rotated
    with pl.spmd(attention3_prepare_KV_HEADS, name_hint='qwen_attention_key_norm'):
        head = pl.tile.get_block_idx()
        start = head * attention3_prepare_HEAD_DIM
        raw = pl.slice(k_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [0, start])
        normalized = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = attention3_prepare_normalize_head_inline(raw, k_gamma, normalized)
        k_norm[0:attention3_prepare_ROWS, start:start + attention3_prepare_HEAD_DIM] = normalized
        rotated = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        rotated = attention3_prepare_rotary_head_inline(normalized, cos, sin, rotated)
        k_rot[0:attention3_prepare_ROWS, start:start + attention3_prepare_HEAD_DIM] = rotated
    return (q_norm, k_norm, q_rot, k_rot, value, gate)

@pl.jit.inline
def decode_gated_tail_inline(core: pl.Tensor, z_projection: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor, gated_output: pl.Tensor, attention_output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    gate_rows = pl.reshape(z_projection, [BATCH * DELTA_HEADS, DELTA_DIM])
    core_rows = pl.reshape(core, [delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM])
    with pl.spmd(delta_window16_ROWS * delta_window16_HEADS // delta_window16_NORM_TILE, name_hint='qwen_window_gated_rmsnorm'):
        row = pl.tile.get_block_idx() * delta_window16_NORM_TILE
        if row // DELTA_HEADS < active_count:
            source = pl.cast(pl.slice(core_rows, [delta_window16_NORM_TILE, delta_window16_DIM], [row, 0]), pl.FP32)
            gate_tile = pl.cast(pl.slice(gate_rows, [delta_window16_NORM_TILE, delta_window16_DIM], [row, 0]), pl.FP32)
            weight_tile = pl.cast(pl.slice(norm_weight, [1, gated_norm_DIM], [0, 0]), pl.FP32)
            variance = pl.mul(pl.row_sum(pl.mul(source, source)), 1.0 / delta_window16_DIM)
            inverse = pl.rsqrt(pl.add(variance, 1e-06), high_precision=True)
            normalized = pl.cast(pl.row_expand_mul(source, inverse), pl.BF16, mode='rint')
            weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight_tile), pl.BF16, mode='rint')
            sigmoid = pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
            gated_output[row:row + delta_window16_NORM_TILE, 0:delta_window16_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), sigmoid), pl.BF16, mode='rint')
    return gated_output

@pl.jit.inline
def decode_delta_slots_inline(qkv: pl.Tensor, z: pl.Tensor, a: pl.Tensor, b: pl.Tensor, conv_weight: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, gated_batch: pl.Tensor, linear: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    state_layers = pl.tensor.dim(conv_states, 1)
    conv_rows = pl.reshape(conv_states, [1 * state_layers * 4, QKV])
    state_rows = pl.reshape(recurrent_states, [state_layers * (1 * DELTA_HEADS * DELTA_DIM), DELTA_DIM])
    conv_output = pl.create_tensor([BATCH, QKV], dtype=pl.BF16)
    with pl.spmd(BATCH * (QKV // 5120), name_hint='qwen_decode_batched_conv'):
        task = pl.tile.get_block_idx()
        batch = task // (QKV // 5120)
        col = task % (QKV // 5120) * 5120
        if batch < active_count:
            state_start = (batch * state_layers + linear) * 4
            previous_1 = pl.cast(pl.slice(conv_rows, [1, 5120], [state_start + 1, col]), pl.FP32)
            previous_2 = pl.cast(pl.slice(conv_rows, [1, 5120], [state_start + 2, col]), pl.FP32)
            previous_3 = pl.cast(pl.slice(conv_rows, [1, 5120], [state_start + 3, col]), pl.FP32)
            current = pl.cast(pl.slice(qkv, [1, 5120], [batch, col]), pl.FP32)
            tap_0 = pl.cast(pl.slice(conv_weight, [1, 5120], [0, col]), pl.FP32)
            tap_1 = pl.cast(pl.slice(conv_weight, [1, 5120], [1, col]), pl.FP32)
            tap_2 = pl.cast(pl.slice(conv_weight, [1, 5120], [2, col]), pl.FP32)
            tap_3 = pl.cast(pl.slice(conv_weight, [1, 5120], [3, col]), pl.FP32)
            convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
            convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
            activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
            conv_output[batch:batch + 1, col:col + 5120] = pl.cast(activated, pl.BF16, mode='rint')
            conv_rows[state_start:state_start + 1, col:col + 5120] = pl.cast(previous_1, pl.BF16, mode='rint')
            conv_rows[state_start + 1:state_start + 2, col:col + 5120] = pl.cast(previous_2, pl.BF16, mode='rint')
            conv_rows[state_start + 2:state_start + 3, col:col + 5120] = pl.cast(previous_3, pl.BF16, mode='rint')
            conv_rows[state_start + 3:state_start + 4, col:col + 5120] = pl.cast(current, pl.BF16, mode='rint')
        else:
            conv_output[batch:batch + 1, col:col + 5120] = pl.full([1, 5120], dtype=pl.BF16, value=0.0)
    gate_values = pl.create_tensor([BATCH, 64], dtype=pl.FP32)
    beta_bf16 = pl.create_tensor([BATCH, 64], dtype=pl.BF16)
    gate_values, beta_bf16 = delta_prepare_decay_beta_inline(a, b, a_log, dt_bias, gate_values, beta_bf16)
    beta_values = pl.create_tensor([BATCH, 64], dtype=pl.FP32)
    with pl.spmd(1, name_hint='qwen_decode_batched_beta'):
        beta_row = pl.tile.get_block_idx() * BATCH
        beta_values[beta_row:beta_row + BATCH, 0:64] = pl.cast(pl.slice(beta_bf16, [BATCH, 64], [beta_row, 0]), pl.FP32)
    q_norm = pl.create_tensor([BATCH * DELTA_HEADS, DELTA_DIM], dtype=pl.FP32)
    k_norm = pl.create_tensor([BATCH * DELTA_HEADS, DELTA_DIM], dtype=pl.FP32)
    with pl.spmd(BATCH * 2, name_hint='qwen_decode_direct_qk_norm') as norm_task:
        direct_norm_task = pl.tile.get_block_idx()
        direct_norm_batch = direct_norm_task // 2
        direct_norm_half = direct_norm_task % 2
        direct_norm_q = pl.cast(pl.reshape(pl.slice(conv_output, [1, 1024], [direct_norm_batch, direct_norm_half * 1024]), [8, DELTA_DIM]), pl.FP32)
        direct_norm_k = pl.cast(pl.reshape(pl.slice(conv_output, [1, 1024], [direct_norm_batch, 2048 + direct_norm_half * 1024]), [8, DELTA_DIM]), pl.FP32)
        direct_norm_q_unit = pl.mul(pl.row_expand_mul(direct_norm_q, pl.rsqrt(pl.add(pl.row_sum(pl.mul(direct_norm_q, direct_norm_q)), 1e-06), high_precision=True)), DELTA_DIM ** (-0.5))
        direct_norm_k_unit = pl.row_expand_mul(direct_norm_k, pl.rsqrt(pl.add(pl.row_sum(pl.mul(direct_norm_k, direct_norm_k)), 1e-06), high_precision=True))
        direct_norm_heads = pl.cast(pl.mul(pl.cast(pl.arange(0, [1, 24], dtype=pl.INT32), pl.FP32), 1.0 / 3.0), pl.INT32, mode='floor')
        direct_norm_indices = pl.row_expand(pl.create_tensor([24, DELTA_DIM], dtype=pl.INT32), pl.reshape(direct_norm_heads, [24, 1]))
        direct_norm_flat_indices = pl.col_expand_add(pl.mul(direct_norm_indices, 128), pl.arange(0, [1, 128], dtype=pl.INT32))
        direct_norm_start = direct_norm_batch * DELTA_HEADS + direct_norm_half * 24
        q_norm[direct_norm_start:direct_norm_start + 24, 0:DELTA_DIM] = pl.gather(direct_norm_q_unit, index=direct_norm_flat_indices)
        k_norm[direct_norm_start:direct_norm_start + 24, 0:DELTA_DIM] = pl.gather(direct_norm_k_unit, index=direct_norm_flat_indices)
    core = pl.create_tensor([BATCH, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
    core_rows = pl.reshape(core, [BATCH, VALUE_WIDTH])
    with pl.spmd(40, name_hint='qwen_decode_delta_state_update', deps=[norm_task]) as update_task:
        reuse_state_block = pl.tile.get_block_idx()
        for reuse_state_head_part in pl.range(10):
            reuse_state_head_task = reuse_state_block + reuse_state_head_part * 40
            reuse_state_batch = reuse_state_head_task // DELTA_HEADS
            reuse_state_head = reuse_state_head_task % DELTA_HEADS
            if reuse_state_batch < active_count:
                reuse_state_row = reuse_state_batch * DELTA_HEADS + reuse_state_head
                reuse_state_k = pl.slice(k_norm, [1, DELTA_DIM], [reuse_state_row, 0])
                reuse_state_g = pl.read(gate_values, [reuse_state_batch, reuse_state_head])
                reuse_state_decay = pl.exp(pl.mul(pl.full([1, DELTA_DIM], dtype=pl.FP32, value=1.0), reuse_state_g))
                reuse_state_beta_value = pl.read(beta_values, [reuse_state_batch, reuse_state_head])
                for reuse_state_half in pl.range(2):
                    reuse_state_value_start = reuse_state_half * 64
                    reuse_state_output_start = reuse_state_head * DELTA_DIM + reuse_state_value_start
                    reuse_state_state_start = (reuse_state_batch * state_layers + linear) * DELTA_HEADS * DELTA_DIM + reuse_state_output_start
                    reuse_state_state = pl.slice(state_rows, [64, DELTA_DIM], [reuse_state_state_start, 0])
                    reuse_state_decayed = pl.col_expand_mul(reuse_state_state, reuse_state_decay)
                    reuse_state_remembered = pl.row_sum(pl.col_expand_mul(reuse_state_decayed, reuse_state_k))
                    reuse_state_v_row = pl.cast(pl.slice(conv_output, [1, 64], [reuse_state_batch, 4096 + reuse_state_head * DELTA_DIM + reuse_state_value_start]), pl.FP32)
                    reuse_state_v_col = pl.reshape(reuse_state_v_row, [64, 1])
                    reuse_state_delta = pl.mul(pl.sub(reuse_state_v_col, reuse_state_remembered), reuse_state_beta_value)
                    reuse_state_following = pl.add(reuse_state_decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([64, DELTA_DIM], dtype=pl.FP32, value=1.0), reuse_state_delta), reuse_state_k))
                    state_rows[reuse_state_state_start:reuse_state_state_start + 64, 0:DELTA_DIM] = reuse_state_following
    with pl.spmd(20, name_hint='qwen_decode_delta_output_group4', deps=[update_task]):
        reuse_output_coarse_block = pl.tile.get_block_idx()
        for reuse_output_coarse_part in pl.range(5):
            reuse_output_coarse_task = reuse_output_coarse_block + reuse_output_coarse_part * 20
            if reuse_output_coarse_task < BATCH * (DELTA_HEADS // 4):
                reuse_output_coarse_batch = reuse_output_coarse_task // (DELTA_HEADS // 4)
                reuse_output_coarse_head = reuse_output_coarse_task % (DELTA_HEADS // 4) * 4
                reuse_output_coarse_output_start = reuse_output_coarse_head * DELTA_DIM
                if reuse_output_coarse_batch < active_count:
                    reuse_output_coarse_state_start = (reuse_output_coarse_batch * state_layers + linear) * DELTA_HEADS * DELTA_DIM + reuse_output_coarse_output_start
                    reuse_output_coarse_output_following = pl.slice(state_rows, [4 * DELTA_DIM, DELTA_DIM], [reuse_output_coarse_state_start, 0])
                    reuse_output_coarse_q = pl.slice(q_norm, [4, DELTA_DIM], [reuse_output_coarse_batch * DELTA_HEADS + reuse_output_coarse_head, 0])
                    reuse_output_coarse_product = pl.cast(pl.matmul(reuse_output_coarse_q, reuse_output_coarse_output_following, pl.FP32, b_trans=True), pl.BF16, mode='rint')
                    for reuse_output_coarse_local in pl.unroll(4):
                        reuse_output_coarse_col = reuse_output_coarse_output_start + reuse_output_coarse_local * DELTA_DIM
                        core_rows[reuse_output_coarse_batch:reuse_output_coarse_batch + 1, reuse_output_coarse_col:reuse_output_coarse_col + DELTA_DIM] = pl.slice(reuse_output_coarse_product, [1, DELTA_DIM], [reuse_output_coarse_local, reuse_output_coarse_local * DELTA_DIM])
                else:
                    core_rows[reuse_output_coarse_batch:reuse_output_coarse_batch + 1, reuse_output_coarse_output_start:reuse_output_coarse_output_start + 4 * DELTA_DIM] = pl.full([1, 4 * DELTA_DIM], dtype=pl.BF16, value=0.0)
    gated_rows = pl.reshape(gated_batch, [BATCH * DELTA_HEADS, DELTA_DIM])
    ignored_projection = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    decode_gated_tail_inline(core, z, norm_weight, out_weight, gated_rows, ignored_projection, active_count)
    return (conv_states, recurrent_states, gated_batch)

@pl.jit.inline
def decode_attention_slots_inline(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor, mask: pl.Tensor, out_weight: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, offsets: pl.Tensor, gated_batch: pl.Tensor, full: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    with pl.spmd(BATCH * 4, name_hint='qwen_decode_batched_update_kv') as update_task:
        task = pl.tile.get_block_idx()
        batch = task // 4
        head = task % 4
        if batch < active_count:
            position = pl.read(offsets, [batch])
            col = head * FULL_HEAD_DIM
            physical = pl.read(block_tables, [batch, (2 * (head // 2) + (head + full) % 2) * 256 + position // 16 // 8])
            slot = position % BLOCK_SIZE
            new_key = pl.slice(key, [1, FULL_HEAD_DIM], [batch, col])
            new_value = pl.slice(value, [1, FULL_HEAD_DIM], [batch, col])
            key_pool[full:full + 1, 0:1, physical:physical + 1, position // 16 % 8 * 16 + slot:position // 16 % 8 * 16 + slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(new_key, [1, 1, 1, 1, FULL_HEAD_DIM])
            value_pool[full:full + 1, 0:1, physical:physical + 1, position // 16 % 8 * 16 + slot:position // 16 % 8 * 16 + slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(new_value, [1, 1, 1, 1, FULL_HEAD_DIM])
    query_rows = pl.create_tensor([BATCH * 4 * 8, FULL_HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(BATCH * 4, name_hint='qwen_decode_group6_query') as grouped_query_task:
        task = pl.tile.get_block_idx()
        batch = task // 4
        kv_head = task % 4
        if batch < active_count:
            query_rows[task * 8:task * 8 + 8, 0:FULL_HEAD_DIM] = pl.full([8, FULL_HEAD_DIM], dtype=pl.BF16, value=0.0)
            six_queries = pl.reshape(pl.slice(query, [1, 6 * FULL_HEAD_DIM], [batch, kv_head * 6 * FULL_HEAD_DIM]), [6, FULL_HEAD_DIM])
            query_rows[task * 8:task * 8 + 6, 0:FULL_HEAD_DIM] = six_queries
    scores = pl.create_tensor([BATCH * 4 * 8, CONTEXT], dtype=pl.FP32)
    probabilities = pl.create_tensor([BATCH * 4 * 16, CONTEXT], dtype=pl.BF16)
    with pl.spmd(BATCH * 4, name_hint='qwen_decode_group6_init_scores') as init_scores_task:
        task = pl.tile.get_block_idx()
        batch = task // 4
        if batch < active_count:
            for part in pl.range((pl.read(offsets, [batch]) + 256) // 256):
                col = part * 256
                initial = pl.full([8, 256], dtype=pl.FP32, value=0.0)
                initial_mask = pl.reshape(pl.slice(mask, [1, 1, 256], [batch, 0, col]), [1, 256])
                scores[task * 8:task * 8 + 8, col:col + 256] = pl.col_expand_add(initial, initial_mask)
    if active_count == 1:
        with pl.spmd(16, name_hint='qwen_decode_group6_scores_C1_split4', deps=[update_task, init_scores_task, grouped_query_task]):
            lane = pl.tile.get_block_idx()
            task = lane // 4
            batch = 0
            kv_head = task
            partition = lane % 4
            q = pl.slice(query_rows, [8, FULL_HEAD_DIM], [task * 8, 0])
            position = pl.read(offsets, [batch])
            for block in pl.range(partition, position // BLOCK_SIZE + 1, 4):
                key_start = block * BLOCK_SIZE
                physical = pl.read(block_tables, [batch, (2 * (kv_head // 2) + (kv_head + full) % 2) * 256 + block // 8])
                k = pl.reshape(pl.slice(key_pool, [1, 1, 1, BLOCK_SIZE, FULL_HEAD_DIM], [full, 0, physical, block % 8 * 16, 0]), [BLOCK_SIZE, FULL_HEAD_DIM])
                raw = pl.matmul(q, k, pl.FP32, b_trans=True)
                rounded = pl.cast(raw, pl.BF16, mode='rint')
                scaled = pl.cast(pl.mul(pl.cast(rounded, pl.FP32), FULL_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
                score_mask = pl.reshape(pl.slice(mask, [1, 1, BLOCK_SIZE], [batch, 0, key_start]), [1, BLOCK_SIZE])
                score_rows = pl.slice(scaled, [8, BLOCK_SIZE], [0, 0])
                scores[task * 8:task * 8 + 8, key_start:key_start + BLOCK_SIZE] = pl.col_expand_add(pl.cast(score_rows, pl.FP32), score_mask)
    elif active_count == 8:
        with pl.spmd(BATCH * 4 * 4, name_hint='qwen_decode_group6_scores_C8_split4', deps=[update_task, init_scores_task, grouped_query_task]):
            lane = pl.tile.get_block_idx()
            task = lane // 4
            partition = lane % 4
            batch = task // 4
            kv_head = task % 4
            if batch < active_count:
                q = pl.slice(query_rows, [8, FULL_HEAD_DIM], [task * 8, 0])
                position = pl.read(offsets, [batch])
                for block in pl.range(partition, position // BLOCK_SIZE + 1, 4):
                    key_start = block * BLOCK_SIZE
                    physical = pl.read(block_tables, [batch, (2 * (kv_head // 2) + (kv_head + full) % 2) * 256 + block // 8])
                    k = pl.reshape(pl.slice(key_pool, [1, 1, 1, BLOCK_SIZE, FULL_HEAD_DIM], [full, 0, physical, block % 8 * 16, 0]), [BLOCK_SIZE, FULL_HEAD_DIM])
                    raw = pl.matmul(q, k, pl.FP32, b_trans=True)
                    rounded = pl.cast(raw, pl.BF16, mode='rint')
                    scaled = pl.cast(pl.mul(pl.cast(rounded, pl.FP32), FULL_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
                    score_mask = pl.reshape(pl.slice(mask, [1, 1, BLOCK_SIZE], [batch, 0, key_start]), [1, BLOCK_SIZE])
                    score_rows = pl.slice(scaled, [8, BLOCK_SIZE], [0, 0])
                    scores[task * 8:task * 8 + 8, key_start:key_start + BLOCK_SIZE] = pl.col_expand_add(pl.cast(score_rows, pl.FP32), score_mask)
    else:
        with pl.spmd(BATCH * 4, name_hint='qwen_decode_group6_scores', deps=[update_task, init_scores_task, grouped_query_task]):
            task = pl.tile.get_block_idx()
            batch = task // 4
            kv_head = task % 4
            if batch < active_count:
                q = pl.slice(query_rows, [8, FULL_HEAD_DIM], [task * 8, 0])
                position = pl.read(offsets, [batch])
                for block in pl.range(position // BLOCK_SIZE + 1):
                    key_start = block * BLOCK_SIZE
                    physical = pl.read(block_tables, [batch, (2 * (kv_head // 2) + (kv_head + full) % 2) * 256 + block // 8])
                    k = pl.reshape(pl.slice(key_pool, [1, 1, 1, BLOCK_SIZE, FULL_HEAD_DIM], [full, 0, physical, block % 8 * 16, 0]), [BLOCK_SIZE, FULL_HEAD_DIM])
                    raw = pl.matmul(q, k, pl.FP32, b_trans=True)
                    rounded = pl.cast(raw, pl.BF16, mode='rint')
                    scaled = pl.cast(pl.mul(pl.cast(rounded, pl.FP32), FULL_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
                    score_mask = pl.reshape(pl.slice(mask, [1, 1, BLOCK_SIZE], [batch, 0, key_start]), [1, BLOCK_SIZE])
                    score_rows = pl.slice(scaled, [8, BLOCK_SIZE], [0, 0])
                    scores[task * 8:task * 8 + 8, key_start:key_start + BLOCK_SIZE] = pl.col_expand_add(pl.cast(score_rows, pl.FP32), score_mask)
    with pl.spmd(BATCH * 4, name_hint='qwen_decode_group6_softmax'):
        task = pl.tile.get_block_idx()
        batch = task // 4
        if batch < active_count:
            maximum = pl.row_max(pl.slice(scores, [8, 256], [task * 8, 0]))
            for part in pl.range(1, (pl.read(offsets, [batch]) + 256) // 256):
                tile = pl.slice(scores, [8, 256], [task * 8, part * 256])
                maximum = pl.maximum(maximum, pl.row_max(tile))
            denominator = pl.row_sum(pl.exp(pl.row_expand_sub(pl.slice(scores, [8, 256], [task * 8, 0]), maximum)))
            for part in pl.range(1, (pl.read(offsets, [batch]) + 256) // 256):
                tile = pl.slice(scores, [8, 256], [task * 8, part * 256])
                denominator = pl.add(denominator, pl.row_sum(pl.exp(pl.row_expand_sub(tile, maximum))))
            for part in pl.range((pl.read(offsets, [batch]) + 256) // 256):
                col = part * 256
                tile = pl.slice(scores, [8, 256], [task * 8, col])
                probabilities8 = pl.cast(pl.row_expand_div(pl.exp(pl.row_expand_sub(tile, maximum)), denominator), pl.BF16, mode='rint')
                probabilities[task * 16:task * 16 + 16, col:col + 256] = pl.full([16, 256], dtype=pl.BF16, value=0.0)
                probabilities[task * 16:task * 16 + 8, col:col + 256] = probabilities8
    with pl.spmd(BATCH * 4, name_hint='qwen_decode_group6_weighted'):
        task = pl.tile.get_block_idx()
        batch = task // 4
        kv_head = task % 4
        if batch < active_count:
            weighted_sum = pl.full([16, FULL_HEAD_DIM], dtype=pl.FP32, value=0.0)
            position = pl.read(offsets, [batch])
            for block in pl.range(position // BLOCK_SIZE + 1):
                key_start = block * BLOCK_SIZE
                probs = pl.slice(probabilities, [16, BLOCK_SIZE], [task * 16, key_start])
                physical = pl.read(block_tables, [batch, (2 * (kv_head // 2) + (kv_head + full) % 2) * 256 + block // 8])
                values = pl.reshape(pl.slice(value_pool, [1, 1, 1, BLOCK_SIZE, FULL_HEAD_DIM], [full, 0, physical, block % 8 * 16, 0]), [BLOCK_SIZE, FULL_HEAD_DIM])
                weighted_sum = pl.add(weighted_sum, pl.matmul(probs, values, pl.FP32))
            weighted_rounded = pl.cast(weighted_sum, pl.BF16, mode='rint')
            weighted = pl.slice(weighted_rounded, [6, FULL_HEAD_DIM], [0, 0])
            col = kv_head * 6 * FULL_HEAD_DIM
            gate_values = pl.cast(pl.reshape(pl.slice(gate, [1, 6 * FULL_HEAD_DIM], [batch, col]), [6, FULL_HEAD_DIM]), pl.FP32)
            sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(gate_values)), 1.0)), pl.BF16, mode='rint')
            result = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
            gated_batch[batch:batch + 1, col:col + 6 * FULL_HEAD_DIM] = pl.reshape(result, [1, 6 * FULL_HEAD_DIM])
    return (key_pool, value_pool, gated_batch)

@pl.jit.inline
def decode_io_swiglu_inline(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    with pl.spmd(swiglu_INTERMEDIATE // 1024, name_hint='qwen_decode_wide_swiglu'):
        column = pl.tile.get_block_idx() * 1024
        gate_tile = pl.cast(pl.slice(gate, [swiglu_ROWS, 1024], [0, column]), pl.FP32)
        up_tile = pl.cast(pl.slice(up, [swiglu_ROWS, 1024], [0, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        multiplied = pl.mul(pl.cast(activated, pl.FP32), up_tile)
        output[0:swiglu_ROWS, column:column + 1024] = pl.cast(multiplied, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def fused_wide_inline(left: pl.Tensor, right: pl.Tensor, gamma: pl.Tensor, residual: pl.Tensor, normalized: pl.Tensor):
    with pl.spmd(8 // 8, name_hint='decode_residual_norm_full8_pipeline'):
        row_start = pl.tile.get_block_idx() * 8
        square_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        for load_group in pl.pipeline(0, 5120 // 512, stage=2):
            column = load_group * 512
            a = pl.cast(pl.slice(left, [8, 512], [row_start, column]), pl.FP32)
            b = pl.cast(pl.slice(right, [8, 512], [row_start, column]), pl.FP32)
            rounded = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
            residual[row_start:row_start + 8, column:column + 512] = rounded
            source_wide = pl.cast(rounded, pl.FP32)
            square_wide = pl.mul(source_wide, source_wide)
            for reduction_part in pl.unroll(4):
                square_part = pl.slice(square_wide, [8, 128], [0, reduction_part * 128])
                square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(square_part), [1, 8]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / 5120), 1e-06), high_precision=True), [8, 1])
        for output_group in pl.pipeline(0, 5120 // 512, stage=2):
            output_column = output_group * 512
            output_source = pl.cast(pl.slice(residual, [8, 512], [row_start, output_column]), pl.FP32)
            output_weight = pl.slice(gamma, [1, 512], [0, output_column])
            value = pl.col_expand_mul(pl.row_expand_mul(output_source, inverse), output_weight)
            normalized[row_start:row_start + 8, output_column:output_column + 512] = pl.cast(value, pl.BF16, mode='rint')
    return (residual, normalized)

@pl.jit.inline
def batch_step_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    running = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    cos_batch = pl.create_tensor([BATCH, ROTARY], dtype=pl.BF16)
    sin_batch = pl.create_tensor([BATCH, ROTARY], dtype=pl.BF16)
    with pl.spmd(BATCH, name_hint='qwen_decode_pack_rope'):
        batch = pl.tile.get_block_idx()
        cos_batch[batch:batch + 1, 0:ROTARY] = pl.reshape(pl.slice(cos, [1, 1, ROTARY], [batch, 0, 0]), [1, ROTARY])
        sin_batch[batch:batch + 1, 0:ROTARY] = pl.reshape(pl.slice(sin, [1, 1, ROTARY], [batch, 0, 0]), [1, ROTARY])
    with pl.spmd(BATCH * (HIDDEN // 256), name_hint='qwen_decode_token_embedding'):
        task = pl.tile.get_block_idx()
        batch = task // (HIDDEN // 256)
        col = task % (HIDDEN // 256) * 256
        if batch < active_count:
            token = pl.read(input_ids, [batch, 0])
            embedded = pl.slice(embedding, [1, 256], [token, col])
            running[batch:batch + 1, col:col + 256] = embedded
            traces[batch:batch + 1, 0:1, 0:1, col:col + 256] = pl.reshape(embedded, [1, 1, 1, 256])
        else:
            running[batch:batch + 1, col:col + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        norm = rmsnorm_rmsnorm_inline(running, input_gamma_layer, norm)
        residual = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
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
            qkv = pl.create_tensor([BATCH, QKV], dtype=pl.BF16)
            z = pl.create_tensor([BATCH, VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([BATCH, 128], dtype=pl.BF16)
            b = pl.create_tensor([BATCH, 128], dtype=pl.BF16)
            qkv = linear_hidden_projection_inline(norm, qkv_weight_layer, qkv)
            z = linear_hidden_projection_inline(norm, z_weight_layer, z)
            a = linear_hidden_projection_gate_inline(norm, a_weight_layer, a)
            b = linear_hidden_projection_gate_inline(norm, b_weight_layer, b)
            gated_delta = pl.create_tensor([BATCH, VALUE_WIDTH], dtype=pl.BF16)
            decode_delta_slots_inline(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated_delta, linear, active_count)
            attention_delta = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            attention_delta = linear_hidden_projection_inline(gated_delta, delta_out_layer, attention_delta)
            residual, post_norm = fused_wide_inline(running, attention_delta, post_gamma_layer, residual, post_norm)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q_norm = pl.create_tensor([BATCH, FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm = pl.create_tensor([BATCH, FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([BATCH, FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([BATCH, FULL_KV_WIDTH], dtype=pl.BF16)
            value = pl.create_tensor([BATCH, FULL_KV_WIDTH], dtype=pl.BF16)
            gate = pl.create_tensor([BATCH, FULL_Q_WIDTH], dtype=pl.BF16)
            q_norm, k_norm, q_rot, k_rot, value, gate = attention3_prepare_attention3_prepare_inline(norm, q_weight_full, k_weight_full, v_weight_full, q_gamma_full, k_gamma_full, cos_batch, sin_batch, q_norm, k_norm, q_rot, k_rot, value, gate)
            gated_full = pl.create_tensor([BATCH, FULL_Q_WIDTH], dtype=pl.BF16)
            decode_attention_slots_inline(q_rot, k_rot, value, gate, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, gated_full, full, active_count)
            attention_full = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            attention_full = linear_hidden_projection_inline(gated_full, out_weight_full, attention_full)
            residual, post_norm = fused_wide_inline(running, attention_full, post_gamma_layer, residual, post_norm)
        mlp_gate = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)
        mlp_up = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)
        mlp_gate = linear_hidden_projection_inline(post_norm, mlp_gate_layer, mlp_gate)
        mlp_up = linear_hidden_projection_inline(post_norm, mlp_up_layer, mlp_up)
        mlp_middle = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)
        mlp_middle = decode_io_swiglu_inline(mlp_gate, mlp_up, mlp_middle)
        mlp_output = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        mlp_output = linear_hidden_projection_inline(mlp_middle, mlp_down_layer, mlp_output)
        with pl.spmd(BATCH, name_hint='qwen_decode_residual_store'):
            io_batch = pl.tile.get_block_idx()
            io_left = pl.cast(pl.slice(residual, [1, HIDDEN], [io_batch, 0]), pl.FP32)
            io_right = pl.cast(pl.slice(mlp_output, [1, HIDDEN], [io_batch, 0]), pl.FP32)
            io_following = pl.cast(pl.add(io_left, io_right), pl.BF16, mode='rint')
            running[io_batch:io_batch + 1, 0:HIDDEN] = io_following
            if io_batch < active_count:
                traces[io_batch:io_batch + 1, layer + 1:layer + 2, 0:1, 0:HIDDEN] = pl.reshape(io_following, [1, 1, 1, HIDDEN])
    normalized = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    normalized = rmsnorm_rmsnorm_inline(running, final_gamma, normalized)
    packed_logits = pl.create_tensor([BATCH, VOCAB], dtype=pl.BF16)
    linear_hidden_projection_inline_nd(normalized, lm_head, packed_logits)
    with pl.spmd(BATCH * 61, name_hint='qwen_decode_coarse_logits'):
        io_task = pl.tile.get_block_idx()
        io_batch = io_task // 61
        io_part = io_task % 61
        io_column = io_part * 4096
        if io_batch < active_count:
            if io_part < 60:
                logits[io_batch:io_batch + 1, 0:1, io_column:io_column + 4096] = pl.reshape(pl.slice(packed_logits, [1, 4096], [io_batch, io_column]), [1, 1, 4096])
            else:
                logits[io_batch:io_batch + 1, 0:1, 245760:248320] = pl.reshape(pl.slice(packed_logits, [1, 2560], [io_batch, 245760]), [1, 1, 2560])
    with pl.spmd(BATCH, name_hint='qwen_decode_store_final_norm'):
        batch = pl.tile.get_block_idx()
        if batch < active_count:
            traces[batch:batch + 1, 65:66, 0:1, 0:HIDDEN] = pl.reshape(pl.slice(normalized, [1, HIDDEN], [batch, 0]), [1, 1, 1, HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.inline
def linear_hidden_projection_inline_nd(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
    output_columns = pl.tensor.dim(weight, 0)
    input_columns = pl.tensor.dim(weight, 1)
    with pl.spmd(output_columns // linear_OUTPUT_TILE, name_hint='qwen_hidden_projection'):
        start = pl.tile.get_block_idx() * linear_OUTPUT_TILE
        accumulator = pl.create_tensor([linear_ROWS, linear_OUTPUT_TILE], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // linear_K_TILE, stage=2):
            k_start = chunk * linear_K_TILE
            hidden_tile = pl.slice(hidden, [linear_ROWS, linear_K_TILE], [0, k_start])
            weight_tile = pl.slice(weight, [linear_OUTPUT_TILE, linear_K_TILE], [start, k_start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=True, init_cond=chunk == 0)
        result[0:linear_ROWS, start:start + linear_OUTPUT_TILE] = pl.cast(accumulator, pl.BF16, mode='rint')
    return result

@pl.jit.inline
def linear_hidden_projection_gate_inline(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
    output_columns = pl.tensor.dim(result, 1)
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(output_columns // 128, name_hint='qwen_nz_gate_projection'):
        start = pl.tile.get_block_idx() * 128
        accumulator = pl.create_tensor([linear_ROWS, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(hidden, [linear_ROWS, 256], [0, k_start])
            weight_tile = pl.slice(weight, [256, 128], [k_start, start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
        result[0:linear_ROWS, start:start + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return result

@pl.jit.inline
def combined_greedy_inline(logits: pl.Tensor, counts: pl.Tensor, maxima: pl.InOut[pl.Tensor], indices: pl.InOut[pl.Tensor], nan_indices: pl.InOut[pl.Tensor], tokens: pl.InOut[pl.Tensor], active: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(logits, 1)
    flat = pl.reshape(logits, [combined_sampler_BATCH * rows, combined_sampler_VOCAB])
    with pl.spmd(combined_sampler_BATCH * combined_sampler_CHUNKS, name_hint='qwen_greedy_partials') as partials:
        block = pl.tile.get_block_idx()
        slot = block // combined_sampler_CHUNKS
        part = block % combined_sampler_CHUNKS
        if slot < active:
            row = pl.read(counts, [slot]) - 1
            if part < combined_sampler_FULL_CHUNKS:
                combined_sampler_greedy_partial(flat, maxima, indices, nan_indices, slot, part, part * combined_sampler_CHUNK, slot * rows + row)
            elif part == combined_sampler_FULL_CHUNKS:
                combined_sampler_greedy_partial_tail(flat, maxima, indices, nan_indices, slot, part, combined_sampler_FULL_CHUNKS * combined_sampler_CHUNK, slot * rows + row)
            else:
                pl.store(pl.tile.muls(pl.tile.full([1, 8], dtype=pl.FP32, value=combined_sampler_NEGATIVE_FINITE), 4.0), [slot, part * 8], maxima)
                pl.store(pl.tile.full([1, 8], dtype=pl.FP32, value=combined_sampler_SENTINEL), [slot, part * 8], indices)
                pl.store(pl.tile.full([1, 8], dtype=pl.FP32, value=combined_sampler_SENTINEL), [slot, part * 8], nan_indices)
    with pl.spmd(combined_sampler_BATCH, name_hint='qwen_greedy_finalize', deps=[partials]):
        final_slot = pl.tile.get_block_idx()
        if final_slot < active:
            part_max = pl.tile.reshape(pl.load(maxima, [final_slot, 0], [1, combined_sampler_PARTIAL_COLUMNS]), [8, 64])
            part_index = pl.tile.reshape(pl.load(indices, [final_slot, 0], [1, combined_sampler_PARTIAL_COLUMNS]), [8, 64])
            part_nan = pl.tile.reshape(pl.load(nan_indices, [final_slot, 0], [1, combined_sampler_PARTIAL_COLUMNS]), [8, 64])
            final_scratch = pl.tile.create([8, 64], dtype=pl.FP32)
            row_maxima = pl.tile.row_max(part_max, final_scratch)
            final_max = pl.tile.col_max(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_maxima))
            final_max_column = pl.tile.reshape(final_max, [8, 1])
            final_expanded = pl.tile.row_expand(pl.tile.create([8, 64], dtype=pl.FP32), final_max_column)
            row_indices = pl.tile.row_min(pl.tile.select(pl.tile.cmp(part_max, final_expanded, cmp_type=0), part_index, combined_sampler_SENTINEL), final_scratch)
            final_index = pl.tile.col_min(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_indices))
            row_nan = pl.tile.row_min(part_nan, final_scratch)
            final_nan = pl.tile.col_min(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_nan))
            selected = pl.tile.select(pl.tile.cmps(final_nan, combined_sampler_SENTINEL, cmp_type=2), final_nan, final_index)
            pl.store(pl.tile.cast(selected, pl.INT32, mode='rint'), [final_slot, 0], tokens)
    return (maxima, indices, nan_indices, tokens)

@pl.jit
def rank_decode_greedy_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[1, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[1, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], maxima: pl.InOut[pl.Tensor[[BATCH, 512], pl.FP32]], indices: pl.InOut[pl.Tensor[[BATCH, 512], pl.FP32]], nan_indices: pl.InOut[pl.Tensor[[BATCH, 512], pl.FP32]], selected_tokens: pl.InOut[pl.Tensor[[BATCH, 8], pl.INT32]], active_count: pl.Scalar[pl.INT32]):
    conv_states, recurrent_states, key_pool, value_pool, traces, logits = batch_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    combined_greedy_inline(logits, token_counts, maxima, indices, nan_indices, selected_tokens, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits, maxima, indices, nan_indices, selected_tokens)

@pl.jit.host
def decode_greedy_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64, HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ], mlp_up_weight: pl.Tensor[[64, HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ], mlp_down_weight: pl.Tensor[[64, INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ], delta_qkv_weight: pl.Tensor[[48, HIDDEN, QKV], pl.BF16, pl.NZ], delta_z_weight: pl.Tensor[[48, HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ], delta_a_weight: pl.Tensor[[48, HIDDEN, 128], pl.BF16, pl.NZ], delta_b_weight: pl.Tensor[[48, HIDDEN, 128], pl.BF16, pl.NZ], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48, VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ], full_q_weight: pl.Tensor[[16, HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ], full_k_weight: pl.Tensor[[16, HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ], full_v_weight: pl.Tensor[[16, HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16, FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[1, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[1, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 1, KV_PHYSICAL_PAGES, 128, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], maxima: pl.InOut[pl.Tensor[[BATCH, 512], pl.FP32]], indices: pl.InOut[pl.Tensor[[BATCH, 512], pl.FP32]], nan_indices: pl.InOut[pl.Tensor[[BATCH, 512], pl.FP32]], selected_tokens: pl.InOut[pl.Tensor[[BATCH, 8], pl.INT32]], active_count: pl.Scalar[pl.INT32]):
    rank_decode_greedy_fwd(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, maxima, indices, nan_indices, selected_tokens, active_count, device=0)
