"""Device-resident slot maintenance for the frozen Qwen3.8-27B state ABI."""
import pypto.language as pl
BATCH = 1
CONV_ELEMENTS = 48 * 4 * 10240
RECURRENT_ELEMENTS = 48 * 48 * 128 * 128
TILE = 32768

@pl.jit
def zero_state(conv: pl.InOut[pl.Tensor], recurrent: pl.InOut[pl.Tensor], slot: pl.Scalar[pl.INDEX]):
    conv_rows = pl.reshape(conv, [BATCH, CONV_ELEMENTS])
    recurrent_rows = pl.reshape(recurrent, [BATCH, RECURRENT_ELEMENTS])
    with pl.spmd(CONV_ELEMENTS // TILE, name_hint='qwen_zero_conv_state'):
        column = pl.tile.get_block_idx() * TILE
        conv_rows[slot:slot + 1, column:column + TILE] = pl.full([1, TILE], dtype=pl.BF16, value=0.0)
    with pl.spmd(RECURRENT_ELEMENTS // TILE, name_hint='qwen_zero_recurrent_state'):
        column = pl.tile.get_block_idx() * TILE
        recurrent_rows[slot:slot + 1, column:column + TILE] = pl.full([1, TILE], dtype=pl.FP32, value=0.0)
    return (conv, recurrent)

@pl.jit.host
def zero_state_fwd(conv: pl.InOut[pl.Tensor], recurrent: pl.InOut[pl.Tensor], slot: pl.Scalar[pl.INDEX]):
    zero_state(conv, recurrent, slot, device=0)
