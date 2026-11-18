@pl.jit.inline
def boundary_mask(cache_offsets: pl.Tensor, token_counts: pl.Tensor):
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
    return mask
