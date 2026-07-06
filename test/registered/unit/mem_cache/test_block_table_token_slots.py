import importlib.util
from pathlib import Path

import torch


def _load_block_table_module():
    module_path = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "mem_cache"
        / "block_table.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_block_table_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extend_block_table_uses_out_cache_loc_for_current_chunk_pages():
    module = _load_block_table_module()
    page_size = 128
    req_to_token = torch.zeros((3, 512), dtype=torch.int32)
    req_to_token[1, 0] = 1000
    req_to_token[1, 128] = 1128
    req_to_token[1, 256] = 1256
    req_to_token[1, 384] = 1234567
    req_to_token[2, 0] = 2000
    req_to_token[2, 128] = 2128

    out_cache_loc = torch.arange(2000, 2200, dtype=torch.int32)

    token_slots = module.build_extend_block_table_token_slots(
        req_to_token=req_to_token,
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
        max_len=500,
        page_size=page_size,
        prefix_lens=torch.tensor([300, 128], dtype=torch.int64),
        extend_lens=torch.tensor([200, 0], dtype=torch.int64),
        out_cache_loc=out_cache_loc,
    )

    torch.testing.assert_close(
        token_slots[0],
        torch.tensor([1000, 1128, 1256, 2084], dtype=torch.int32),
    )
    torch.testing.assert_close(
        token_slots[1],
        torch.tensor([2000, 2128, 0, 0], dtype=torch.int32),
    )


def test_target_verify_block_table_uses_out_cache_loc_at_new_page_boundary():
    module = _load_block_table_module()
    page_size = 128
    req_to_token = torch.zeros((4, 512), dtype=torch.int32)
    req_to_token[1, 0] = 1000
    req_to_token[1, 128] = 1128
    req_to_token[1, 256] = 1234567
    req_to_token[2, 0] = 2000
    req_to_token[2, 128] = 2128
    req_to_token[2, 256] = 2256
    req_to_token[3, 0] = 3456789

    out_cache_loc = torch.arange(9000, 9012, dtype=torch.int32)

    token_slots = module.build_extend_block_table_token_slots(
        req_to_token=req_to_token,
        req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        max_len=304,
        page_size=page_size,
        prefix_lens=torch.tensor([256, 300, 0], dtype=torch.int64),
        extend_lens=torch.tensor([4, 4, 4], dtype=torch.int64),
        out_cache_loc=out_cache_loc,
    )

    torch.testing.assert_close(
        token_slots,
        torch.tensor(
            [
                [1000, 1128, 9000],
                [2000, 2128, 2256],
                [9008, 0, 0],
            ],
            dtype=torch.int32,
        ),
    )
