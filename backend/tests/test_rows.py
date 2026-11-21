"""`/api/rows` writes its table in chunks; the JSON a client decodes is unchanged."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from backend import app as backend_app
from backend.app import RowsRequest

_N = 2 * backend_app._ROWS_CHUNK + 3  # a partial last chunk
_FRAME = pd.DataFrame(
    {
        "a": np.arange(_N, dtype=np.float64) / 3,
        "b": np.arange(_N, dtype=np.int64),
        "c": [np.nan if i % 7 == 0 else i * 1.5 for i in range(_N)],
        "d": [f"s{i}" for i in range(_N)],
    }
)


def _decoded(req: RowsRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(backend_app, "_load_dataset", lambda key: _FRAME)
    response = backend_app.rows(req)

    async def drain() -> str:
        return "".join([c async for c in response.body_iterator])

    return json.loads(asyncio.run(drain()))


def _expected(ids: list[int], cols: list[str]) -> dict[str, Any]:
    """What the endpoint sent before it streamed: the whole table, parsed back."""
    sub = _FRAME.iloc[ids][cols]
    return {"columns": cols, "rows": json.loads(sub.to_json(orient="records"))}


@pytest.mark.parametrize(
    ("ids", "columns"),
    [
        (list(range(_N)), None),
        ([5, 0, 5, _N - 1, 1], ["c", "a"]),
        ([], ["a"]),
        ([3, 4], []),
    ],
)
def test_rows_decode_as_the_whole_table_did(
    ids: list[int], columns: list[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = _decoded(RowsRequest(dataset="x", ids=ids, columns=columns), monkeypatch)
    assert got == _expected(ids, list(_FRAME.columns) if columns is None else columns)


def test_rows_out_of_range_is_still_a_400(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(HTTPException) as err:
        _decoded(RowsRequest(dataset="x", ids=[_N]), monkeypatch)
    assert err.value.status_code == 400
