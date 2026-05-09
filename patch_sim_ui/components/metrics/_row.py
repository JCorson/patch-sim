"""Shared dict-to-row renderer for metrics-panel detail tables.

Each per-analysis panel (AP, calcium, burst) presents a scrollable table of
per-event metric rows.  The rows share structure: each cell is a small
``rx.text`` wrapped in ``rx.table.cell``.  The columns differ in which dict
key they read and how the value is rendered (``int + 1`` for the 1-based
``#`` index, plain string for most metric columns, or a sweep-number cell
that is blank on single-sweep runs).

The :class:`RowColumn` descriptor and :func:`metrics_row` helper capture
this shape so each panel only needs to declare its column list.
"""

from dataclasses import dataclass
from typing import Literal

import reflex as rx


@dataclass(frozen=True)
class RowColumn:
    """Declarative descriptor for one cell of a metrics-table row.

    Attributes:
        key: Dict key to read from the row dict.
        kind: ``"int_1based"`` renders ``value + 1`` as an int (used for the
            1-based ``#`` column).  ``"str"`` (default) renders the value
            coerced to ``str``.  ``"gated_int"`` renders the value as an int
            when ``int(value) > 0``, otherwise renders ``empty_text`` (used
            for the ``Sweep`` column, which is blank on single-sweep runs).
        empty_text: Text shown for ``gated_int`` cells when the gate is
            False.  Ignored for other kinds.
    """

    key: str
    kind: Literal["int_1based", "str", "gated_int"] = "str"
    empty_text: str = ""


def metrics_row(item: dict, columns: tuple[RowColumn, ...]) -> rx.Component:
    """Render a single ``rx.table.row`` from ``item`` per the column spec.

    Args:
        item: Pre-formatted dict of values for one row of the metrics table.
        columns: Ordered tuple of :class:`RowColumn` descriptors.  One
            ``rx.table.cell`` is emitted per column.

    Returns:
        An ``rx.table.row`` containing one cell per column.
    """
    cells: list[rx.Component] = []
    for column in columns:
        if column.kind == "int_1based":
            cells.append(
                rx.table.cell(rx.text(item[column.key].to(int) + 1, size="1")),
            )
        elif column.kind == "gated_int":
            cells.append(
                rx.table.cell(
                    rx.cond(
                        item[column.key].to(int) > 0,
                        rx.text(item[column.key].to(int), size="1"),
                        rx.text(column.empty_text, size="1"),
                    ),
                ),
            )
        else:
            cells.append(
                rx.table.cell(rx.text(item[column.key].to(str), size="1")),
            )
    return rx.table.row(*cells)
