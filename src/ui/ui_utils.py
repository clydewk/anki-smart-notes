"""
Copyright (C) 2024 Michael Piazza

This file is part of Smart Notes.

Smart Notes is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Smart Notes is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Smart Notes.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6 import sip

from ..logger import redact_text

if TYPE_CHECKING:
    from anki.notes import Note
    from aqt.editor import Editor

from aqt import (
    QDialog,
    QFont,
    QFormLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    Qt,
    QWidget,
    mw,
)
from aqt.utils import restoreGeom, saveGeom


def show_message_box(
    message: str,
    details: str | None = None,
    custom_ok: str | None = None,
    show_cancel: bool = False,
    copy_button_text: str | None = None,
    copy_button_content: str | None = None,
):
    msg = QMessageBox()
    msg.setText(redact_text(message))

    if details:
        msg.setInformativeText(redact_text(details))

    ok_button = None
    if custom_ok:
        ok_button = QPushButton(custom_ok)
        msg.addButton(ok_button, QMessageBox.ButtonRole.AcceptRole)

    else:
        msg.addButton(QMessageBox.StandardButton.Ok)

    if show_cancel:
        msg.addButton(QMessageBox.StandardButton.Cancel)

    copy_button = None
    if copy_button_text and copy_button_content:
        copy_button = QPushButton(copy_button_text)
        msg.addButton(copy_button, QMessageBox.ButtonRole.ActionRole)

    val = msg.exec()
    clicked_button = msg.clickedButton()

    if clicked_button == copy_button:
        # Copy to clipboard

        if mw and (clipboard := mw.app.clipboard()) is not None:
            clipboard.setText(copy_button_content)
        return False

    return clicked_button == ok_button or val == QMessageBox.StandardButton.Ok


def default_form_layout() -> QFormLayout:
    form = QFormLayout()
    form.setVerticalSpacing(12)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    return form


# UI constants
font_small = QFont()

font_large = QFont()
font_large.setBold(True)

font_bold = QFont()
font_bold.setBold(True)


def editor_is_current(editor: Editor, note: Note) -> bool:
    return bool(editor.web and not sip.isdeleted(editor.web) and editor.note is note)


def scroll_page(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(widget)
    for label in widget.findChildren(QLabel):
        label.setWordWrap(True)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        policy.setHeightForWidth(True)
        label.setSizePolicy(policy)
    layout = widget.layout()
    if layout is not None:
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
    return scroll


def bound_dialog(dialog: QDialog) -> None:
    display = dialog.screen()
    if display is None:
        return
    screen = display.availableGeometry()
    dialog.setMinimumSize(min(600, screen.width()), min(400, screen.height()))
    layout = dialog.layout()
    if layout is not None:
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
    dialog.resize(min(800, screen.width()), min(650, screen.height()))
    if mw and mw.pm.profile is not None:
        profile = mw.pm.profile
        key = f"smart_notes_{type(dialog).__name__}"
        restoreGeom(dialog, key)
        dialog.finished.connect(
            lambda _: saveGeom(dialog, key) if mw and mw.pm.profile is profile else None
        )
    dialog.resize(
        min(dialog.width(), screen.width()), min(dialog.height(), screen.height())
    )
