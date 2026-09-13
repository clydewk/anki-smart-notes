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

from typing import Any, Generic, TypeVar

from aqt import QComboBox, QSizePolicy, Qt

from .reactive_widget import ReactiveWidget
from .state_manager import StateManager

T = TypeVar("T")


class ReactiveComboBox(ReactiveWidget[T], QComboBox, Generic[T]):
    _fields_key: str
    _selected_key: str

    def __init__(
        self,
        state: StateManager[T],
        fields_key: str,
        selected_key: str,
        render_map: dict[str, str] | None = None,
        # Internally can't use int bc huge ints will cause overflow (thx insane anki deck ids), but
        # pretend to outside consumers
        int_keys: bool = False,
        editable: bool = False,
        **kwargs: Any,
    ):
        if render_map is None:
            render_map = {}
        super().__init__(state, **kwargs)
        self.setEditable(editable)
        self.setMinimumContentsLength(10)
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fields_key = fields_key
        self._selected_key = selected_key
        self.state_to_ui = render_map
        self.ui_to_state = {v: k for k, v in render_map.items()}

        # Bind from state change to view
        state.bind(self)

        self.currentTextChanged.connect(self._on_current_text_changed)

        # Bind from view change to state
        def select(new_value: str) -> None:
            if int_keys and not new_value:
                return
            state.update(
                {self._selected_key: int(new_value) if int_keys else new_value}
            )

        self.on_change.connect(select)

    def _update_from_state(self, updates: dict[str, Any]) -> None:
        fields: list[str] = [str(e) for e in updates[self._fields_key]]
        selected: str = str(updates[self._selected_key])

        if selected not in fields and selected not in {"", "None"}:
            fields.append(selected)
        labels = [self.state_to_ui.get(field, field) for field in fields]
        if labels != [self.itemText(index) for index in range(self.count())]:
            self.clear()
            self.addItems(labels)
        for index, field in enumerate(fields):
            label = self.state_to_ui.get(field, field)
            self.setItemData(
                index,
                f"{label}\n{field}" if label != field else field,
                Qt.ItemDataRole.ToolTipRole,
            )
        if selected in {"", "None"}:
            self.setCurrentIndex(-1)
        else:
            self.setCurrentText(self.state_to_ui.get(selected, selected))
        self.setToolTip(str(self.currentData(Qt.ItemDataRole.ToolTipRole) or ""))

    def _on_current_text_changed(self, text: str) -> None:
        self.setToolTip(str(self.currentData(Qt.ItemDataRole.ToolTipRole) or text))
        if self._state.updating:
            return

        new_state = self.ui_to_state.get(text, text)
        self.on_change.emit(str(new_state))
