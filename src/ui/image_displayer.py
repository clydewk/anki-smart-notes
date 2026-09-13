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

from aqt.qt import QLabel, QPixmap, QResizeEvent, QSizePolicy, Qt, QVBoxLayout, QWidget


class ImageDisplayer(QWidget):
    def __init__(
        self,
        image: bytes | None = None,
        height: int = 500,
        width: int = 500,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.label)
        self._pixmap = QPixmap()
        self.resize(width, height)
        if image:
            self.set_image(image)

    def set_image(self, image: bytes) -> None:
        if not self._pixmap.loadFromData(image):
            self.label.clear()
            raise ValueError("Unable to display the generated image.")
        self._resize_image()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._resize_image()

    def _resize_image(self) -> None:
        if not self._pixmap.isNull():
            self.label.setPixmap(
                self._pixmap.scaled(
                    self.label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
