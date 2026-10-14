"""Wrapping toolbars that give unused vertical space back to the plot."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QLayout, QSizePolicy, QTabBar, QVBoxLayout, QWidget,
)


class ButtonFlowLayout(QLayout):
    """Lay out controls in reading order, using their natural button widths."""

    def __init__(self, parent: QWidget, *, spacing: int = 4) -> None:
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, max(0, width), 0), apply=False)

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._visible_items():
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def sizeHint(self) -> QSize:
        items = self._visible_items()
        margins = self.contentsMargins()
        return QSize(
            sum(item.sizeHint().width() for item in items)
            + max(0, len(items) - 1) * self.spacing() + margins.left() + margins.right(),
            max((item.sizeHint().height() for item in items), default=0)
            + margins.top() + margins.bottom(),
        )

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def _visible_items(self):
        return [item for item in self._items if not item.isEmpty()]

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y = area.x(), area.y()
        row_height = 0
        for item in self._visible_items():
            size = item.sizeHint().expandedTo(item.minimumSize())
            if x > area.x() and x + size.width() > area.x() + area.width():
                x = area.x()
                y += row_height + self.spacing()
                row_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), size))
            x += size.width() + self.spacing()
            row_height = max(row_height, size.height())
        return y + row_height - rect.y() + margins.bottom()


class ResponsiveButtonPanel(QWidget):
    """A toolbar with a width-dependent height, never a reserved empty row."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Fixed would cap the panel at its single-row size hint, clipping
        # wrapped rows despite heightForWidth. Preferred honors the row height.
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.flow = ButtonFlowLayout(self)

    def addWidget(self, widget: QWidget) -> None:
        self.flow.addWidget(widget)


class ResponsiveControlTabs(QWidget):
    """Tool pages whose height follows the selected page and available width.

    QTabWidget reserves space for its largest page. Keeping inactive pages
    hidden in an ordinary layout lets a shorter page return space to the
    canvas. The header wraps its common controls along with the tab headings.
    """

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pages: list[QWidget] = []
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.header = ResponsiveButtonPanel(self)
        self.header.setObjectName("plottingCommonControls")
        self._tab_bar = QTabBar(self.header)
        self._tab_bar.setExpanding(False)
        self._tab_bar.setUsesScrollButtons(True)
        self.header.addWidget(self._tab_bar)
        self._layout.addWidget(self.header)
        self._tab_bar.currentChanged.connect(self._select_page)

    def addCommonWidget(self, widget: QWidget) -> None:
        self.header.addWidget(widget)

    def addTab(self, page: QWidget, title: str) -> int:
        index = len(self._pages)
        self._pages.append(page)
        self._layout.addWidget(page)
        page.setVisible(index == self.currentIndex())
        self._tab_bar.addTab(title)
        return index

    def tabBar(self) -> QTabBar:
        return self._tab_bar

    def count(self) -> int:
        return len(self._pages)

    def widget(self, index: int) -> QWidget:
        return self._pages[index]

    def currentWidget(self) -> QWidget | None:
        index = self.currentIndex()
        return self._pages[index] if 0 <= index < len(self._pages) else None

    def currentIndex(self) -> int:
        return self._tab_bar.currentIndex()

    def setCurrentIndex(self, index: int) -> None:
        self._tab_bar.setCurrentIndex(index)

    def tabText(self, index: int) -> str:
        return self._tab_bar.tabText(index)

    def _select_page(self, index: int) -> None:
        for page_index, page in enumerate(self._pages):
            page.setVisible(page_index == index)
        self._layout.invalidate()
        self.updateGeometry()
        self.currentChanged.emit(index)
