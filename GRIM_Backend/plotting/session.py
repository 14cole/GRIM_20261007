"""Per-canvas state and lifetime, with a compatibility facade for GUI actions."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlotSession:
    state: dict = field(default_factory=dict)
    runner: object = None
    disposed: bool = False
    resources: list = field(default_factory=list)

    def cancel(self):
        pending = bool(self.state.get("_ordinary_plot_pending", False))
        if self.runner is not None:
            self.runner.cancel()
        self.state["_ordinary_plot_pending"] = False
        return pending

    def own(self, resource):
        self.resources.append(resource)
        return resource

    def release(self):
        resources, self.resources = self.resources, []
        for resource in resources:
            close = getattr(resource, "disconnect", None) or getattr(resource, "close", None)
            if callable(close):
                close()

    def close(self):
        self.disposed = True
        self.cancel()
        if self.runner is not None:
            self.runner.close()
            self.runner = None
        self.release()


_STATE_FIELDS = {
    "_plot_render_generation", "_plot_render_notes", "_plot_line_point_limit",
    "_plot_selection_failed", "_pending_parameter_selection_notice",
    "_ordinary_plot_pending", "_ordinary_plot_emit",
    "_ordinary_plot_fit",
    "_ordinary_plot_slider_before",
    "_plot_markers", "_active_marker", "_marker_drag", "_polar_display_unit",
    "_zoom_box_drag", "_pan_drag",
}


def plot_context_facade(context_type):
    """Old action names resolve directly to their owning context, never copies."""
    def decorate(cls):
        def context(owner):
            values = owner.__dict__
            return values.get("_plot_contexts", {}).get(values.get("_active_plot_tab"))

        for name in context_type.__dataclass_fields__:
            if name == "session":
                continue

            def get(owner, key=name):
                current = context(owner)
                if current is not None:
                    return getattr(current, key)
                if key not in owner.__dict__:
                    raise AttributeError(key)
                return owner.__dict__[key]

            def put(owner, value, key=name):
                current = context(owner)
                if current is not None:
                    setattr(current, key, value)
                else:
                    owner.__dict__[key] = value

            setattr(cls, name, property(get, put))

        for name in _STATE_FIELDS:
            def get(owner, key=name):
                current = context(owner)
                values = current.session.state if current is not None else owner.__dict__
                if key not in values:
                    raise AttributeError(key)
                return values[key]

            def put(owner, value, key=name):
                current = context(owner)
                values = current.session.state if current is not None else owner.__dict__
                values[key] = value

            setattr(cls, name, property(get, put))
        return cls
    return decorate
