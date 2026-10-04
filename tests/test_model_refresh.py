import importlib
import logging
from types import SimpleNamespace

import pytest


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    from tinychat.utils import log_utils
    monkeypatch.setattr(log_utils, "build_logger", lambda *args: logging.getLogger("test"))
    server = importlib.import_module("tinychat.serve.gradio_web_server")
    monkeypatch.setattr(server, "args", SimpleNamespace(
        controller_url="http://controller", model_list_mode="reload",
    ), raising=False)
    return server


@pytest.mark.parametrize("selected, models, expected", [
    ("", ["new-worker"], "new-worker"),
    ("second", ["first", "second"], "second"),
    ("removed", ["replacement"], "replacement"),
    ("removed", [], ""),
])
def test_refresh_queries_controller(server, monkeypatch, selected, models, expected):
    calls = []

    def post(url):
        calls.append(url)
        return SimpleNamespace(status_code=200, json=lambda: {"models": models})

    monkeypatch.setattr(server.requests, "post", post)
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    update = server.refresh_model_list(selected, request)
    assert update["choices"] == models
    assert update["value"] == expected
    assert calls == ["http://controller/refresh_all_workers", "http://controller/list_models"]


def test_refresh_button_updates_only_model_selector(server, monkeypatch):
    monkeypatch.setattr(server, "models", [], raising=False)
    # Media examples are unrelated to model discovery and may be absent in sparse checkouts.
    monkeypatch.setattr(server.gr, "Examples", lambda *args, **kwargs: None)
    demo = server.build_demo(embed_mode=True)
    config = demo.get_config_file()
    buttons = [c for c in config["components"] if c["props"].get("value") == "Refresh models"]
    assert len(buttons) == 1
    selector = next(c for c in config["components"] if c["props"].get("label") == "Model")
    event = next(d for d in config["dependencies"] if buttons[0]["id"] in d["targets"])
    assert event["inputs"] == [selector["id"]]
    assert event["outputs"] == [selector["id"]]
    assert event["queue"] is False
