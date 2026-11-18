#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import importlib
import json
import threading
from types import SimpleNamespace
from typing_extensions import override
import pytest
from ten_runtime import (
    App,
    TenEnv,
)


class FakeApp(App):
    def __init__(self):
        super().__init__()
        self.event: threading.Event | None = None

    # In the case of a fake app, we use `on_init` to allow the blocked testing
    # fixture to continue execution, rather than using `on_configure`. The
    # reason is that in the TEN runtime C core, the relationship between the
    # addon manager and the (fake) app is bound after `on_configure_done` is
    # called. So we only need to let the testing fixture continue execution
    # after this action in the TEN runtime C core, and at the upper layer
    # timing, the earliest point is within the `on_init()` function of the upper
    # TEN app. Therefore, we release the testing fixture lock within the user
    # layer's `on_init()` of the TEN app.
    @override
    def on_init(self, ten_env: TenEnv) -> None:
        assert self.event
        self.event.set()

        ten_env.on_init_done()

    @override
    def on_configure(self, ten_env: TenEnv) -> None:
        ten_env.init_property_from_json(
            json.dumps(
                {
                    "ten": {
                        "log": {
                            "handlers": [
                                {
                                    "matchers": [{"level": "debug"}],
                                    "formatter": {
                                        "type": "plain",
                                        "colored": True,
                                    },
                                    "emitter": {
                                        "type": "console",
                                        "config": {"stream": "stdout"},
                                    },
                                }
                            ]
                        }
                    }
                }
            ),
        )

        ten_env.on_configure_done()


class FakeAppCtx:
    def __init__(self, event: threading.Event):
        self.fake_app: FakeApp | None = None
        self.event = event


def run_fake_app(fake_app_ctx: FakeAppCtx):
    app = FakeApp()
    app.event = fake_app_ctx.event
    fake_app_ctx.fake_app = app
    app.run(False)


@pytest.fixture(scope="session", autouse=True)
def global_setup_and_teardown():
    event = threading.Event()
    fake_app_ctx = FakeAppCtx(event)

    fake_app_thread = threading.Thread(
        target=run_fake_app, args=(fake_app_ctx,)
    )
    fake_app_thread.start()

    event.wait()

    assert fake_app_ctx.fake_app is not None

    # Yield control to the test; after the test execution is complete, continue
    # with the teardown process.
    yield

    # Teardown part.
    fake_app_ctx.fake_app.close()
    fake_app_thread.join()


@pytest.fixture
def server():
    """A local Kalpa server for one test."""
    from mock_server import MockKalpaServer

    mock = MockKalpaServer().start()
    yield mock
    mock.stop()


class FakeClock:
    """Stands in for the `time` module. It moves only when a test moves it.

    The steps used are multiples of 125 ms, which are exact in binary, so
    the milliseconds computed from them are exact too.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms / 1000


@pytest.fixture
def running():
    """The extension's modules as the TEN runtime runs them.

    The runtime imports the extension as the top-level package
    `kalpa_tts_python`; the tests import it as
    `ten_packages.extension.kalpa_tts_python`. Those are two copies, so a
    test that replaces something in the running extension does it here.
    """
    return SimpleNamespace(
        extension=importlib.import_module("kalpa_tts_python.extension"),
        kalpa_tts=importlib.import_module("kalpa_tts_python.kalpa_tts"),
    )


@pytest.fixture
def clock(monkeypatch, running):
    """The clock the extension and its client read, under the test's hand."""
    fake = FakeClock()
    monkeypatch.setattr(running.extension, "time", fake)
    monkeypatch.setattr(running.kalpa_tts, "time", fake)
    return fake


@pytest.fixture
def config(server):
    """Extension properties that point at the local server."""
    from mock_server import API_KEY, SAMPLE_RATE

    return {
        "params": {
            "api_key": API_KEY,
            "base_url": server.url,
            "sample_rate": SAMPLE_RATE,
        }
    }
