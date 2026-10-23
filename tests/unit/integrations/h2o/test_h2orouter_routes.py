import sys
import types

import pytest

from litellm.integrations.h2o.h2orouter_routes import mount_h2orouter


class _App:
    pass


@pytest.fixture
def no_h2orouter(monkeypatch):
    """Make `import h2orouter.mount` raise, as it does on a proxy without the package installed"""
    monkeypatch.setitem(sys.modules, "h2orouter", None)
    monkeypatch.setitem(sys.modules, "h2orouter.mount", None)


@pytest.fixture
def fake_h2orouter(monkeypatch):
    """Install a stub `h2orouter.mount` and hand back the call log"""

    calls = []

    def _install(mount_impl):
        pkg = types.ModuleType("h2orouter")
        mod = types.ModuleType("h2orouter.mount")

        def mount(app):
            calls.append(app)
            return mount_impl(app)

        mod.mount = mount
        pkg.mount = mod
        monkeypatch.setitem(sys.modules, "h2orouter", pkg)
        monkeypatch.setitem(sys.modules, "h2orouter.mount", mod)
        return calls

    return _install


def test_absent_package_leaves_the_proxy_startable(no_h2orouter):
    """A proxy without h2orouter is the normal deployment and must boot unchanged

    If the ImportError guard is removed this raises during proxy startup, which takes down every
    deployment that does not use the router; that is the failure this test exists to prevent
    """
    assert mount_h2orouter(_App()) == []


def test_mounted_routes_are_returned_and_the_app_is_the_one_passed(fake_h2orouter):
    """The caller gets the real route list back, and h2orouter is handed the caller's app

    Returning a hardcoded empty list, or mounting onto anything other than the app passed in, both
    pass a test that only checks "did not raise"; neither passes this one
    """
    app = _App()
    calls = fake_h2orouter(lambda a: ["/v1/routers", "/v1/h2orouter/files"])
    assert mount_h2orouter(app) == ["/v1/routers", "/v1/h2orouter/files"]
    assert calls == [app]


def test_a_broken_extension_does_not_stop_the_proxy_booting(fake_h2orouter):
    """mount() blowing up degrades to "no control plane", never to a dead proxy

    This is the case the ImportError guard alone does not cover: the package imports fine and then
    fails while mounting, for instance against a FastAPI whose router API has moved
    """
    fake_h2orouter(lambda a: (_ for _ in ()).throw(RuntimeError("route table moved")))
    assert mount_h2orouter(_App()) == []


def test_a_non_exception_failure_is_not_swallowed(fake_h2orouter):
    """KeyboardInterrupt and SystemExit must still reach the proxy

    `except Exception` is deliberate rather than a bare `except`; widening it would make a shutdown
    signal look like a mount failure and leave the proxy running
    """
    fake_h2orouter(lambda a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        mount_h2orouter(_App())
