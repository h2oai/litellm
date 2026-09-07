import sys
import types

import pytest

from litellm.integrations.h2o.lena_routes import mount_lena


class _App:
    pass


@pytest.fixture
def no_lenarouter(monkeypatch):
    """Make `import lenarouter.mount` raise, as it does on a proxy without the package installed"""
    monkeypatch.setitem(sys.modules, "lenarouter", None)
    monkeypatch.setitem(sys.modules, "lenarouter.mount", None)


@pytest.fixture
def fake_lenarouter(monkeypatch):
    """Install a stub `lenarouter.mount` and hand back the call log"""

    calls = []

    def _install(mount_impl):
        pkg = types.ModuleType("lenarouter")
        mod = types.ModuleType("lenarouter.mount")

        def mount(app):
            calls.append(app)
            return mount_impl(app)

        mod.mount = mount
        pkg.mount = mod
        monkeypatch.setitem(sys.modules, "lenarouter", pkg)
        monkeypatch.setitem(sys.modules, "lenarouter.mount", mod)
        return calls

    return _install


def test_absent_package_leaves_the_proxy_startable(no_lenarouter):
    """A proxy without lenarouter is the normal deployment and must boot unchanged

    If the ImportError guard is removed this raises during proxy startup, which takes down every
    deployment that does not use the router; that is the failure this test exists to prevent
    """
    assert mount_lena(_App()) == []


def test_mounted_routes_are_returned_and_the_app_is_the_one_passed(fake_lenarouter):
    """The caller gets the real route list back, and lenarouter is handed the caller's app

    Returning a hardcoded empty list, or mounting onto anything other than the app passed in, both
    pass a test that only checks "did not raise"; neither passes this one
    """
    app = _App()
    calls = fake_lenarouter(lambda a: ["/v1/routers", "/v1/lena/files"])
    assert mount_lena(app) == ["/v1/routers", "/v1/lena/files"]
    assert calls == [app]


def test_a_broken_extension_does_not_stop_the_proxy_booting(fake_lenarouter):
    """mount() blowing up degrades to "no control plane", never to a dead proxy

    This is the case the ImportError guard alone does not cover: the package imports fine and then
    fails while mounting, for instance against a FastAPI whose router API has moved
    """
    fake_lenarouter(lambda a: (_ for _ in ()).throw(RuntimeError("route table moved")))
    assert mount_lena(_App()) == []


def test_a_non_exception_failure_is_not_swallowed(fake_lenarouter):
    """KeyboardInterrupt and SystemExit must still reach the proxy

    `except Exception` is deliberate rather than a bare `except`; widening it would make a shutdown
    signal look like a mount failure and leave the proxy running
    """
    fake_lenarouter(lambda a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        mount_lena(_App())
