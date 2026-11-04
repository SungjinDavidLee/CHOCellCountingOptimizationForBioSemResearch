"""전제 조건 부재(Skip)를 pytest 의 skip 으로 변환한다.

테스트 본문은 pytest 의존 없이 작성되어 있어 `python tests/test_pipeline.py`
로도 그대로 돌아간다. pytest 로 돌릴 때만 이 훅이 Skip 을 가로챈다.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_pipeline import Skip  # noqa: E402


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    try:
        pyfuncitem.obj(**{k: pyfuncitem.funcargs[k]
                          for k in pyfuncitem._fixtureinfo.argnames})
    except Skip as e:
        pytest.skip(str(e))
    return True
