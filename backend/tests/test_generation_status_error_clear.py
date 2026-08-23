"""
Regression test for history.update_generation_status() leaving a stale
``error`` value on rows that go on to complete successfully.

The startup sweep in app.py (``_run_startup``) marks any row still stuck in
``generating``/``loading_model`` as ``failed`` with
error="Server was shut down during generation". If that same row later goes
on to complete successfully, the "completed" write must not leave the old
error text behind -- a completed generation with a stale error field is
misleading to any client reading its history.

Usage:
    python -m pytest backend/tests/test_generation_status_error_clear.py -v
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Repo root on sys.path so ``backend`` imports as a package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import Base, Generation, VoiceProfile
from backend.services import history


@pytest.fixture
def db_session(tmp_path):
    """Isolated in-memory-style sqlite session, independent of the app DB."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = session_local()
    session.add(VoiceProfile(id="profile-1", name="Test Profile"))
    session.add(
        Generation(
            id="gen-stale-error",
            profile_id="profile-1",
            text="hello world",
            audio_path="",
            status="generating",
            error="Server was shut down during generation",
        )
    )
    session.commit()

    yield session
    session.close()


@pytest.mark.asyncio
async def test_completing_a_generation_clears_stale_error(db_session):
    """A successful run must clear any error left over from a prior attempt."""
    result = await history.update_generation_status(
        generation_id="gen-stale-error",
        status="completed",
        db=db_session,
        audio_path="generations/gen-stale-error_processed.wav",
        duration=13.68,
    )

    assert result.status == "completed"
    assert result.error is None

    reloaded = db_session.query(Generation).filter_by(id="gen-stale-error").first()
    assert reloaded.status == "completed"
    assert reloaded.error is None


@pytest.mark.asyncio
async def test_failing_a_generation_still_records_the_error(db_session):
    """The failure path is untouched -- an explicit error must still stick."""
    result = await history.update_generation_status(
        generation_id="gen-stale-error",
        status="failed",
        db=db_session,
        error="Generation cancelled",
    )

    assert result.status == "failed"
    assert result.error == "Generation cancelled"
