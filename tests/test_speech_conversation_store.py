import sqlite3
from pathlib import Path

from survng.app.speech.store import SpeechStore


def _chunk(
    store: SpeechStore,
    conversation_id: str,
    index: int,
    *,
    text: str = "",
    status: str = "done",
    final: bool = False,
    interrupted: bool = False,
    camera_id: str = "door",
    offset: int = 0,
) -> int:
    start = 100 + offset + index * 29
    row_id = store.insert_segment(
        camera_id=camera_id, started_at=start, ended_at=start + 30,
        peak_probability=0.9, status=status, conversation_id=conversation_id,
        chunk_index=index, final=final, interrupted=interrupted,
    )
    if text:
        store.set_transcript(
            row_id, status=status, text=text, language="en",
            score=None, model="tiny.en", device="CPU",
        )
    return row_id


def test_conversation_search_returns_whole_group_without_moving_its_cursor(tmp_path: Path) -> None:
    store = SpeechStore(tmp_path)
    first = _chunk(store, "one", 0, text="Bring the Blue box")
    other = _chunk(store, "two", 0, text="Talk later", final=True, camera_id="garage")
    _chunk(store, "one", 1, text="blue box downstairs.", final=True)
    _chunk(store, "filler", 0, text="Thanks for watching!", status="low_confidence", final=True)

    page = store.list_segments(view="conversation", statuses=["done"], limit=1)
    assert [item["id"] for item in page] == [other]
    [conversation] = store.list_segments(view="conversation", before_id=other)
    assert conversation["id"] == first
    assert conversation["text"] == "Bring the Blue box downstairs."
    assert conversation["duration_seconds"] == 59
    assert conversation["chunk_count"] == 2
    assert conversation["conversation_open"] is False
    assert conversation["transcript_gaps"] == []

    for fts_enabled in (store.fts_enabled, False):
        store.fts_enabled = fts_enabled
        assert store.list_segments(view="conversation", query="Bring downstairs") == [conversation]
        assert store.list_segments(view="conversation", query="watching") == []
    assert store.list_segments(view="conversation", query="downstairs", camera_ids=["garage"]) == []


def test_conversation_gaps_keep_partial_text_across_failures_and_retention(tmp_path: Path) -> None:
    store = SpeechStore(tmp_path)
    first = _chunk(store, "one", 0, text="Turn the lights.")
    _chunk(store, "one", 1, status="skipped")
    expired = _chunk(store, "one", 2, text="A private detail.")
    _chunk(store, "one", 3, status="no_text")
    last = _chunk(store, "one", 4, text="the lights off.", final=True, interrupted=True)
    store.expire_content([expired])
    assert not store.set_transcript(
        expired, status="done", text="Another private detail.",
        language="en", score=None, model="tiny.en", device="CPU",
    )
    [partial] = store.list_segments(view="conversation")
    assert partial["text"] == "Turn the lights. […] the lights off."

    store.delete_segments([first])
    [conversation] = store.list_segments(view="conversation")
    assert conversation["status"] == "done"
    assert conversation["text"] == "the lights off."
    assert [gap["reason"] for gap in conversation["transcript_gaps"]] == [
        "missing", "skipped", "expired", "interrupted",
    ]
    assert not conversation["conversation_open"]
    assert not store.list_segments(view="conversation", query="private")

    store.delete_segments([last])
    [conversation] = store.list_segments(view="conversation")
    assert not conversation["conversation_open"]
    assert conversation["transcript_gaps"][-1]["reason"] == "missing"


def test_restart_closes_open_conversations_and_preserves_complete_ones(tmp_path: Path) -> None:
    store = SpeechStore(tmp_path)
    _chunk(store, "closed", 0, text="Already complete.", final=True)
    first = _chunk(store, "open", 0, text="Keep this text.")
    _chunk(store, "open", 1, status="queued")

    [conversation, _] = store.list_segments(view="conversation")
    assert conversation["conversation_open"]
    assert conversation["status"] == "queued"

    store = SpeechStore(tmp_path)
    store.abandon_pending("restart")
    store.close_unfinished_conversations()
    [conversation, closed] = store.list_segments(view="conversation")
    assert conversation["id"] == first
    assert conversation["text"] == "Keep this text."
    assert not conversation["conversation_open"]
    assert [gap["reason"] for gap in conversation["transcript_gaps"]] == ["skipped", "interrupted"]
    assert not closed["transcript_gaps"]


def test_legacy_segment_and_incident_timing_views_stay_distinct(tmp_path: Path) -> None:
    store = SpeechStore(tmp_path)
    legacy = store.insert_segment(
        camera_id="door", started_at=90, ended_at=92, peak_probability=0.9, status="detected",
    )
    utterance = store.insert_segment(
        camera_id="door", started_at=103, ended_at=105, peak_probability=0.9,
        status="detected", transcript_in_conversation=True,
    )
    chunk = _chunk(store, "one", 0, text="Hello there.", final=True)
    assert [row["id"] for row in store.list_segments()] == [utterance, legacy]
    assert [row["id"] for row in store.list_segments(view="transcript", start=102, end=110)] == [chunk]
    assert [row["id"] for row in store.list_segments(view="conversation")] == [chunk, legacy]
    assert store.timing_rows("door", 90, 150) == [
        (legacy, 90.0, 92.0, "detected", False),
        (utterance, 103.0, 105.0, "detected", True),
    ]


def test_schema_migration_preserves_legacy_transcripts(tmp_path: Path) -> None:
    database = sqlite3.connect(tmp_path / "speech.sqlite3")
    try:
        database.executescript(
            """
            create table speech_metadata (key text primary key, value text not null);
            insert into speech_metadata values ('schema_version', '3');
            create table speech_segments (
                id integer primary key autoincrement,
                camera_id text not null,
                started_at real not null,
                ended_at real not null,
                peak_probability real not null default 0,
                status text not null,
                text text not null default '',
                language text not null default '',
                score real,
                model text not null default '',
                device text not null default '',
                reason text not null default '',
                created_at real not null,
                updated_at real not null,
                content_expired integer not null default 0,
                voiced_dbfs real,
                noise_dbfs real
            );
            insert into speech_segments
                (camera_id, started_at, ended_at, status, text, created_at, updated_at, voiced_dbfs)
            values ('door', 90, 92, 'done', 'Original transcript', 90, 92, -37);
            """
        )
    finally:
        database.close()
    store = SpeechStore(tmp_path)
    [segment] = store.list_segments(view="conversation")
    assert segment["text"] == "Original transcript"
    assert segment["voiced_dbfs"] == -37
    assert "conversation_id" not in segment
    assert store.get(segment["id"]) == segment
