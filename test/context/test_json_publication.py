from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from threading import Event

import pytest

from frame.file_system.textual_data import load_dict_from_json, save_dict_to_json


def test_readers_only_see_complete_json_during_concurrent_publication(tmp_path):
    path = tmp_path / "context.json"
    save_dict_to_json({"revision": -1, "payload": [-1] * 1000}, path)
    start = Event()

    def writer(offset):
        start.wait()
        for revision in range(offset, offset + 30):
            save_dict_to_json(
                {"revision": revision, "payload": [revision] * 1000}, path
            )

    def reader():
        start.wait()
        for _ in range(150):
            data = load_dict_from_json(path)
            assert data["payload"] == [data["revision"]] * 1000

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(writer, 0), pool.submit(writer, 100)]
        futures.extend(pool.submit(reader) for _ in range(2))
        start.set()
        for future in futures:
            future.result()
    assert list(tmp_path.iterdir()) == [path]


def test_failed_json_publication_preserves_previous_document(tmp_path, monkeypatch):
    path = tmp_path / "context.json"
    save_dict_to_json({"successful": True}, path)

    def failed_dump(*args, **kwargs):
        args[1].write('{"partial":')
        raise ValueError("Serialization failed")

    monkeypatch.setattr("frame.file_system.textual_data.dump", failed_dump)
    with pytest.raises(ValueError, match="Serialization failed"):
        save_dict_to_json({"successful": False}, path)
    assert load_dict_from_json(path) == {"successful": True}
    assert list(tmp_path.iterdir()) == [path]


def test_json_read_retries_transient_legacy_write(tmp_path, monkeypatch):
    path = tmp_path / "context.json"
    path.write_text("")
    monkeypatch.setattr(
        "frame.file_system.textual_data.sleep",
        lambda _: save_dict_to_json({"recovered": True}, path),
    )
    assert load_dict_from_json(path) == {"recovered": True}


def test_json_read_does_not_hide_persistent_corruption(tmp_path, monkeypatch):
    path = tmp_path / "context.json"
    path.write_text('{"partial":')
    sleeps = []
    monkeypatch.setattr("frame.file_system.textual_data.sleep", sleeps.append)
    with pytest.raises(JSONDecodeError) as failure:
        load_dict_from_json(path)
    assert len(sleeps) == 2
    assert str(path) in " ".join(failure.value.__notes__)
