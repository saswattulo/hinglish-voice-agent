import datetime as dt

from app.clinic import Clinic
from app.llm import PhraseChunker
from app.metrics import aggregate, percentile

MONDAY = dt.date(2026, 11, 2)


def test_chunker_cuts_first_phrase_early_then_at_sentences():
    c = PhraseChunker()
    out = []
    for piece in ["जी, Dr. Sharma ", "का appointment, ", "कल सुबह दस बजे ", "free है। ", "Confirm ", "करूँ?"]:
        out += c.push(piece)
    out.append(c.flush())
    # First cut at the first comma past the minimum length, no more comma cuts after that.
    assert out[0] == "जी, Dr. Sharma का appointment, "
    assert out[1] == "कल सुबह दस बजे free है। "
    assert out[2] == "Confirm करूँ?"


def test_chunker_does_not_split_on_dr_abbreviation_without_space_logic():
    c = PhraseChunker(first_min_chars=100)
    assert c.push("Dr.Sharma") == []


def test_chunker_hard_cuts_very_long_text():
    c = PhraseChunker(max_chars=20)
    out = c.push("a" * 5 + " " + "b" * 30)
    assert out and len(out[0]) <= 20


def test_clinic_availability_and_booking():
    clinic = Clinic(today=MONDAY)
    a = clinic.check_availability("sharma", "2026-11-03")
    assert a["ok"] and "10:00" in a["free_slots"]
    b = clinic.book_appointment("Dr. Sharma", "2026-11-03", "10:00", "Ravi", "+91 98450 12345")
    assert b["ok"] and b["phone"] == "9845012345"
    again = clinic.book_appointment("Dr. Sharma", "2026-11-03", "10:00", "Asha", "9845000000")
    assert not again["ok"]
    assert "10:00" not in clinic.check_availability("Dr. Sharma", "2026-11-03")["free_slots"]


def test_clinic_rejects_sunday_past_and_unknown_doctor():
    clinic = Clinic(today=MONDAY)
    assert not clinic.check_availability("Dr. Sharma", "2026-11-08")["ok"]  # Sunday
    assert not clinic.check_availability("Dr. Sharma", "2026-10-30")["ok"]  # past
    assert not clinic.check_availability("Dr. House", "2026-11-03")["ok"]
    assert not clinic.run_tool("book_appointment", {"doctor": "Dr. Sharma"})["ok"]


def test_percentile_and_aggregate():
    assert percentile([100, 200, 300, 400], 50) == 250
    assert percentile([], 50) is None
    s = {"turns": [{"derived_ms": {"turn_end_to_first_audio_ms": v}} for v in (500, 700, None)]}
    agg = aggregate([s])
    assert agg["turn_end_to_first_audio_ms"] == {"n": 2, "p50": 600, "p95": 690}
