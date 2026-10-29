"""A small in-memory clinic: doctors, open slots and bookings, exposed as Claude tools."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

CLINIC_NAME = "Sanjeevani Clinic"

DOCTORS = {
    "Dr. Sharma": "General Physician",
    "Dr. Iyer": "Paediatrician",
    "Dr. Khan": "Dermatologist",
}

DAILY_SLOTS = ["10:00", "10:30", "11:00", "11:30", "12:00", "16:00", "16:30", "17:00", "17:30"]


@dataclass
class Clinic:
    today: dt.date = field(default_factory=dt.date.today)
    booked: dict[tuple[str, str, str], dict[str, str]] = field(default_factory=dict)

    def _resolve_doctor(self, name: str) -> str | None:
        key = name.lower().replace("dr.", "").replace("dr", "").strip()
        for doc in DOCTORS:
            if key and key in doc.lower():
                return doc
        return None

    def _resolve_date(self, date: str) -> dt.date | None:
        try:
            d = dt.date.fromisoformat(date)
        except ValueError:
            return None
        if d < self.today or d > self.today + dt.timedelta(days=14) or d.weekday() == 6:
            return None  # past, too far out, or Sunday (closed)
        return d

    def check_availability(self, doctor: str, date: str) -> dict[str, Any]:
        doc = self._resolve_doctor(doctor)
        if not doc:
            return {"ok": False, "error": f"Unknown doctor. Available: {', '.join(DOCTORS)}"}
        d = self._resolve_date(date)
        if not d:
            return {"ok": False, "error": "Clinic is closed that day or the date is outside the next 14 days."}
        free = [s for s in DAILY_SLOTS if (doc, d.isoformat(), s) not in self.booked]
        return {"ok": True, "doctor": doc, "date": d.isoformat(), "free_slots": free}

    def book_appointment(
        self, doctor: str, date: str, time: str, patient_name: str, phone: str
    ) -> dict[str, Any]:
        avail = self.check_availability(doctor, date)
        if not avail["ok"]:
            return avail
        if time not in avail["free_slots"]:
            return {"ok": False, "error": "That slot is not free.", "free_slots": avail["free_slots"]}
        digits = "".join(c for c in phone if c.isdigit())
        if len(digits) < 10:
            return {"ok": False, "error": "Phone number needs 10 digits."}
        booking = {
            "booking_id": f"SJV-{len(self.booked) + 101}",
            "doctor": avail["doctor"],
            "date": avail["date"],
            "time": time,
            "patient_name": patient_name,
            "phone": digits[-10:],
        }
        self.booked[(avail["doctor"], avail["date"], time)] = booking
        return {"ok": True, **booking}

    def run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "check_availability":
                return self.check_availability(str(args["doctor"]), str(args["date"]))
            if name == "book_appointment":
                return self.book_appointment(
                    str(args["doctor"]), str(args["date"]), str(args["time"]),
                    str(args["patient_name"]), str(args["phone"]),
                )
        except KeyError as e:
            return {"ok": False, "error": f"Missing argument: {e.args[0]}"}
        return {"ok": False, "error": f"Unknown tool {name}"}


# Tools that change state. They wait until the user's turn is confirmed over
# before running, so a speculative LLM run can never book anything by itself.
SIDE_EFFECT_TOOLS = {"book_appointment"}

TOOLS = [
    {
        "name": "check_availability",
        "description": "List free appointment slots for one doctor on one date.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "doctor": {"type": "string", "description": "Doctor name, e.g. 'Dr. Sharma'"},
                "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            },
            "required": ["doctor", "date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "book_appointment",
        "description": "Book a slot after the patient has confirmed doctor, date, time, name and phone.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "doctor": {"type": "string"},
                "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM from free_slots"},
                "patient_name": {"type": "string"},
                "phone": {"type": "string", "description": "10 digit mobile number"},
            },
            "required": ["doctor", "date", "time", "patient_name", "phone"],
            "additionalProperties": False,
        },
    },
]


def system_prompt(today: dt.date) -> str:
    doctors = "\n".join(f"- {d}: {s}" for d, s in DOCTORS.items())
    return f"""You are Priya, the receptionist on the phone line of {CLINIC_NAME} in Bangalore.
You help callers book an appointment. Today is {today.strftime('%A')}, {today.isoformat()}.
The clinic is open Monday to Saturday, 10:00-12:30 and 16:00-18:00. Closed on Sunday.

Doctors:
{doctors}

How you speak:
- This is a live voice call. Latency-sensitive; begin your visible answer immediately.
- Speak natural Hinglish, the way people in Indian cities talk: Hindi sentence structure mixed with common English words.
- Write Hindi words in Devanagari and English words in Latin script, for example: "जी, Dr. Sharma का appointment कल सुबह दस बजे free है।"
- Keep every reply to one or two short sentences. Ask one thing at a time.
- No lists, markdown, emojis or symbols. Say times and numbers the way you would speak them.
- If the caller speaks only English, reply in simple English.

How you book:
1. Find out what the problem is or which doctor they want, and which day.
2. Use check_availability before offering times. Offer at most two slots.
3. Collect the patient's name and 10 digit mobile number. Read the number back to confirm.
4. Only call book_appointment after the caller says yes to the exact doctor, date and time.
5. After booking, give the booking ID and say goodbye politely.
Never invent slots or bookings. For medical emergencies, tell them to call 108 immediately."""


GREETING = "नमस्ते, Sanjeevani Clinic में आपका स्वागत है। मैं Priya बोल रही हूँ, बताइए मैं आपकी क्या मदद कर सकती हूँ?"
