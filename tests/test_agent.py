"""The agent's rules live in its tools, so they are tested here without a model.
Each test plays the model's part by calling the tools directly, the way a model's tool calls would."""
import datetime

import src.agent as agent
from src.agent import Job, Phone

PARCEL = {"id": 7, "length": 63.5, "width": 35.5, "height": 29}
NO_PREFERENCE = {"id": "R1", "first": "Jesse", "last": "Hoekstein", "building": "B1", "preference": None}
PREFERS_TOMORROW = {"id": "R2", "first": "Floor", "last": "Wijnhoven", "building": "A1", "preference": "deliver tomorrow"}
TIMING = {"heads_up_s": 0, "arrival_s": 0, "listen_s": 0}


def job(person, deadline_s=60):
    agent.log = lambda event: None  # no writing to results/events.jsonl from tests
    return Job(dict(PARCEL), person, Phone(), deadline_s, lambda: None, TIMING)


def answered(j, reply):
    j.phone.reply(j.person["id"], reply)
    assert j.wait_reply(j.person["id"], 5) == {"reply": reply}


def test_preference_is_booked_without_a_message():
    j = job(PREFERS_TOMORROW)
    assert "error" in j.send_message("R2", "Hi", "meet_driver")
    assert "error" in j.update_booking(7, "hold_at_hub")  # not what they prefer
    assert j.update_booking(7, "tomorrow")["booked"] == "Tomorrow's round"


def test_first_question_is_the_driver_with_its_time():
    j = job(NO_PREFERENCE)
    assert "error" in j.send_message("R1", "Hi", "options")
    assert "error" in j.send_message("R1", "Hi", "none")
    j.send_message("R1", "Your parcel won't fit.", "meet_driver")
    text = j.phone.chats["R1"][0]["text"]
    assert text.startswith("Your parcel won't fit.") and "14:30 and 15:00" in text and "YES or NO" in text


def test_other_options_only_after_the_driver_question_is_answered():
    j = job(NO_PREFERENCE)
    j.send_message("R1", "Hi", "meet_driver")
    assert "error" in j.send_message("R1", "Other options", "options")
    assert "error" in j.update_booking(7, "beside_lockers")  # not offered yet
    answered(j, "No, at work")
    j.send_message("R1", "No problem", "options")
    assert all(label in j.phone.chats["R1"][-1]["text"] for label, _ in agent.OPTIONS.values())
    assert "error" in j.update_booking(7, "beside_lockers")  # offered, but not answered yet
    answered(j, "3")
    assert j.update_booking(7, "beside_lockers")["booked"].startswith("Truck")


def test_yes_books_the_driver_and_the_round_texts_twice():
    j = job(NO_PREFERENCE)
    j.send_message("R1", "Hi", "meet_driver")
    assert "error" in j.update_booking(7, "meet_driver")  # no reply yet
    answered(j, "yes")
    j.update_booking(7, "meet_driver")
    j.listen()  # sends the heads-up and the arrival call, then the job is over
    texts = [line["text"] for line in j.phone.chats["R1"]]
    assert "30 minutes" in texts[-2] and texts[-1].startswith("Call from the driver")


def test_another_day_must_be_a_real_date_within_two_weeks():
    j = job(NO_PREFERENCE)
    j.send_message("R1", "Hi", "meet_driver")
    answered(j, "no")
    j.send_message("R1", "Ok", "options")
    answered(j, "2, next Friday")
    today = datetime.date.today()
    assert "error" in j.update_booking(7, "on_date", "next Friday")
    assert "error" in j.update_booking(7, "on_date", today.isoformat())
    assert "error" in j.update_booking(7, "on_date", (today + datetime.timedelta(days=30)).isoformat())
    assert j.update_booking(7, "on_date", (today + datetime.timedelta(days=4)).isoformat())["booked"].startswith("Delivery on")


def test_silence_means_hold_at_hub():
    j = job(NO_PREFERENCE, deadline_s=0.2)
    j.send_message("R1", "Hi", "meet_driver")
    assert j.wait_reply("R1", 30)["deadline_passed"]
    assert "error" in j.update_booking(7, "tomorrow")
    assert j.update_booking(7, "hold_at_hub")["booked"] == "Hold at hub"


def test_a_message_after_booking_is_passed_on_to_be_answered():
    j = job(NO_PREFERENCE)
    j.timing = {"heads_up_s": 0, "arrival_s": 0, "listen_s": 5}
    j.send_message("R1", "Hi", "meet_driver")
    answered(j, "no")
    j.send_message("R1", "Ok", "options")
    answered(j, "4")
    j.update_booking(7, "hold_at_hub")
    j.phone.reply("R1", "Thanks, until when?")
    assert j.listen() == "The recipient wrote: Thanks, until when?"
    assert j.unanswered()


def test_the_code_confirms_a_booking_and_the_model_cant_add_to_it():
    j = job(NO_PREFERENCE)
    j.send_message("R1", "Hi", "meet_driver")
    answered(j, "no")
    j.send_message("R1", "Ok", "options")
    answered(j, "2, next week")
    day = datetime.date.today() + datetime.timedelta(days=7)
    j.update_booking(7, "on_date", day.isoformat())
    text = j.phone.chats["R1"][-1]["text"]
    assert text.startswith("Booked: delivery on") and "text you" not in text
    assert "error" in j.send_message("R1", "We will text you before we arrive.", "none")
    j.phone.reply("R1", "Thanks")
    assert j.send_message("R1", "You're welcome.", "none") == {"sent": True}


def test_silence_gets_the_held_message_and_a_preference_gets_none():
    j = job(NO_PREFERENCE, deadline_s=0.2)
    j.send_message("R1", "Hi", "meet_driver")
    j.wait_reply("R1", 30)
    j.update_booking(7, "hold_at_hub")
    assert j.phone.chats["R1"][-1]["text"] == agent.HELD
    j = job(PREFERS_TOMORROW)
    j.update_booking(7, "tomorrow")
    assert j.phone.chats == {}


def test_only_this_parcel_and_this_recipient():
    j = job(NO_PREFERENCE)
    assert "error" in j.get_parcel(8)
    assert "error" in j.send_message("R9", "Hi", "meet_driver")
    assert j.phone.chats == {}


def test_message_limit():
    j = job(NO_PREFERENCE)
    j.send_message("R1", "Hi", "meet_driver")
    for _ in range(agent.MAX_MESSAGES - 1):
        j.send_message("R1", "Hello?", "none")
    assert "error" in j.send_message("R1", "Hello?", "none")
