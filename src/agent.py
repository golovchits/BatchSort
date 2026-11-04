"""The recipient agent: sorts out a parcel too big for every compartment before the truck leaves.

A tool-calling loop on Muse Spark Open (Baseten), or Muse Spark (Meta Model API). The model decides
what to do next. The code decides what it may do. The rules live in the
tools, so a model that tries to break one gets an error back instead of a booking:
- A standing preference is applied without a message.
- The first question is always whether the recipient can meet the driver at their lockers today.
  Only after a no are the other 4 options offered. The code adds the question and the options to
  the model's text, so the recipient is never offered anything else.
- The code writes the confirmation of a booking, so it only promises what will happen.
- A booking must be the preference, an offered option the recipient answered, or hold at hub once
  the deadline has passed.
- Every message from the recipient gets an answer. The job keeps listening after it books.
- If the model stops without a booking, the parcel is held at the hub.
Every tool call and message is appended to results/events.jsonl. Messages go to the simulated phone only.

Try it without the camera:  python -m src.agent --demo [--model spark]
"""
import argparse
import datetime
import json
import os
import queue
import time

import httpx
import yaml
from dotenv import load_dotenv

from src.station import log

load_dotenv()

CONFIG = yaml.safe_load(open("configs/default.yaml"))["agent"]
MODELS = {  # url, key name, model, how the key is sent, extra settings
    "open": ("https://inference.baseten.co/v1/chat/completions", "BASETEN_API_KEY", "meta-models/muse-spark-open-1026", "Api-Key", {}),
    "spark": ("https://api.meta.ai/v1/chat/completions", "META_API_KEY", "muse-spark-1.3", "Bearer", {"reasoning_effort": "minimal"}),
}
MEET_DRIVER = "meet_driver"
OPTIONS = {  # offered after a no to the driver: what the recipient reads, and what the hub then does
    "tomorrow": ("Deliver it tomorrow", "Tomorrow's round"),
    "on_date": ("Deliver it on another day: tell us which", "Delivery on {date}"),
    "beside_lockers": ("Leave it beside the lockers today", "Truck, note: leave beside the lockers"),
    "hold_at_hub": ("Hold it at the hub for you to pick up", "Hold at hub"),
}
BOOKINGS = {MEET_DRIVER: "Driver hands it over at the lockers", **{action: done for action, (_, done) in OPTIONS.items()}}
PREFERENCES = {"leave beside the lockers": "beside_lockers", "hold at hub": "hold_at_hub", "deliver tomorrow": "tomorrow"}
MAX_TURNS = 12  # model calls between two things happening. The longest stretch needs about 4
MAX_MESSAGES = 8  # from the model, per parcel. The driver's round messages don't count
LATEST_DAY = 14  # a delivery date at most this many days ahead
DRIVER_MINUTES = 30  # the driver's arrival window, and how long before it the heads-up goes
PICK_UP = "You can pick it up from 18:00 today, then 08:00 to 20:00 every day."
HELD = "We didn't hear back in time, so we're holding your parcel at the hub. " + PICK_UP
CONFIRMATIONS = {  # what the code sends after a booking, so nothing is promised that won't happen
    MEET_DRIVER: "Great, we will bring your parcel to the lockers between {start} and {end} today. "
                 "We will text you 30 minutes before arrival and call when the driver is at the lockers.",
    "tomorrow": "Booked: delivery tomorrow.",
    "on_date": "Booked: delivery on {date}{times}.",
    "beside_lockers": "Booked: the driver will leave your parcel beside your lockers today, between {start} and {end}.",
    "hold_at_hub": "Booked: your parcel stays at the hub. " + PICK_UP,
}

INSTRUCTIONS = """You work at a parcel hub and text the recipient of a parcel too big for every locker
compartment. Sort it out with the tools.
1. Look up the parcel and the recipient's standing preference. If there is one, book it and stop. Don't message.
2. Otherwise send one short message with ask="meet_driver": say the parcel won't fit their locker. The
   question whether they can meet the driver at the lockers today, with the time, is added for you.
3. YES: book meet_driver.
4. NO: send the other options with ask="options". They are added for you, so don't list them. Book the
   one they choose. For another day, book on_date with the date as YYYY-MM-DD.
5. The confirmation of a booking is sent for you, so don't write one. If the recipient writes after that,
   always answer with send_message and ask="none", in one or two sentences, and promise nothing it doesn't say. If a reply is unclear, ask once
   more with ask="none".
Write plain English, short and friendly, no emojis, no dashes. Never book anything the recipient wasn't offered."""

TOOLS = [{"type": "function", "function": {"name": name, "description": description, "parameters": {
    "type": "object", "properties": properties, "required": required}}}
    for name, description, properties, required in [
        ("get_parcel", "The parcel's size, why it won't fit, who it is for, today's date and the driver's times.",
         {"parcel_id": {"type": "integer"}}, ["parcel_id"]),
        ("get_preference", "The recipient's standing preference for parcels that don't fit, or none.",
         {"recipient_id": {"type": "string"}}, ["recipient_id"]),
        ("send_message", "Text the recipient. ask adds a question to the end: meet_driver, options, or none.",
         {"recipient_id": {"type": "string"}, "text": {"type": "string"},
          "ask": {"type": "string", "enum": [MEET_DRIVER, "options", "none"]}}, ["recipient_id", "text", "ask"]),
        ("wait_reply", "Wait for the recipient's reply. Returns it, or says the deadline passed.",
         {"recipient_id": {"type": "string"}, "seconds": {"type": "integer"}}, ["recipient_id", "seconds"]),
        ("update_booking", "Book what happens to the parcel. date only for on_date, as YYYY-MM-DD.",
         {"parcel_id": {"type": "integer"}, "action": {"type": "string", "enum": list(BOOKINGS)},
          "date": {"type": "string"}}, ["parcel_id", "action"]),
    ]
]


class Phone:
    """The simulated phone: what each recipient was sent, and their replies. Thread safe."""

    def __init__(self, on_change=lambda: None):
        self.chats, self.replies, self.on_change = {}, {}, on_change

    def send(self, recipient, text):
        self.chats.setdefault(recipient, []).append({"from": "hub", "text": text, "time": time.time()})
        self.on_change()

    def reply(self, recipient, text):
        self.chats.setdefault(recipient, []).append({"from": "recipient", "text": text, "time": time.time()})
        self.replies.setdefault(recipient, queue.Queue()).put(text)
        self.on_change()

    def wait(self, recipient, seconds):
        try:
            return self.replies.setdefault(recipient, queue.Queue()).get(timeout=max(seconds, 0))
        except queue.Empty:
            return ""


def driver_window(building):
    """When the driver is at this building's lockers today. Sample times until a route planner is
    connected: every building has its own half hour in the afternoon (configs/default.yaml)."""
    start = datetime.datetime.strptime(CONFIG["driver_due"][building], "%H:%M")
    return start.strftime("%H:%M"), (start + datetime.timedelta(minutes=DRIVER_MINUTES)).strftime("%H:%M")


class Job:
    """One parcel, one run of the loop. Holds what has happened so far, which decides what is allowed."""

    def __init__(self, parcel, person, phone, deadline_s, on_change, timing=None):
        self.parcel, self.person, self.phone, self.on_change = parcel, person, phone, on_change
        self.timing = timing or CONFIG["timing"]
        self.deadline = time.time() + deadline_s
        self.preference = PREFERENCES.get(person.get("preference") or "")
        self.window = driver_window(person["building"])
        self.sent, self.offered, self.answered = 0, set(), set()  # answered: offered before a reply came
        self.timed_out, self.booked, self.route = False, None, []  # route: (when, text) the driver's round sends

    def status(self, text):
        self.parcel["agent"] = text
        self.on_change()

    def chat(self):
        return self.phone.chats.get(self.person["id"], [])

    def unanswered(self):
        return bool(self.chat()) and self.chat()[-1]["from"] == "recipient"

    def got(self, reply):
        self.answered |= self.offered
        self.status(f"reply: {reply}")

    def say(self, text, why):
        """A message the code sends, not the model."""
        self.phone.send(self.person["id"], text)
        log({"parcel": self.parcel["id"], "decision": "agent message", "why": why, "text": text, "recipient": self.person["id"]})

    def get_parcel(self, parcel_id):
        if parcel_id != self.parcel["id"]:
            return {"error": f"you are handling parcel {self.parcel['id']} only"}
        p, today = self.parcel, datetime.date.today()
        return {"parcel_id": p["id"], "code": p.get("code", ""), "sides_cm": [round(p[k]) for k in ("length", "width", "height")],
                "why": "too big for every compartment, the largest is 60 x 50 x 40 cm",
                "recipient_id": self.person["id"], "recipient_name": f"{self.person['first']} {self.person['last']}",
                "today": today.strftime("%A %Y-%m-%d"), "driver_at_lockers": f"{self.window[0]} to {self.window[1]}, today and every weekday",
                "hub_pick_up": "from 18:00 today, then 08:00 to 20:00 every day", "truck_leaves_in_s": round(self.deadline - time.time())}

    def get_preference(self, recipient_id):
        if recipient_id != self.person["id"]:
            return {"error": f"the recipient is {self.person['id']}"}
        return {"preference": self.preference or "none"}

    def send_message(self, recipient_id, text, ask):
        if recipient_id != self.person["id"]:
            return {"error": f"the recipient is {self.person['id']}"}
        if self.preference:
            return {"error": "a standing preference applies: book it, don't message"}
        if self.sent >= MAX_MESSAGES:
            return {"error": "no more messages"}
        if self.booked and not self.unanswered():
            return {"error": "the confirmation was sent for you. Write only to answer a message from the recipient"}
        if not self.offered and ask != MEET_DRIVER:
            return {"error": 'first ask whether they can meet the driver, with ask="meet_driver"'}
        if ask == "options" and MEET_DRIVER not in self.answered:
            return {"error": "the other options come only after they answered the driver question"}
        if ask == MEET_DRIVER:
            text += f"\n\nCan you meet the driver at your lockers today between {self.window[0]} and {self.window[1]}? Reply YES or NO."
            self.offered.add(MEET_DRIVER)
        elif ask == "options":
            choices = " ".join(f"{n}) {label}." for n, (label, _) in enumerate(OPTIONS.values(), 1))
            text += f"\n\nReply 1, 2, 3 or 4: {choices}"
            self.offered |= set(OPTIONS)
        self.phone.send(recipient_id, text.strip())
        self.sent += 1
        self.status("messaged the recipient")
        return {"sent": True}

    def wait_reply(self, recipient_id, seconds):
        if recipient_id != self.person["id"]:
            return {"error": f"the recipient is {self.person['id']}"}
        if not self.sent:
            return {"error": "nothing was sent, so no reply will come"}
        reply = self.phone.wait(recipient_id, min(seconds, self.deadline - time.time()))
        if reply:
            self.got(reply)
            return {"reply": reply}
        if time.time() >= self.deadline:
            self.timed_out = True
            return {"reply": None, "deadline_passed": True, "next": "book hold_at_hub"}
        return {"reply": None, "deadline_passed": False, "seconds_left": round(self.deadline - time.time())}

    def update_booking(self, parcel_id, action, date=""):
        if parcel_id != self.parcel["id"] or action not in BOOKINGS:
            return {"error": f"book parcel {self.parcel['id']} with one of {list(BOOKINGS)}"}
        allowed = (action == self.preference or (action in self.answered and not self.preference)
                   or (self.timed_out and action == "hold_at_hub"))
        if not allowed:
            return {"error": "not allowed: book the preference, an option the recipient was offered and answered, "
                             "or hold_at_hub once the deadline has passed"}
        times = ""
        if action == "on_date":
            try:
                day = datetime.date.fromisoformat(date)
            except ValueError:
                return {"error": "give the date as YYYY-MM-DD"}
            if not 1 <= (day - datetime.date.today()).days <= LATEST_DAY:
                return {"error": f"the date must be after today and at most {LATEST_DAY} days ahead"}
            date = day.strftime("%A %d %B")
            if day.weekday() < 5:  # the driver's times are for weekdays
                times = f", between {self.window[0]} and {self.window[1]}"
        because = "preference" if self.preference else "reply" if action in self.answered else "deadline"
        self.book(action, because, date)
        if self.chat():  # a standing preference is booked without a message
            text = HELD if because == "deadline" else CONFIRMATIONS[action].format(start=self.window[0], end=self.window[1], date=date, times=times)
            self.say(text, "booking confirmation")
        if action == MEET_DRIVER:  # the driver's round texts a heads-up and calls on arrival
            self.route = [(time.time() + self.timing["heads_up_s"],
                           f"The driver is on the way with your parcel and will be at your lockers at about {self.window[0]}, "
                           f"in {DRIVER_MINUTES} minutes."),
                          (time.time() + self.timing["arrival_s"],
                           "Call from the driver: I'm at your lockers now with your parcel. See you in a minute.")]
        else:
            self.route = []
        return {"booked": self.parcel["agent_outcome"], "next": "the recipient has been told. Write only if they send another message"}

    def book(self, action, because, date=""):
        self.booked = action
        self.parcel["agent_outcome"] = BOOKINGS[action].format(date=date)
        self.status(f"{self.parcel['agent_outcome']} ({because})")
        log({"parcel": self.parcel["id"], "decision": "agent booking", "action": action, "why": because,
             "outcome": self.parcel["agent_outcome"], "recipient": self.person["id"]})

    def listen(self):
        """Wait for what happens next: the recipient writes, the deadline passes, or the driver's round
        sends a message. Returns the news for the model, or None once the job is over."""
        while True:
            now = time.time()
            if self.route and self.route[0][0] <= now:
                self.say(self.route.pop(0)[1], "the driver's round")
                continue
            if not self.booked and not self.timed_out and now >= self.deadline:
                self.timed_out = True
                return "The truck leaves now and no answer came. Book hold_at_hub."
            if self.route:
                until = self.route[0][0]
            elif not self.booked and not self.timed_out:
                until = self.deadline
            else:  # booked: still answer the recipient for a while after the last message
                if not self.chat():  # booked from a standing preference: nobody to answer
                    return None
                until = self.chat()[-1]["time"] + self.timing["listen_s"]
                if now >= until:
                    return None
            reply = self.phone.wait(self.person["id"], min(until - now, 1))
            if reply:
                self.got(reply)
                return f"The recipient wrote: {reply}"


def call_model(model, messages):
    url, key_name, name, scheme, extra = MODELS[model]
    response = httpx.post(url, headers={"Authorization": f"{scheme} {os.environ[key_name]}"}, timeout=90,
                          json={"model": name, "messages": messages, "tools": TOOLS, **extra})
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def talk(job, model, messages):
    """Let the model call tools until it stops. Returns False if the model could not be reached."""
    for _ in range(MAX_TURNS):
        try:
            reply = call_model(model, messages)
        except (httpx.HTTPError, KeyError) as error:
            log({"parcel": job.parcel["id"], "decision": "agent error", "why": str(error)})
            return False
        calls = reply.get("tool_calls") or []
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls} if calls
                        else {"role": "assistant", "content": reply.get("content") or ""})
        if not calls:
            return True  # the model thinks it is done for now
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
                result = getattr(job, name)(**args) if name in {t["function"]["name"] for t in TOOLS} else {"error": "no such tool"}
            except (TypeError, ValueError) as error:
                args, result = {}, {"error": f"bad arguments: {error}"}
            log({"parcel": job.parcel["id"], "agent_tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return True


def run(parcel, person, phone, model=None, deadline_s=None, on_change=lambda: None, timing=None):
    """Handle one too-big parcel from the first message to the driver's arrival. Blocks, so run it in a thread."""
    model = model or CONFIG["model"]
    job = Job(parcel, person, phone, deadline_s or CONFIG["reply_deadline_s"], on_change, timing)
    job.status("looking at it")
    log({"parcel": parcel["id"], "decision": "agent start", "model": MODELS[model][2], "recipient": person["id"]})
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"Parcel {parcel['id']} is too big for every compartment. Sort it out."}]
    reachable = True
    while reachable:
        reachable = talk(job, model, messages)
        if job.unanswered():  # the model left a message without an answer: once more, then the code answers
            messages.append({"role": "user", "content": "The recipient's last message has no answer yet. Answer it."})
            if not (reachable and talk(job, model, messages)) or job.unanswered():
                job.say("Thanks, we got your message.", "the model did not answer")
        news = job.listen() if reachable else None
        if news is None:
            break
        messages.append({"role": "user", "content": news})
    if not job.booked:  # an error, or the model never booked: hold it, as after a missed deadline
        job.book("hold_at_hub", "agent stopped without a booking")
        if job.chat():
            job.say(HELD, "held at the hub without the model")
    return job.booked


def demo(model):
    """Four cases on fictional parcels, without the camera. Replies come from a script, as soon as the
    hub asks a question. Times are shortened: the deadline is 20 s, the driver arrives after 10."""
    world = yaml.safe_load(open("data/mock/world.yaml"))
    people = [r for r in world["recipients"] if not r.get("preference")]
    with_preference = next(r for r in world["recipients"] if r.get("preference"))
    timing = {"heads_up_s": 5, "arrival_s": 10, "listen_s": 8}
    cases = [("meets the driver", people[18], ["Yes, I'll be home", "On my way down"]),
             ("another day", people[1], ["No, I'm at work all day", f"{datetime.date.today() + datetime.timedelta(days=7):%A %d %B} please"]),
             ("a standing preference", with_preference, []),
             ("silence", people[2], [])]
    for n, (case, person, answers) in enumerate(cases, 901):
        phone, script = Phone(), list(answers)

        def answer(p=person, phone=phone, script=script):  # the recipient answers each question, then the driver's call
            last = (phone.chats.get(p["id"]) or [{}])[-1]
            if script and last.get("from") == "hub" and ("Reply" in last["text"] or "Call from" in last["text"]):
                phone.reply(p["id"], script.pop(0))
        phone.on_change = answer
        parcel = {"id": n, "code": f"NW00000000{n}", "length": 63.5, "width": 35.5, "height": 29}
        started = time.time()
        outcome = run(parcel, person, phone, model, deadline_s=20, timing=timing)
        print(f"\n{case}: {person['id']} -> {parcel['agent_outcome']} ({outcome}) in {time.time() - started:.0f} s")
        for line in phone.chats.get(person["id"], []):
            print(f"   {line['from']}: {line['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--model", choices=list(MODELS))
    arguments = parser.parse_args()
    if arguments.demo:
        demo(arguments.model or CONFIG["model"])
