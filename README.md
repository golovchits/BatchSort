# BatchSort

A whole table of parcels, sorted in one look.

Built solo for the Meta Global AI Developer Hackathon, Indie track. [Watch the demo video](https://drive.google.com/file/d/1RNKZbUel5u0VYQhPSLz6__LUaNSIV4wo/view?usp=sharing) (2:37).

## The problem

A parcel locker network takes in parcels from every carrier at one hub, then brings each building all of its parcels in one delivery. Before a parcel goes into a locker, a worker checks four things by hand. Who is it for? How big is it? Which compartment fits? Can it share one with another parcel for the same person?

That takes 10 seconds or more per parcel, one parcel at a time. In the Netherlands, the number of parcel lockers almost doubled last year.

## What BatchSort does

The worker puts a table of parcels on a mat, labels up. A phone above the mat sees colour and depth. Each parcel gets a card on a screen as soon as its barcode is seen. Seconds after the mat is still, every card has one of three answers.

- **Green.** The right person, a clear lead over anyone else, and a compartment the parcel fits with room to spare. The card shows the building and compartment, for example "B1 · 4".
- **Amber.** Something is uncertain. The card says what and offers the likely answers. One tap settles it.
- **Red.** No match, or no compartment fits.

A parcel too big for every compartment goes to an agent. It texts the recipient and asks if they can meet the driver at their lockers today. If they say yes, it books that, texts them 30 minutes before, and the driver calls on arrival. If they say no, it offers another day, a spot beside the lockers, or a hold at the hub.

## Built during the hackathon

Everything in this repository was built between 26 October and 4 November 2026. The open-source libraries it uses are listed at the end, with their licences.

All names, codes, buildings and labels are fictional, made by the scripts in `tools/`. The parcels in the video are my own boxes with these printed labels.

## How it works

1. **The mat.** Four printed markers at its corners give the camera real centimetres. The phone hangs 67 cm above it.
2. **The phone app** (`ios/DepthStream`, for iPhone) sends colour and depth about 10 times a second. When the system asks, it also sends one full-size photo.
3. **Barcodes** are read straight from the video. A new code gets a card at once.
4. **Depth** shows when a hand is reaching in and when the mat is still. Once it is still, the system asks for the photo.
5. **SAM 3.1** finds the outline of each parcel in the photo. Depth inside the outline gives its height, so the parcel gets three sides in centimetres.
6. **Muse Spark** reads each label from the photo. It returns the name, the 5-letter locker code, the address and whether the parcel is soft or rigid.
7. **Matching** compares the code and the last name with all 200 fictional people registered at the hub. This is plain code, not the model, so every green traces to a number.
8. **Booking** picks the smallest free compartment the parcel fits. A second parcel for the same person goes into their compartment if both still fit.

Every decision is written to `results/events.jsonl` with its reason and its numbers.

## The rules for green

A parcel is green only if all three hold. Anything else is amber or red.

- **The label was read.** If the tracking number Muse Spark read differs from the barcode, it is amber, because the photo may show a neighbour's label.
- **The match is clear.** The score is the average of how close the code and the last name are, from 0 to 100. Green needs at least 85, and at least 15 points more than the next person.
- **The size fits with margin.** Every side of the parcel is made 2 cm bigger, and it must still fit with 1 cm to spare. Compartments are 30 x 10, 35 x 16, 40 x 22 and 50 x 40 cm (width x height), all 60 cm deep.

The margin only ever moves a parcel to a bigger compartment or to amber, never to a smaller one. A bigger compartment than needed is safe. A compartment too small is a wrong green.

## The agent

The agent runs on Muse Spark Open, served by Baseten. Muse Spark on the Meta Model API also works, with `model: spark` in `configs/default.yaml`.

The model decides what to say and when. The code decides what it is allowed to do. It can only act through five actions. It can look up the parcel, look up the recipient's standing preference, send a text, wait for a reply, and book an outcome. Each action checks the rules and returns an error if one is broken.

- The first text always asks about meeting the driver today. The code adds the question, so the model can't skip it.
- The other choices are offered only after a no.
- It can only book what the recipient was offered and answered, or a hold at the hub once the deadline has passed.
- The code writes the confirmation of every booking, so it only promises what will happen.
- If the model stops without booking, or can't be reached, the parcel is held at the hub and the recipient is told.

Texts go to a simulated phone in the browser, never to a real number. Every text, action and result is logged.

## What I measured

The settings were tuned on 7 dev boxes and 40 label images. Then 6 test parcels the system had never seen were run once, on 4 November. The sorting rules were not changed afterwards. This is a case study on 13 parcels, not a benchmark.

**The test run.** 5 of 6 parcels went green, all 5 to the right person and into a compartment they fit. 0 wrong greens. One label had a mistyped code and still matched the right person. The sixth label had no code, and 11 people share that last name, so it went amber as it should.

- **Size.** Heights within 1.8 cm of a ruler. Footprints up to 7 cm too big, which is the safe side. On one box the height came out 1.8 cm short, so it was booked into a compartment it fits with 1 cm to spare instead of 2. The margin is what kept that fit.
- **Speed.** 5.5 to 10.5 s from the photo to the answer.

**Earlier, on dev parcels.**

- **Height.** Within 0.7 cm on 5 dev parcels, and all 5 got the right compartment size.
- **Footprint.** 1.5 to 5 cm too big on the same parcels. I didn't find the cause. Too big is the safe side, as it never books a compartment that is too small.
- **Labels.** 6 of 6 dev photos read exactly. 0 wrong greens.
- **Matching.** On 40 label images with misspelled names and wrong codes, the right person scored 90 or more and the best wrong person 80. That gap set the 85 and 15.
- **Model times.** Muse Spark reads a label in about 3 s for about $0.006. SAM 3.1 takes about 4 s on a quarter-size photo.
- **The agent.** 4 scripted cases, run live against the model. The recipient meets the driver, picks another day, has a standing preference, or never replies. Each ended in the right booking.

I did not time BatchSort against a worker with a stopwatch. The 10 seconds per parcel comes from my own count of the manual checks.

## What didn't work, and limits

- **Parcels past the edge of the view.** A rule marked any parcel reaching the edge of the camera view amber. On the final setup it fired on almost every box, so I removed it. A parcel that really runs out of view can now be measured short. My only such parcel (63.5 cm long) was dropped from the video.
- **Touching parcels** wait for SAM 3.1 to split them, which adds a few seconds.
- **Mock labels are cleaner than real ones.** Real labels are creased, wet and taped over. Expect more amber.
- **Amber with no code.** When a label has no locker code, up to 13 people share a last name. Nothing on the label tells them apart, so the worker picks. A model guess would only be a guess.
- **The code doesn't check every word the agent writes.** Its first message and its answers to later questions are its own. In one early run it promised a text the system never sends. The code now writes booking confirmations for that reason, but a later answer could still promise too much.
- **One room, one person, 13 parcels.** No real hub, no real carriers, no real recipients.

## Try it

Needs Python 3.11. Install the libraries with `pip install -r requirements.txt`. Copy `.env.example` to `.env` and add a Meta Model API key and a Baseten key.

**Without the phone.** Run the tests (40, no keys needed).

```
python -m pytest
```

Run the agent on its 4 scripted cases. This calls Muse Spark Open, so it needs the Baseten key. It takes about 2 minutes.

```
python -m src.agent --demo
```

**With the phone.** You need an iPhone with a depth sensor (a Pro model) and a Mac on the same Wi-Fi.

1. Print `data/markers.pdf` and tape the 4 markers on a table, 72.5 by 50.5 cm apart (centre to centre).
2. Print labels from `data/mock/labels_dev.pdf` and stick them on boxes.
3. Open `ios/DepthStream` in Xcode, run it on the iPhone, and type in the Mac's address. Mount the phone about 67 cm above the mat, looking down.
4. Run `python app.py`. Open http://localhost:8766 for the worker screen and http://localhost:8766/phone for the simulated phone.

`python app.py --record sessions/name` saves a session, and `--replay sessions/name` plays it back. Recorded sessions are too big for this repository.

## What is where

| Folder | Holds |
|---|---|
| `app.py` | Runs the station. Video, barcodes, depth checks, photos |
| `src/` | The parts: mat, parcels, labels, matching, booking, agent, screen |
| `ios/DepthStream` | The iPhone app, about 190 lines of Swift |
| `web/` | The worker screen and the simulated phone |
| `tools/` | Scripts that made the fictional world, labels and markers |
| `data/mock/` | The fictional world, printable labels and label images |

## Open-source software

| Library | Used for | Licence |
|---|---|---|
| OpenCV (opencv-python) | Markers, image maths | Apache 2.0 |
| NumPy | Depth maths | BSD 3-Clause |
| zxing-cpp | Barcodes from the video | Apache 2.0 |
| RapidFuzz | Name and code matching | MIT |
| meta-sam-parser (Meta) | Reading SAM 3.1 outlines | SAM License |
| httpx | Calls to the model APIs | BSD 3-Clause |
| websockets | Phone to computer, computer to browser | BSD 3-Clause |
| PyYAML | Config and mock data | MIT |
| python-dotenv | Keys from `.env` | BSD 3-Clause |
| ReportLab | Printable labels | BSD |
| python-barcode | Barcodes on the labels | MIT |
| pytest | Tests | MIT |

The iPhone app uses only Apple's own frameworks. The video voice was made with Kokoro (Apache 2.0) and the animation with GSAP and the Inter font (SIL Open Font License). Those tools are not part of this repository.
