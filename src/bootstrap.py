"""
bootstrap.py

Use the zero-shot model to pre-label a bunch of utterance-context pairs
so you don't have to annotate from scratch like some kind of animal.

the idea: write down a bunch of examples that probably violate various maxims,
run the classifier on them, dump a CSV with the model's guesses, then go
through and fix the ones it got wrong. which it will. but fixing is faster
than labeling from nothing, and that's the whole game.

the seed examples here are biased toward flouting because flouting is
interesting and violating is just... sad. i'll add more violating examples
later when i'm in a worse mood.

Usage:
    python bootstrap.py
    python bootstrap.py --output ../data/annotated/bootstrap_labeled.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# same sys.path incantation. at this point it's a tradition.
sys.path.insert(0, str(Path(__file__).parent))

from zero_shot import classify

# utterance-context pairs organized by what i THINK they are.
# the model may disagree. that's the point. disagreement is data.
SEED_PAIRS = [
    # --- Relation flouting ---
    # the classic "i'm not going to answer your question and we both know it" move
    ("The weather is nice today.", "Why were you late to the meeting?"),
    ("Have you tried the new coffee place?", "Did you finish the report?"),
    ("My cat did the funniest thing yesterday.", "Can we talk about the budget?"),
    ("I hear it might rain tomorrow.", "Are you going to apologize?"),
    ("That's a nice shirt.", "Did you read my email?"),
    ("How about those playoffs?", "We need to discuss your performance review."),
    ("I'm thinking about getting a new plant.", "When are you going to pay me back?"),

    # --- Relation violating ---
    # genuinely missing the point. not evasion, just... off.
    ("I had cereal for breakfast.", "What time should we schedule the standup?"),
    ("My sister lives in Portland.", "Have you submitted the pull request yet?"),
    ("Oh that reminds me, I need to buy milk.", "So what's the plan for the migration?"),

    # --- Quantity flouting ---
    # saying less than you know, or more than anyone asked for
    ("Some students passed.", "Did everyone pass the exam?"),
    ("I've been to a country or two.", "Have you traveled much?"),
    ("She's not the worst singer.", "What did you think of her performance?"),
    ("Well, where do I even begin. So first I woke up at 6:47, not 6:45 like usual, and then I brushed my teeth for exactly two minutes...", "How was your morning?"),
    ("He has a pulse.", "How's the new intern doing?"),
    ("There were some issues.", "How did the deployment go?"),

    # --- Quantity violating ---
    # not strategically withholding — just failing to say enough or saying too much
    # without any implicature payoff. the boring cousin of flouting.
    ("Yes.", "Can you walk me through how the authentication flow works?"),
    ("It was okay I guess.", "Can you give me detailed feedback on my presentation?"),
    ("Well the thing is we started the project in January and at first we were thinking about using React but then someone suggested Vue and we had a whole meeting about it and then...", "What framework did you end up using?"),

    # --- Quality flouting ---
    # irony, hyperbole, metaphor — saying something false ON PURPOSE
    ("Oh sure, I LOVE waiting in line for three hours.", "How do you feel about the DMV?"),
    ("I've told you a million times.", "Can you remind me of the password?"),
    ("He's a real Einstein.", "What do you think of the new hire?"),
    ("My heart literally exploded.", "How was the concert?"),
    ("Yeah and I'm the Queen of England.", "I finished the entire project last night."),
    ("That went well.", "The presentation crashed halfway through and the client left."),

    # --- Quality violating ---
    # actually getting things wrong, not on purpose. just... wrong.
    ("The capital of Australia is Sydney.", "What's the capital of Australia?"),
    ("I'm pretty sure the meeting is at 3.", "When is the meeting? It's at 2."),
    ("That word means happy.", "What does 'lugubrious' mean?"),
    ("It's a mammal.", "Is a penguin a mammal?"),
    ("Shakespeare wrote it in the 1800s.", "When was Hamlet written?"),
    ("Antibiotics should clear that right up.", "I've had this cold for a few days. It's viral."),

    # --- Manner flouting/violating ---
    # the sin of being unclear, wordy, or weirdly structured
    ("I may or may not have potentially been in a situation where something could have occurred.", "What happened?"),
    ("The thing with the stuff at the place.", "What are you talking about?"),
    ("First I did step 3, then step 1, then I went back to step 2.", "How did you assemble the shelf?"),
    ("She performed a series of bilateral contractions of the orbicularis oculi muscles.", "What did she do?"),
    ("It's not not possible that it's not untrue.", "Is this correct?"),

    # --- Manner violating ---
    # not being weird on purpose — just genuinely unclear.
    # the difference between a poet and someone who can't write an email.
    ("So basically the thing is that it's like when you have the situation where it's not working.", "What's the bug?"),
    ("I need the report but not the one from last week the other one with the graphs but not the pie charts.", "Which report do you need?"),
    ("We should do it before after the meeting.", "When should we deploy?"),
    ("The function returns the thing that the other function needs to do the thing.", "How does this code work?"),

    # --- Opting out ---
    # "i'm not playing this game." technically uncooperative locally but
    # cooperative in the meta sense — you're being honest about not answering.
    # rare in the wild but theoretically important.
    ("I'm not at liberty to discuss that.", "What happened in the board meeting?"),
    ("I'd rather not say.", "How much do you make?"),
    ("No comment.", "Did you know about the layoffs before they were announced?"),

    # --- Clash ---
    # when two maxims fight and one has to lose.
    # saying "i don't know where she is" when you suspect but can't prove it —
    # quality (don't say what you lack evidence for) wins over quantity
    # (be as informative as required). grice acknowledged this happens
    # but didn't dwell on it. neither will we. much.
    ("Someone told me it might be in building C but I'm not sure.", "Where is the exam?"),
    ("I think she's upset but I don't want to put words in her mouth.", "How does Sarah feel about the reorg?"),
    ("I heard a rumor but I really shouldn't repeat it.", "What's going on with the Johnson account?"),

    # --- Cooperative ---
    # boring but necessary. the control group needs love too.
    ("The meeting is at 3pm in room 204.", "When and where is the meeting?"),
    ("I finished the report and sent it to Sarah.", "What did you do today?"),
    ("Yes, I'll be there.", "Can you make it to dinner tonight?"),
    ("It costs forty-five dollars.", "How much is it?"),
    ("I left them on the kitchen counter.", "Have you seen my keys?"),
    ("No, I haven't heard from her since Tuesday.", "Have you talked to Maria?"),
    ("The train arrives at 5:15.", "When does the train get here?"),
    ("It's three blocks north on the left side of the street.", "How do I get to the pharmacy?"),
    ("We're using PostgreSQL 15 with PostGIS.", "What database are you running?"),
    ("I disagree — I think we should go with option B because it's cheaper.", "Should we go with option A?"),

    # ====== ROUND 3 ======
    # filling in the gaps. violating examples are boring but the model
    # needs to know the difference between "doing it on purpose" and
    # "genuinely failing at communication." also more cooperative examples
    # because 10 is not enough for a majority class baseline.

    # --- Relation violating ---
    # not evasion — just genuinely losing the thread.
    # the conversational equivalent of walking into a room and forgetting why.
    ("Oh I forgot to mention, the printer is broken.", "Can you review this PR by end of day?"),
    ("My dog is three years old.", "What's the status on the API migration?"),
    ("I think the cafeteria closes at 2.", "Have you seen the error logs from last night?"),
    ("We should really organize the supply closet.", "What's our Q3 revenue forecast?"),
    ("The parking lot was full this morning.", "Did the database backup complete?"),

    # --- Quantity violating ---
    # not strategically withholding or over-sharing — just misjudging
    # how much information is appropriate. no implicature, just miscalibration.
    ("Fine.", "Can you tell me everything that happened at the client meeting?"),
    ("Good.", "How did the surgery go? We've been worried sick."),
    ("So anyway after that meeting which was on a Tuesday I think or maybe Wednesday and Sarah was there and also possibly Mark though he might have left early and we discussed the thing about the budget which reminded me of last quarter when we had that other budget issue...", "Did they approve the proposal?"),
    ("Things.", "What did you buy at the store?"),
    ("It went.", "How was your first day at the new job? Tell me everything!"),

    # --- Manner violating ---
    # not artfully obscure — just bad at saying things clearly.
    # the person who writes emails you have to read three times.
    ("The deliverable is to be completed by the stakeholder in alignment with the synergistic framework.", "What do I actually need to do?"),
    ("So like the thing is you go there and then you do that and then the other thing happens.", "How do I set up the development environment?"),
    ("It's in the place where we put the thing last time we did the thing.", "Where's the backup drive?"),
    ("You need to not un-disable the setting.", "How do I turn on notifications?"),
    ("The meeting got rescheduled to before when it was after.", "When is the meeting now?"),

    # --- Quality flouting (more) ---
    # the model struggles here. more irony and hyperbole to train on.
    # every one of these is literally false and everyone knows it.
    ("Oh great, another meeting. Just what I needed.", "We have a sync at 4."),
    ("Wow, you're so fast.", "Sorry it took me three weeks to reply to your email."),
    ("What a surprise.", "The deploy failed again."),
    ("Breaking news: water is wet.", "Did you know the project is over budget?"),
    ("Sure, and pigs fly.", "I'll definitely have it done by Friday."),

    # --- Cooperative (more) ---
    # the model needs more examples of people just... answering the question.
    # radical honesty. appropriate informativeness. what a concept.
    ("It's on the second shelf in the supply room.", "Where do we keep the printer paper?"),
    ("I sent it to the team yesterday around 3pm.", "Did you share the meeting notes?"),
    ("Python 3.11 with FastAPI.", "What's the tech stack for the new service?"),
    ("About forty-five minutes if traffic is normal.", "How long is the drive?"),
    ("She said she'd have it ready by Thursday.", "When is the design review?"),
    ("Two years, mostly on the backend.", "How long have you been on this team?"),
    ("No, we decided to go with the vendor option instead.", "Are we building it in-house?"),
    ("It's in the shared drive under Q3 Reports.", "Where can I find last quarter's numbers?"),

    # ====== ROUND 4 ======
    # going for volume now. 91 examples is not 200 examples.
    # leaning hard into areas the model keeps getting wrong and
    # categories with thin representation. the goal is to get to
    # "fine-tuning might actually work" territory.

    # --- Relation flouting (more variety) ---
    # the model STILL thinks these are Quantity. more examples = more signal.
    # trying different flavors of deflection: humor, discomfort, power moves.
    ("Did you see the game last night?", "We need to talk about your attendance."),
    ("I wonder what's for lunch.", "Have you thought about what I said?"),
    ("Anyway, how's your mom doing?", "Are you going to sign the divorce papers?"),
    ("Look, a squirrel!", "Can you explain why the tests are failing?"),
    ("You know what I love about this office? The lighting.", "Did you take the money from petty cash?"),
    ("Speaking of which, have you been to that new Thai place?", "So about the missing inventory..."),
    ("Isn't it someone's birthday this week?", "When are you going to finish the migration?"),
    ("I really need to clean my desk.", "What did you say to the client?"),

    # --- Quantity flouting (more variety) ---
    # scalar implicatures, strategic understatement, damning with faint praise.
    # the horn scale is doing overtime here.
    ("It's not terrible.", "What do you think of my thesis?"),
    ("I've read a book or two on the subject.", "Are you qualified for this?"),
    ("We've had some feedback.", "How did the users react to the redesign?"),
    ("There have been... developments.", "What happened while I was on vacation?"),
    ("I have thoughts.", "What do you think about the CEO's new strategy?"),
    ("Not everyone was thrilled.", "How did the team take the news?"),
    ("I've seen worse.", "How's the code quality on this legacy project?"),
    ("Parts of it were interesting.", "Did you like the movie?"),

    # --- Quality flouting (more irony/sarcasm/hyperbole) ---
    # the model's nemesis. if it can learn to spot sarcasm it deserves
    # a PhD and tenure. or at least a nice cookie.
    ("Oh absolutely, I live for spreadsheets.", "Can you update the tracking sheet?"),
    ("Clearly I'm the problem here.", "The server crashed after your deploy."),
    ("Because that worked so well last time.", "Let's try the same approach again."),
    ("I'm drowning in free time.", "Can you take on one more project?"),
    ("Oh no, my heart bleeds for you.", "I only got a 15% raise this year."),
    ("What a tragedy.", "The vending machine is out of chips."),
    ("I'm sure it'll be a page-turner.", "We have a new compliance document to review."),
    ("Right, because that's totally how databases work.", "Can't we just delete the duplicates?"),

    # --- Quality violating (more) ---
    # confidently wrong. the most dangerous kind of wrong.
    # no ironic intent, no mutual knowledge of falsity. just... wrong.
    ("The Great Wall of China is visible from space.", "Can you see it from up there?"),
    ("Humans only use 10% of their brains.", "Why can't I remember anything?"),
    ("Lightning never strikes the same place twice.", "Should we worry about that tree?"),
    ("It takes seven years to digest gum.", "I accidentally swallowed my gum."),
    ("We lose most of our body heat through our heads.", "Should I bring a hat?"),
    ("Goldfish have a three-second memory.", "Do you think the fish recognizes me?"),
    ("Cracking your knuckles causes arthritis.", "Can you stop doing that?"),
    ("You should wait 30 minutes after eating to swim.", "Can we go to the pool now?"),

    # --- Manner flouting (more variety) ---
    # deliberately obscure, weirdly structured, or performatively unclear.
    # the verbal equivalent of a modern art installation: you know
    # someone meant something, you're just not sure what.
    ("In a manner of speaking, through a lens of contextual reframing, one might suggest it was suboptimal.", "How was the meeting?"),
    ("Step 1: don't. Step 2: see step 1.", "How should I respond to this angry email?"),
    ("It's like that thing, but the other way, and also sideways.", "Can you describe the architecture?"),
    ("The answer is yes, no, and it depends, in that order.", "Should we launch on Monday?"),
    ("Let me put it this way: imagine a duck.", "Can you explain the billing system?"),
    ("It's not not a problem, but it's not a not-problem either.", "Is this a bug?"),
    ("Think of it as aggressively adequate.", "How would you rate the contractor's work?"),
    ("Chronologically? Alphabetically? By emotional intensity?", "What happened today?"),

    # --- Manner violating (more) ---
    # genuinely unclear communication. not art — just bad signal.
    # the kind of message that makes you close your laptop and stare at the wall.
    ("So the thing with the API is that it does the thing when you call it but sometimes it doesn't do the thing and then you have to do the other thing.", "What's wrong with the API?"),
    ("I put it over there by the thing next to where we had the meeting about the stuff.", "Where did you put the contract?"),
    ("Yeah so basically you just kind of do it and then it works or it doesn't.", "How do I configure the load balancer?"),
    ("The issue is related to the implementation of the functionality pertaining to the user-facing interface.", "What's the bug?"),
    ("First you do the last part, then skip to the middle, then do the beginning.", "How do I file an expense report?"),
    ("It's like email but not email, more like a message but not a text.", "What platform should I use to contact them?"),
    ("The system does what it does when it does it.", "Can you document how the cron job works?"),
    ("We need to leverage the synergies of the cross-functional paradigm shift.", "What's the plan for next quarter?"),

    # --- Cooperative (lots more) ---
    # the boring backbone. you need enough of these that the model learns
    # "most conversation is actually cooperative" as a prior. otherwise
    # it'll see violations everywhere, which is either a bad model
    # or a very cynical worldview. maybe both.
    ("Three bugs, all in the auth module.", "What did you find in the code review?"),
    ("Sure, I'll send it over after lunch.", "Can you share the slide deck?"),
    ("It's a React app with a Node backend.", "What are we building?"),
    ("Tuesday at 2pm works for me.", "When can we meet to discuss the roadmap?"),
    ("I'd recommend the salmon.", "What's good here?"),
    ("About 200 lines, mostly tests.", "How big is the PR?"),
    ("She's on PTO until next Wednesday.", "Is Sarah available for a meeting?"),
    ("Version 3.2, released last month.", "What version of the SDK are you using?"),
    ("It's a known issue, fix is in the next release.", "Why does the export keep failing?"),
    ("Yep, merged it this morning.", "Did the hotfix go out?"),
    ("Chicken parm, if they still have it.", "What do you want for lunch?"),
    ("I think we should wait for the test results first.", "Should we ship this today?"),
    ("The deadline is March 15th.", "When is the proposal due?"),
    ("It's the third door on the right, past the kitchen.", "Where's the conference room?"),
    ("About 12 milliseconds per request.", "What's the average latency?"),
    ("No, that was deprecated in version 4.", "Does the old endpoint still work?"),

    # ====== ROUND 5 ======
    # balancing act. Relation has 23, Quantity has 25, everything else
    # is mid-30s. the model got F1=0.00 on Relation at eval because
    # the unstratified split probably gave it zero Relation examples
    # to evaluate on. more data won't fix the split but it'll make
    # the split less catastrophic. law of large numbers, do your thing.

    # --- Relation flouting (more) ---
    # different flavors of "i heard your question and i am choosing violence"
    ("So I started watching this new show last night.", "Can we talk about what happened at Thanksgiving?"),
    ("Have you noticed the new art in the lobby?", "Why didn't you cc me on that email?"),
    ("I should really start going to the gym.", "Are you going to address the complaint?"),
    ("This coffee is actually pretty good today.", "Did you tell HR about the incident?"),
    ("Oh hey, is that a new phone?", "We need to discuss your expenses."),
    ("I was thinking about taking up pottery.", "What did the auditors say?"),
    ("You know what, I've never been to Iceland.", "Can you explain this charge on the company card?"),
    ("The sunset was gorgeous yesterday.", "Why were the servers down for six hours?"),
    ("I think I'm going to repaint my kitchen.", "Have you told the client about the delay?"),
    ("Did you know octopuses have three hearts?", "When are you going to respond to legal?"),
    ("My neighbor got the cutest puppy.", "We need your incident report by end of day."),
    ("I've been really into sourdough lately.", "What happened to the production database?"),

    # --- Quantity flouting (more) ---
    # strategic underinformativeness, overinformativeness, and the
    # whole spectrum of "i know exactly how much to say and i'm
    # choosing a different amount on purpose"
    ("We had a conversation.", "What did the investors say about the pitch?"),
    ("It could have gone better.", "How did your driving test go?"),
    ("I have some concerns.", "What do you think of the merger?"),
    ("Well, it exists.", "Have you seen the new website design?"),
    ("Let's just say there were surprises.", "How was the product launch?"),
    ("People had feelings.", "How did the layoffs go?"),
    ("It was... an experience.", "How was the team offsite?"),
    ("Some things were said.", "What happened in the meeting with the CEO?"),
    ("I wouldn't say no.", "Do you want to grab dinner?"),
    ("There's room for improvement.", "How did I do on the presentation?"),
    ("Not nothing happened.", "Did anything come of the investigation?"),
    ("Mistakes were made.", "What went wrong with the release?"),
]


def bootstrap(output_path: str):
    """
    run zero-shot on all seed pairs and dump to CSV.
    the 'gold' column is empty — that's for you, human annotator.
    the 'predicted' column is what the model thinks. argue with it.
    """
    results = []

    print(f"Running zero-shot on {len(SEED_PAIRS)} pairs...")
    print("(this loads the model once and then it's fast. ish.)\n")

    for i, (utterance, context) in enumerate(SEED_PAIRS, 1):
        pred = classify(utterance, context)
        results.append({
            "utterance": utterance,
            "context": context,
            "predicted_maxim": pred.predicted_maxim,
            "predicted_violation_type": pred.violation_type,
            "confidence": f"{pred.confidence:.3f}",
            # leave these blank for the human to fill in.
            # the whole point is you look at predicted_maxim and go
            # "yeah" or "no actually that's Relation you fool"
            "gold_maxim": "",
            "gold_violation_type": "",
        })
        # progress dots because waiting in silence is a manner violation
        print(f"  [{i}/{len(SEED_PAIRS)}] {utterance[:50]:<50} → {pred.predicted_maxim} ({pred.confidence:.0%})")

    # write it out
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} pre-labeled examples to {output_path}")
    print("Now go fill in gold_maxim and gold_violation_type.")
    print("When you're done, drop the predicted columns and rename gold → maxim/violation_type")
    print("to match corpus.csv format. or don't and i'll write a script for that too.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap annotation by pre-labeling with zero-shot model.",
        epilog="The model will be wrong sometimes. That's the point. You fix it."
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "annotated" / "bootstrap_labeled.csv"),
        help="Where to write the pre-labeled CSV.",
    )
    args = parser.parse_args()
    bootstrap(args.output)
