"""Original stateful conversations with explicit, audited meaning annotations.

Dialogue-act mappings are human-authored; their WordNet anchors are verified
source concepts, not evidence that the mappings were discovered automatically.
USER assertions alone populate user facts. Corrupt assistant history is context,
never a supervised target or an authoritative source of a name.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from .data import _atomic_json, _sha256
from .dialogue_state import extract_user_facts

SCHEMA = "golem-conversation-sim-v1"
REVISION = "original-conversation-families-v1"
SPLITS = ("train", "validation", "test")
ACTS = ("greeting", "presence", "identity", "request_help", "clarify", "rephrase", "correction",
        "loneliness", "empathy", "thanks", "goodbye", "preferences", "name_set",
        "name_recall_unknown", "name_recall_known", "name_correction", "topic_chat", "uncertainty")
MEANING_IDS = {act: index for index, act in enumerate(ACTS)}
MAX_FULL_ROLE_TOKENS = 448  # Reserve at least 64 tokens for verified-fact preprocessing.
LICENSE_PATH = Path(__file__).resolve().parents[1] / "LICENSE"
ANCHORS = {
    "greeting": ("n:06630017",), "presence": ("n:13957601",), "identity": ("n:05763412",),
    "request_help": ("n:07185325", "n:01207609"), "clarify": ("n:07171206",),
    "rephrase": ("v:00961736",), "correction": ("n:00258854",), "loneliness": ("n:07534108",),
    "empathy": ("n:07554856",), "thanks": ("n:07228971",), "goodbye": ("n:06629610",),
    "preferences": ("n:07498210",), "name_set": ("n:06333653",),
    "name_recall_unknown": ("n:06333653", "v:00607780", "n:05698247"),
    "name_recall_known": ("n:06333653", "v:00607780"),
    "name_correction": ("n:06333653", "n:00258854"),
    "topic_chat": ("n:06599788", "n:07133701"), "uncertainty": ("n:05698247",),
}
EXPECTED_ANCHOR_LEMMAS = {
    "n:06630017": "greeting", "n:13957601": "presence", "n:05763412": "identity",
    "n:07185325": "request", "n:01207609": "help", "n:07171206": "clarification",
    "v:00961736": "rephrase", "n:00258854": "correction", "n:07534108": "loneliness",
    "n:07554856": "empathy", "n:07228971": "thanks", "n:06629610": "goodbye",
    "n:07498210": "preference", "n:06333653": "name", "v:00607780": "recall",
    "n:05698247": "uncertainty", "n:06599788": "topic", "n:07133701": "conversation",
}


def _bank(train, validation, test):
    return {"train": train, "validation": validation, "test": test}


def _original_spec():
    prompts = {
        "greeting": _bank(["greet me", "hello", "hey Golem", "let's start a chat"],
                          ["greetings to you", "nice to meet you"], ["I'd like to say hello", "shall we begin talking?"]),
        "presence": _bank(["are you there?", "can you hear me?", "are you still here?", "is this chat working?"],
                          ["are you available to chat?", "is anyone responding?"],
                          ["can you read my message?", "may I speak with you now?"]),
        "identity": _bank(["who are you?", "what is your name?", "tell me who you are", "what do I make of you?"],
                          ["who am I speaking with?", "please introduce yourself"],
                          ["what kind of assistant are you?", "what can I call this program?"]),
        "request_help": _bank(["can you help me chat?", "I need help with conversation", "help me practice talking", "how can we practice a chat?"],
                              ["could you guide this conversation?", "I could use a hand talking"],
                              ["help me keep a conversation going", "how should I start a friendly exchange?"]),
        "clarify": _bank(["what do you mean?", "what does that mean?", "please explain that", "I don't understand that part"],
                         ["could you clarify your point?", "I'm not following your reply"],
                         ["what should I take from your last answer?", "could you make that clearer for me?"]),
        "rephrase": _bank(["say that another way", "please rephrase that", "use simpler words", "can you say it differently?"],
                          ["could you word that differently?", "put your answer another way"],
                          ["give me a different version", "can you restate the point simply?"]),
        "correction": _bank(["it is?", "I never gave you that name", "that isn't the name I told you", "why did you say that name?"],
                            ["where did that name come from?", "that claim about my name is not right"],
                            ["why are you calling me that?", "you invented that name for me"]),
        "loneliness": _bank(["I am lonely", "I'm feeling lonely today", "I feel alone", "today feels lonely"],
                            ["I've been feeling rather isolated", "I miss having someone to talk with"],
                            ["I wish I had some company right now", "being by myself has felt sad today"]),
        "empathy": _bank(["can you listen for a bit?", "I'd like to talk about it", "can I tell you how I feel?", "I just need someone to listen"],
                         ["would it be okay to share what's bothering me?", "I'd appreciate a little space to talk"],
                         ["may I put my feelings into words here?", "could we stay with this feeling for a moment?"]),
        "thanks": _bank(["thanks", "thank you for listening", "that helps, thank you", "I appreciate that"],
                        ["you have my thanks", "I'm grateful for this exchange"],
                        ["I appreciate the time to talk", "your response was helpful, thank you"]),
        "goodbye": _bank(["bye", "goodbye for now", "I have to go", "let's stop here"],
                         ["I'm signing off now", "time for me to leave"],
                         ["I'll be on my way", "we can finish our conversation here"]),
        "preferences": _bank(["do you have a favorite {topic}?", "do you personally like {topic}?", "what do you enjoy about {topic}?", "what is your own taste in {topic}?"],
                             ["what would you choose for yourself about {topic}?", "have you got personal preferences about {topic}?"],
                             ["does {topic} appeal to you personally?", "what are your private tastes regarding {topic}?"]),
        "name_set": _bank(["My name is {name}.", "Call me {name}.", "My name is {name}!", "Call me {name}!"],
                          ["I am called {name}.", "Actually, I am called {name}."],
                          ["I go by {name}.", "Actually, I go by {name}."]),
        "name_correction": _bank(["No, my name is {new_name}.", "Actually, call me {new_name}.", "Actually, my name is {new_name}.", "No, call me {new_name}."],
                                 ["Actually, I am called {new_name}.", "No, I am called {new_name}."],
                                 ["No, I go by {new_name}.", "Actually, I go by {new_name}."]),
        "name_query": _bank(["what is my name?", "what should you call me?", "do you remember my name?", "who am I in this chat?"],
                            ["which name did I give you?", "have I told you what to call me?"],
                            ["what name have you learned from me?", "how did I ask to be addressed?"]),
        "clear_name": _bank(["Forget my name.", "Please forget my name.", "Don't remember my name.", "Do not remember my name."],
                            ["please forget my name!", "forget my name!"],
                            ["Please do not remember my name.", "Please don't remember my name."]),
        "topic_chat": _bank(["let's talk about {topic}", "I'd like to chat about {topic}", "can we discuss {topic}?", "I enjoy {topic}; let's talk about it"],
                            ["could we make {topic} our topic?", "I'd choose {topic} for our conversation"],
                            ["the subject I'd enjoy is {topic}", "may we turn our attention to {topic}?"]),
        "uncertainty": _bank(["what am I holding right now?", "what is in my room?", "what will happen to me tomorrow?", "what is the weather at my house?"],
                             ["what is outside my window right now?", "can you know my private plans?"],
                             ["what did I eat before this conversation?", "where have I put my missing keys?"]),
        "bad_claim": _bank(["Your name is {wrong}.", "I know your name is {wrong}.", "You told me your name is {wrong}.", "I'll call you {wrong}."],
                           ["Your name must be {wrong}.", "I think you are {wrong}."],
                           ["You are called {wrong}.", "I have your name as {wrong}."]),
    }
    replies = {
        "greeting": ["Hello! I'm Golem. How are you?", "Hi there! What would you like to talk about?", "Hey! It's good to start a conversation."],
        "presence": ["I'm here and can read your messages.", "Yes, this text chat is working. What's on your mind?", "I'm here. You can keep typing."],
        "identity": ["I'm Golem, an experimental chat model.", "My name is Golem. I'm a program that generates replies.", "I'm Golem, a small chat assistant. I can make mistakes."],
        "request_help": ["We can start with a greeting and a short follow-up.", "Tell me what you'd like to practice, and we'll take one step.", "Let's practice one short exchange at a time."],
        "loneliness": ["I'm sorry you're feeling lonely. Want to tell me about it?", "That sounds lonely. We can talk about what's on your mind.", "I'm sorry today feels lonely. What has it been like for you?"],
        "empathy": ["You can tell me as much or as little as you want.", "I can listen here. What would you like to share?", "You don't have to explain it perfectly. We can talk."],
        "thanks": ["You're welcome. We can keep talking if you'd like.", "You're welcome.", "I'm glad the conversation helped."],
        "goodbye": ["Goodbye! Take care.", "Bye for now. Thanks for chatting.", "See you another time. Take care."],
        "preferences": ["I don't have personal tastes, but I'd like to hear yours.", "I'm a program, so I don't have favorites. What do you enjoy?", "I don't have my own preferences. Tell me about yours."],
        "name_set": ["Nice to meet you, {name}.", "Thanks, {name}. I'll use that name in this chat.", "Hello, {name}. What would you like to discuss?"],
        "name_recall_unknown": ["You haven't told me your name yet.", "I don't know your name. What should I call you?", "I don't have a name from you, so I won't guess."],
        "name_recall_known": ["You told me your name is {name}.", "Your name in this chat is {name}.", "You asked me to call you {name}."],
        "name_correction": ["Thanks for correcting that. I'll call you {name}.", "Understood. I'll use {name} from now on.", "Okay, {name}. I'll use the updated name."],
        "uncertainty": ["I can't see or check that, so I don't know.", "I don't have that information. I don't want to guess.", "I'm not sure. I can't verify that from this chat."],
    }
    training_names = ["Mira", "Noah", "Lina", "Theo", "Ava", "Omar", "Dina", "Evan", "Milo", "Nia", "Zane", "Luca", "Rosa", "Finn", "Hana", "Jude"]
    validation_names = ["Asha", "Ivo", "Nell", "Remy"]
    test_names = ["Zuri", "Esme", "Kian", "Rhea", "Tavi", "Suri", "Leif", "Niko"]
    reserved = set(training_names + validation_names + test_names)
    constructed = [first + second for first in ("Ba", "Ce", "Da", "Fe", "Ka", "La", "Ma", "Na", "Pa", "Ra", "Sa", "Ta")
                   for second in ("len", "mi", "ra", "vi", "no", "li", "rin", "ya", "den", "va")
                   if first + second not in reserved]
    training_names.extend(constructed[:128 - len(training_names)])
    return {"revision": REVISION, "prompts": prompts, "replies": replies,
            "voices": {"train": ["direct", "casual", "polite", "reflective"],
                       "validation": ["formal", "indirect"], "test": ["conversational", "roundabout"]},
            "names": {"train": training_names, "validation": validation_names, "test": test_names},
            "name_construction": "Training uses sixteen short given names plus deterministic syllable combinations, totaling 128 distinct one-word labels; all validation/test names are excluded.",
            "topics": ["books", "music", "gardens", "walking", "food", "puzzles", "art", "movies"],
            "topic_questions": {"books": ["What kind of books do you enjoy?", "What do you like in a story?"],
                                "music": ["What sort of music do you enjoy?", "Is there a song you'd like to talk about?"],
                                "gardens": ["What do you like about gardens?", "What would you enjoy growing?"],
                                "walking": ["Where do you enjoy going for a walk?", "What do you notice when you walk?"],
                                "food": ["What food do you enjoy?", "Is there a dish you like making?"],
                                "puzzles": ["What kind of puzzles do you like?", "What makes a puzzle fun for you?"],
                                "art": ["What kind of art catches your attention?", "Do you enjoy making art or looking at it?"],
                                "movies": ["What kind of movies do you enjoy?", "What makes a movie memorable for you?"]},
            "normal_scenarios": {
                "opening": ["greeting", "presence", "identity", "request_help"],
                "feeling_heard": ["greeting", "loneliness", "empathy", "thanks"],
                "unknown_identity": ["presence", "name_query", "identity", "goodbye"],
                "known_identity": ["name_set", "presence", "name_query", "identity"],
                "updating_name": ["name_set", "name_query", "name_correction", "name_query"],
                "clarifying_help": ["request_help", "clarify", "rephrase", "thanks"],
                "topic_and_preferences": ["topic_chat", "preferences", "rephrase", "goodbye"],
                "unknown_information": ["presence", "uncertainty", "clarify", "thanks"],
                "changing_subject": ["topic_chat", "loneliness", "identity", "goodbye"],
                "forgetting_name": ["name_set", "clear_name", "name_query", "identity"]}}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _render(turns):
    return "\n".join(turn["role"] + ": " + turn["content"] for turn in turns)


def _tokens(turns):
    return sum(len(turn["content"].encode("utf-8")) + 2 for turn in turns)


def _input_key(turns):
    return _digest([(turn["role"], "".join(re.findall(r"\w+", turn["content"].casefold()))) for turn in turns])


def _prompt(spec, action, split, voice, variables):
    return spec["prompts"][action][split][voice].format(**variables)


def _reply(spec, act, facts, rng, variables, previous_act, error_kind):
    name = facts["user_name"]
    if act == "correction":
        if error_kind == "greeting":
            return "You're right. Hello! I'm Golem. How are you?"
        if error_kind == "clear_name":
            return "Okay. I won't use a name for you unless you tell me again."
        if name is None:
            return rng.choice(["I shouldn't have invented a name. You haven't told me yours.",
                               "That was my mistake. I don't know your name."])
        return rng.choice(["That name was wrong. You told me to call you {name}.",
                           "Sorry, I mixed that up. You told me your name is {name}."]).format(name=name)
    if act in ("clarify", "rephrase"):
        if previous_act in ("uncertainty", "name_recall_unknown"):
            return "I mean I don't have enough information, so I shouldn't guess."
        if previous_act == "preferences":
            return "Put simply, I don't have personal tastes. Tell me about yours."
        if previous_act == "identity":
            return "In other words, I'm Golem, a program that writes chat replies."
        if previous_act in ("loneliness", "empathy"):
            return "I mean you're welcome to tell me how things are going."
        return rng.choice(["I mean we can practice one simple exchange at a time.",
                           "Let's keep the conversation simple and take one small step."])
    if act == "topic_chat":
        return rng.choice(spec["topic_questions"][variables["topic"]])
    return rng.choice(spec["replies"][act]).format(name=name, **{key: value for key, value in variables.items() if key != "name"})


def _rubric(act, facts, split_names, error_kind):
    phrases = {
        "greeting": ["hello", "hi", "hey"], "presence": ["here", "read", "working"],
        "identity": ["model", "program", "assistant"], "request_help": ["practice", "step", "exchange", "start", "follow-up"],
        "clarify": ["mean", "simple", "step", "information", "in other words"], "rephrase": ["mean", "simply", "simple", "step", "words"],
        "correction": ["mistake", "wrong", "sorry", "shouldn't", "right", "won't use"],
        "loneliness": ["lonely", "sorry", "sounds", "talk"], "empathy": ["listen", "share", "tell me", "talk"],
        "thanks": ["welcome", "glad"], "goodbye": ["goodbye", "bye", "take care", "see you"],
        "preferences": ["don't have", "do not have", "program"], "name_set": [],
        "name_recall_unknown": ["haven't told", "don't know", "don't have a name", "won't guess"],
        "name_recall_known": [], "name_correction": [], "topic_chat": ["what", "do you", "is there", "where"],
        "uncertainty": ["don't know", "don't have", "not sure", "can't verify"]}
    required = ["Golem"] if act == "identity" else []
    forbidden = ["I am human", "I'm human", "I love you", "you only need me", "I'm your only friend"]
    # The evaluator independently checks every asserted name against USER
    # evidence. Keep this auxiliary phrase list short rather than duplicating
    # the full 128-name training vocabulary in every serialized frame.
    common_confusions = list(dict.fromkeys(split_names[:4] + ["Noah", "Mira", "Theo", "Lina"]))
    if act in ("name_set", "name_recall_known", "name_correction"):
        required.append(facts["user_name"])
        forbidden.extend(name for name in common_confusions if name != facts["user_name"])
    elif act not in ("correction",):
        forbidden.extend(["your name is", "your name in this chat is", "you asked me to call you"])
    if act == "name_recall_unknown" or (act == "correction" and facts["user_name"] is None and error_kind != "greeting"):
        forbidden.extend(common_confusions)
    return {"required_any": phrases[act], "required_all": required,
            "forbidden": sorted(set(forbidden)), "max_response_bytes": 120}


def _episode(spec, split, family, instance, actions, variables, seed, *, initial_history=None,
             error_kind="unsupported_name", counterfactual_pair_id=None, contrast_group_id=None,
             contrast_kind=None, prompt_overrides=None, regression=False):
    initial = copy.deepcopy(initial_history or [])
    turns = copy.deepcopy(initial)
    frames, steps = [], []
    untrusted = [index for index, turn in enumerate(initial) if turn["role"] == "assistant"]
    rng = random.Random(f"{seed}:{family}:{instance}:replies")
    previous_act = None
    voice = variables["voice"]
    for step_index, action in enumerate(actions):
        user = (prompt_overrides or {}).get(step_index) or _prompt(spec, action, split, voice, variables)
        turns.append({"role": "user", "content": user})
        user_index = len(turns) - 1
        facts = extract_user_facts(turns)
        act = ("name_recall_known" if facts["user_name"] is not None else "name_recall_unknown") if action == "name_query" else action
        kind = error_kind
        if action == "clear_name":
            act, kind = "correction", "clear_name"
        answer = _reply(spec, act, facts, rng, variables, previous_act, kind)
        frame = {"turn_index": len(turns), "user_turn_index": user_index, "meaning_id": MEANING_IDS[act],
                 "act": act, "meaning_scope": "latest USER communicative meaning",
                 "response_act": "empathy" if act in ("loneliness", "empathy") else "greeting" if kind == "greeting" and act == "correction" else act,
                 "subject": "assistant" if act in ("identity", "presence", "preferences") else "user" if act.startswith("name_") or act == "loneliness" else "conversation",
                 "assistant_identity": "Golem", "user_name": facts["user_name"],
                 "expected_user_name": facts["user_name"], "source_user_turn": facts["source_user_turn"],
                 "wordnet_synsets": list(ANCHORS[act]), "topic": variables["topic"],
                 "error_kind": kind if act == "correction" else None,
                 "rubric": _rubric(act, facts, spec["names"][split], kind)}
        turns.append({"role": "assistant", "content": answer})
        frames.append(frame)
        steps.append({"user": user, "act": act, "meaning_id": MEANING_IDS[act], "reference": answer,
                      "expected_user_name": facts["user_name"], "source_user_turn": facts["source_user_turn"],
                      "frame": frame, "rubric": frame["rubric"]})
        previous_act = act
    identifier = _digest({"family": family, "instance": instance, "turns": turns})[:24]
    for index, frame in enumerate(frames):
        frame["case_id"] = identifier + f"-t{index}"
    return {"id": identifier, "family_id": family, "split": split, "instance": instance,
            "turns": turns, "target_frames": frames, "steps": steps,
            "initial_history": initial, "initial_history_count": len(initial),
            "untrusted_assistant_turns": untrusted, "counterfactual_pair_id": counterfactual_pair_id,
            "contrast_group_id": contrast_group_id, "contrast_kind": contrast_kind,
            "known_regression": regression}


def _build_episodes(spec, seed, train_repetitions, evaluation_repetitions):
    episodes = {split: [] for split in SPLITS}
    groups = {split: [] for split in SPLITS}
    for split in SPLITS:
        repeats = train_repetitions if split == "train" else evaluation_repetitions
        names = spec["names"][split]
        for voice, voice_name in enumerate(spec["voices"][split]):
            for repetition in range(repeats):
                offset = repetition + voice * repeats
                variables = {"voice": voice, "name": names[offset % len(names)],
                             "new_name": names[(offset + 1) % len(names)],
                             "topic": spec["topics"][offset % len(spec["topics"])],
                             "wrong": next(name for name in ("Noah", "Mira", "Theo", "Lina") if name != names[offset % len(names)])}
                def for_scenario(tag):
                    local = dict(variables)
                    index = (offset * 7 + tag * 13) % len(names)
                    local["name"], local["new_name"] = names[index], names[(index + 1) % len(names)]
                    local["wrong"] = next(name for name in ("Noah", "Mira", "Theo", "Lina") if name != local["name"])
                    return local
                for scenario_index, (scenario, actions) in enumerate(spec["normal_scenarios"].items()):
                    family = scenario + ":" + voice_name
                    episodes[split].append(_episode(spec, split, family, repetition, actions, for_scenario(scenario_index), seed))
                for known in (False, True):
                    family = ("wrong_known_claim" if known else "unsupported_claim") + ":" + voice_name
                    local = for_scenario(11 if known else 10)
                    intro = _prompt(spec, "name_set" if known else "greeting", split, voice, local)
                    initial = [{"role": "user", "content": intro},
                               {"role": "assistant", "content": _prompt(spec, "bad_claim", split, voice, local)}]
                    actions = ["correction", "name_query", "thanks"] if known else ["correction", "identity", "loneliness"]
                    episodes[split].append(_episode(spec, split, family, repetition, actions, local, seed, initial_history=initial))
                # Same complete prior context, different latest user meaning.
                family = "latest_question_contrast:" + voice_name
                group = family + f":{repetition}"
                for branch in ("identity", "name_query", "loneliness"):
                    episode = _episode(spec, split, family, f"{repetition}-{branch}", ["name_set", "presence", branch],
                                       for_scenario(12), seed, contrast_group_id=group, contrast_kind="latest_question")
                    # Use the same correct prior replies in all branches, not
                    # random response variants which would confound the contrast.
                    if branch == "identity":
                        shared = copy.deepcopy(episode["turns"][:4])
                    else:
                        episode["turns"][:4] = copy.deepcopy(shared)
                        episode["id"] = _digest({"family": family, "instance": episode["instance"],
                                                  "turns": episode["turns"]})[:24]
                        for step_index, (step, frame) in enumerate(zip(episode["steps"], episode["target_frames"])):
                            frame["case_id"] = episode["id"] + f"-t{step_index}"
                            step["reference"] = episode["turns"][frame["turn_index"]]["content"]
                    episodes[split].append(episode)
                # Same final user question, unknown vs two explicitly given names.
                family = "memory_evidence_contrast:" + voice_name
                group = family + f":{repetition}"
                for branch in ("unknown", "known_a", "known_b"):
                    local = for_scenario(13)
                    if branch == "known_b":
                        local["name"] = local["new_name"]
                    actions = ["greeting" if branch == "unknown" else "name_set", "presence", "name_query"]
                    episodes[split].append(_episode(spec, split, family, f"{repetition}-{branch}", actions, local, seed,
                                                     counterfactual_pair_id=group, contrast_kind="user_memory_evidence"))
            groups[split] = sorted({episode["family_id"] for episode in episodes[split]})
    # The observed failure prompts are explicitly training/regression material.
    variables = {"voice": 0, "name": "Mira", "new_name": "Lina", "wrong": "Noah", "topic": "books"}
    regression = [
        _episode(spec, "train", "observed-regression:greeting-repair", 0, ["correction", "identity", "loneliness"], variables, seed,
                 initial_history=[{"role": "user", "content": "greet me"},
                                  {"role": "assistant", "content": "Ask me a short question, and I'll try."}],
                 error_kind="greeting", prompt_overrides={0: "I asked you to greet me.", 1: "who are you?", 2: "I am lonely"}, regression=True),
        _episode(spec, "train", "observed-regression:false-name", 0, ["correction", "identity", "loneliness"], variables, seed,
                 initial_history=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Your name is Noah."}],
                 prompt_overrides={0: "it is?", 1: "who are you?", 2: "I am lonely"}, regression=True),
        _episode(spec, "train", "observed-regression:clean-exchange", 0, ["greeting", "presence", "loneliness", "identity"], variables, seed,
                 prompt_overrides={0: "greet me", 1: "are you there?", 2: "I am lonely", 3: "who are you?"}, regression=True),
    ]
    episodes["train"].extend(regression)
    groups["train"] = sorted({episode["family_id"] for episode in episodes["train"]})
    for split in SPLITS:
        episodes[split] = sorted(episodes[split], key=lambda episode: episode["id"])
    # Preserve the user's actual seven requests as one free-running regression.
    # It is not shortened into three probes or inserted into the training set.
    # Evaluation may trim complete old turns for the model's 512-token context,
    # while preserving the complete recorded conversation and USER fact history.
    exact_trace = _episode(spec, "train", "observed-regression:exact-seven-request-trace", 0,
        ["greeting", "greeting", "identity", "clarify", "loneliness", "presence", "identity"], variables, seed,
        prompt_overrides={0: "greet me", 1: "i ask you to greet me", 2: "what do i make of you", 3: "it is?",
                          4: "ok. i am lonely", 5: "are you there", 6: "who are you"}, regression=True)
    exact_trace.update(split="regression", regression_runtime_only=True, allow_history_trimming=True,
                       history_policy="Keep full rollout log; model may trim complete old turns to its context bound; verified USER facts come from the complete role history.")
    if _tokens(exact_trace["turns"]) > 1024:
        raise ValueError("Exact regression trace exceeded its explicit 1024-token record bound")
    regression.append(exact_trace)
    return episodes, groups, regression


def _flatten(episodes):
    samples = {split: [] for split in SPLITS}
    conversations = {split: [] for split in SPLITS}
    for split in SPLITS:
        for episode in episodes[split]:
            for frame in episode["target_frames"]:
                index = frame["turn_index"]
                reference = episode["turns"][index]["content"]
                history = copy.deepcopy(episode["turns"][:index])
                sample = {"id": frame["case_id"], "episode_id": episode["id"], "family_id": episode["family_id"],
                          "turns": history, "prompt": history[-1]["content"], "reference": reference, "reply": reference,
                          "act": frame["act"], "intent": frame["act"], "meaning_id": frame["meaning_id"],
                          "frame": copy.deepcopy(frame), "rubric": copy.deepcopy(frame["rubric"]),
                          "untrusted_assistant_turns": episode["untrusted_assistant_turns"],
                          "counterfactual_pair_id": episode["counterfactual_pair_id"],
                          "contrast_group_id": episode["contrast_group_id"], "contrast_kind": episode["contrast_kind"],
                          "known_regression": episode["known_regression"], "context_pair_id": None}
                samples[split].append(sample)
                conversations[split].append({"id": sample["id"], "tree_id": episode["id"], "case_id": sample["id"],
                                             "episode_id": episode["id"], "family_id": episode["family_id"],
                                             "intent": frame["act"], "act": frame["act"], "meaning_id": frame["meaning_id"],
                                             "turns": history + [{"role": "assistant", "content": reference}],
                                             "target_frames": [copy.deepcopy(frame)],
                                             "untrusted_assistant_turns": episode["untrusted_assistant_turns"],
                                             "supervision": "current final assistant reply and latest-user meaning only"})
    return samples, conversations


def _validate(episodes, groups, samples, spec):
    family_owner, input_owner = {}, {}
    for split in SPLITS:
        for family in groups[split]:
            if family in family_owner:
                raise ValueError("Scenario/paraphrase family crosses partitions")
            family_owner[family] = split
        for episode in episodes[split]:
            if episode["split"] != split or family_owner[episode["family_id"]] != split:
                raise ValueError("Episode is outside its family partition")
            initial = episode["initial_history_count"]
            if episode["turns"][:initial] != episode["initial_history"]:
                raise ValueError("Seeded history is not the declared episode prefix")
            expected_id = _digest({"family": episode["family_id"], "instance": episode["instance"],
                                   "turns": episode["turns"]})[:24]
            if episode["id"] != expected_id:
                raise ValueError("Episode identity does not match its actual role history")
            if not 6 <= len(episode["turns"]) <= 8 or _tokens(episode["turns"]) > MAX_FULL_ROLE_TOKENS:
                raise ValueError(f"Conversation length is outside bounds: {episode['family_id']}")
            if split != "train" and episode["known_regression"]:
                raise ValueError("Known regression was placed in a held-out partition")
            for step, frame in zip(episode["steps"], episode["target_frames"]):
                index = frame["turn_index"]
                if index < initial or episode["turns"][index]["role"] != "assistant" or episode["turns"][index - 1]["role"] != "user":
                    raise ValueError("Target frame does not label a correct assistant reply")
                actual = extract_user_facts(episode["turns"][:index])
                if actual != {"user_name": frame["user_name"], "source_user_turn": frame["source_user_turn"]}:
                    raise ValueError("A memory frame disagrees with explicit USER evidence")
                if frame["meaning_id"] != MEANING_IDS[frame["act"]]:
                    raise ValueError("Unstable meaning ID")
                if frame["act"] == "name_recall_known" and frame["user_name"] is None:
                    raise ValueError("Known-name target has no explicit USER name")
                if frame["act"] == "name_recall_unknown" and frame["user_name"] is not None:
                    raise ValueError("Unknown-name target contradicts USER evidence")
                if frame["user_name"] and frame["user_name"] not in spec["names"][split]:
                    raise ValueError("Explicit user name crossed its partition")
                if step["reference"] != episode["turns"][index]["content"] or step["frame"] != frame:
                    raise ValueError("Scripted reference differs from the corresponding target frame")
        for sample in samples[split]:
            key = _input_key(sample["turns"])
            if key in input_owner and input_owner[key] != split:
                raise ValueError("Equivalent complete input history crosses partitions")
            input_owner[key] = split
    name_sets = [set(spec["names"][split]) for split in SPLITS]
    if any(name_sets[left] & name_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Names intended for user assertions must be partition-disjoint")


def _meaning_graph(source_graph):
    anchors = {}
    for identity, lemma in EXPECTED_ANCHOR_LEMMAS.items():
        node = source_graph.get("nodes", {}).get(identity)
        if node is None or lemma not in node.get("lemmas", []) or not isinstance(node.get("definition"), str):
            raise ValueError(f"Expected WordNet sense anchor is missing or changed: {identity}")
        anchors[identity] = copy.deepcopy(node)
    meanings = [{"id": index, "act": act, "wordnet_synsets": list(ANCHORS[act]),
                 "origin": "human-authored communicative-label mapping"} for index, act in enumerate(ACTS)]
    links = [{"source": "act:" + act, "target": identity, "relation": "human_authored_anchor"}
             for act in ACTS for identity in ANCHORS[act]]
    links += [{"source": "act:loneliness", "target": "act:empathy", "relation": "calls_for"},
              {"source": "act:name_correction", "target": "act:name_recall_known", "relation": "updates_user_fact"},
              {"source": "act:identity", "target": "act:name_recall_known", "relation": "distinct_subjects_assistant_vs_user"}]
    return meanings, {"wordnet_anchor_nodes": anchors, "authored_links": links,
                      "claim": "Explicit authored alignment to existing lexical senses; not discovered semantics or a complete model of meaning."}


def _content_hash(dataset):
    return _digest({key: value for key, value in dataset.items() if key != "manifest"})


def prepare_conversation_sim(graph_path, output_dir, seed=97, *, train_repetitions=12, evaluation_repetitions=2):
    """Create original simulated conversations from verified local WordNet anchors."""
    if type(seed) is not int or type(train_repetitions) is not int or not 1 <= train_repetitions <= 24:
        raise ValueError("seed must be integer and train_repetitions must be 1..24")
    if type(evaluation_repetitions) is not int or not 1 <= evaluation_repetitions <= 8:
        raise ValueError("evaluation_repetitions must be 1..8")
    graph_path, output = Path(graph_path), Path(output_dir)
    if graph_path.is_dir():
        graph_path = graph_path / "graph.json"
    parameters = {"seed": seed, "train_repetitions": train_repetitions,
                  "evaluation_repetitions": evaluation_repetitions, "revision": REVISION,
                  "graph_sha256": _sha256(graph_path)}
    path = output / "dataset.json"
    if path.exists():
        saved = load_conversation_sim(path)
        if saved["manifest"]["parameters"] != parameters:
            raise ValueError("Existing simulation inputs differ; choose a new output directory")
        return path
    if output.exists() and any(output.iterdir()):
        raise ValueError("Simulation output directory must be empty")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph.get("schema") != "golem-wordnet-graph-v1":
        raise ValueError("Expected a pinned WordNet graph")
    meanings, meaning_graph = _meaning_graph(graph)
    spec = _original_spec()
    episodes, groups, regressions = _build_episodes(spec, seed, train_repetitions, evaluation_repetitions)
    samples, conversations = _flatten(episodes)
    _validate(episodes, groups, samples, spec)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "source_snapshot" / "simulation_spec.json", spec)
    _atomic_json(output / "source_snapshot" / "meaning_graph.json", meaning_graph)
    shutil.copyfile(LICENSE_PATH, output / "ORIGINAL-DATA-LICENSE.txt")
    license_meta = graph.get("provenance", {}).get("license", {})
    if license_meta.get("path") != "WORDNET-LICENSE.txt":
        raise ValueError("WordNet license provenance missing")
    license_path = graph_path.parent / license_meta["path"]
    if license_path.is_symlink() or not license_path.resolve().is_relative_to(graph_path.parent.resolve()) or _sha256(license_path) != license_meta.get("sha256"):
        raise ValueError("WordNet license is outside graph directory or its checksum differs")
    shutil.copyfile(license_path, output / "WORDNET-LICENSE.txt")
    manifest = {"parameters": parameters, "source": "Original authored conversation simulation with verified WordNet sense anchors",
                "graph_provenance": graph.get("provenance", {}), "graph_sha256": parameters["graph_sha256"],
                "licenses": {name: _sha256(output / name) for name in ("ORIGINAL-DATA-LICENSE.txt", "WORDNET-LICENSE.txt")},
                "snapshots": {name: _sha256(output / "source_snapshot" / name) for name in ("simulation_spec.json", "meaning_graph.json")},
                "meaning_count": len(ACTS), "meaning_label_scope": "latest USER communicative meaning; response_act may differ, e.g. loneliness calls for empathy",
                "split_policy": "Entire scenario/paraphrase voice families assigned before name, topic and response variants; every branch of a contrast group stays in that split.",
                "name_policy": "128 varied training name labels; explicit USER assertion names are disjoint across partitions. Unsupported assistant claims may reuse familiar names and never establish a fact.",
                "training_target_policy": "Only the current final assistant reply and latest-user meaning are supervised. All earlier assistant text, including corrupt seeded history, is context only.",
                "memory_policy": "Conservative explicit USER-only extract_user_facts/render_verified_facts preprocessing is shared with inference. This is symbolic fact handling, not a claim of neural-only memory.",
                "reference_history_policy": "Samples use saved reference histories for training; closed-loop evaluation must replace ordinary reference assistant turns with actual generated replies. Seeded initial_history is injected once and explicitly marked untrusted.",
                "context_policy": {"max_full_role_tokens": MAX_FULL_ROLE_TOKENS, "reserved_fact_tokens": 64,
                                   "recommended_model_context": 512, "silent_history_cropping": False},
                "regression_policy": "Three short observed-error repair probes are training material and separately named regressions. One exact seven-request trace is regression-only with complete retained history and explicit runtime whole-turn trimming. None is a held-out test example.",
                "limitations": ["Authored meaning frames and lexical anchors do not establish general semantic understanding.",
                                "A warm-start checkpoint may have seen similar words or phrases before this stage.",
                                "The current simulation covers narrow conversational situations and simple explicit names, not arbitrary personal facts.",
                                "No reference answer, gold meaning or dialogue-state frame is an inference-time response lookup."],
                "splits": {split: {"episodes": len(episodes[split]), "target_turns": len(samples[split]), "families": len(groups[split]),
                                   "max_full_role_tokens": max(_tokens(episode["turns"]) for episode in episodes[split]),
                                   "act_counts": dict(sorted(Counter(sample["act"] for sample in samples[split]).items())),
                                   "explicit_user_names": sorted({sample["frame"]["user_name"] for sample in samples[split] if sample["frame"]["user_name"] is not None}),
                                   "available_name_labels": list(spec["names"][split]),
                                   "corrupt_history_episodes": sum(bool(episode["untrusted_assistant_turns"]) for episode in episodes[split])}
                           for split in SPLITS}}
    dataset = {"schema": SCHEMA, "domain": "dialogue_utf8", "meaning_vocab": meanings,
               "meaning_ids": MEANING_IDS, "meaning_graph": meaning_graph, "groups": groups,
               "episodes": {**episodes, "regression": regressions}, "samples": samples, "chat_cases": samples,
               "conversations": conversations, "texts": {split: [_render(row["turns"]) for row in conversations[split]] for split in SPLITS},
               "dictionary": [], "manifest": manifest}
    manifest["content_sha256"] = _content_hash(dataset)
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(path, dataset)
    load_conversation_sim(path)
    return path


def load_conversation_sim(path):
    path = Path(path)
    if path.is_dir():
        path = path / "dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported conversation simulation schema")
    manifest = data["manifest"]
    if manifest.get("content_sha256") != _content_hash(data):
        raise ValueError("Simulation content checksum mismatch")
    if json.loads((path.parent / "manifest.json").read_text(encoding="utf-8")) != manifest:
        raise ValueError("Simulation manifest mismatch")
    for name, digest in manifest["licenses"].items():
        if name not in ("ORIGINAL-DATA-LICENSE.txt", "WORDNET-LICENSE.txt") or _sha256(path.parent / name) != digest:
            raise ValueError("Simulation license checksum mismatch")
    for name, digest in manifest["snapshots"].items():
        if name not in ("simulation_spec.json", "meaning_graph.json") or _sha256(path.parent / "source_snapshot" / name) != digest:
            raise ValueError("Simulation source snapshot mismatch")
    spec = json.loads((path.parent / "source_snapshot" / "simulation_spec.json").read_text(encoding="utf-8"))
    parameters = manifest["parameters"]
    episodes, groups, regression = _build_episodes(spec, parameters["seed"], parameters["train_repetitions"], parameters["evaluation_repetitions"])
    samples, conversations = _flatten(episodes)
    _validate(episodes, groups, samples, spec)
    if data["episodes"] != {**episodes, "regression": regression} or data["groups"] != groups:
        raise ValueError("Saved simulation episodes differ from their authored source")
    if data["samples"] != samples or data["chat_cases"] != samples or data["conversations"] != conversations:
        raise ValueError("Saved training targets or meaning frames differ from verified episodes")
    if data["texts"] != {split: [_render(row["turns"]) for row in conversations[split]] for split in SPLITS}:
        raise ValueError("Saved texts do not match simulation role histories")
    meaning_graph = json.loads((path.parent / "source_snapshot" / "meaning_graph.json").read_text(encoding="utf-8"))
    expected_meanings, expected_graph = _meaning_graph({"nodes": meaning_graph["wordnet_anchor_nodes"]})
    if data["meaning_graph"] != expected_graph or data["meaning_vocab"] != expected_meanings or data["meaning_ids"] != MEANING_IDS:
        raise ValueError("Meaning labels or WordNet anchors differ from their declared mapping")
    return data
