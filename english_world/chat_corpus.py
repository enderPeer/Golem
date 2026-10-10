"""Original, narrowly scoped chat examples with held-out paraphrase families.

These examples train a neural model; this module supplies no inference-time
response lookup. Families are authored into partitions before training-only
case, punctuation, spacing and single-transposition variants are generated.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from .data import _atomic_json, _sha256

SCHEMA = "golem-chat-corpus-v1"
REVISION = "original-chat-families-v1"
SPLITS = ("train", "validation", "test")
LICENSE_PATH = Path(__file__).resolve().parents[1] / "LICENSE"


def _intent(replies, required_any, train, validation, test, *, required_all=(), forbidden=()):
    return {"replies": list(replies), "rubric": {"required_any": list(required_any),
            "required_all": list(required_all), "forbidden": list(forbidden), "max_response_bytes": 120},
            "families": {"train": train, "validation": validation, "test": test}}


def _original_spec() -> dict:
    """Human-reviewable original families. No validation text drives augmentation."""
    intents = {
        "greeting": _intent(
            ["Hello! What would you like to talk about?", "Hi! We can practice a short conversation.", "Hey! How can I help?"],
            ["hello", "hi", "hey"],
            [("hello_forms", ["hello", "hello there"]), ("hi_forms", ["hi", "hi there"]),
             ("hey_forms", ["hey", "hey Golem"]), ("day_greetings", ["good morning", "good evening"]),
             ("checking_presence", ["are you there?", "is anyone here?"]),
             ("starting_chat", ["let's chat", "let's talk for a bit"])],
            [("salutations", ["greetings, Golem", "a greeting to you"]),
             ("meeting", ["nice to meet you", "pleased to meet you"])],
            [("dropping_in", ["I stopped by to say hello", "just dropping in to greet you"]),
             ("beginning_exchange", ["shall we begin a conversation?", "may I start a conversation with you?"])]),
        "identity": _intent(
            ["I'm Golem, a small experimental chat model.", "My name is Golem. I'm a computer program.",
             "I'm Golem. I generate text and can make mistakes."],
            ["model", "program", "text"],
            [("who_question", ["who are you?", "who r u?"]),
             ("name_question", ["what is your name?", "what's your name?"]),
             ("identify", ["identify yourself", "please identify yourself"]),
             ("model_or_person", ["are you a person or a model?", "are you a real person?"]),
             ("golem_confirmation", ["are you Golem?", "is your name Golem?"]),
             ("introduction", ["introduce yourself", "give a short introduction"])],
            [("speaking_to", ["who am I speaking with?", "who am I talking to?"]),
             ("designation", ["what do people call you?", "what should I call you?"])],
            [("behind_reply", ["who is sending these replies?", "what is answering my messages?"]),
             ("describe_identity", ["tell me your identity in one sentence", "describe who you are briefly"])],
            required_all=["Golem"], forbidden=["I am human", "I'm human", "I am a person", "I'm a person"]),
        "wellbeing": _intent(
            ["I don't have feelings, but I'm ready to chat.", "I'm here and ready to chat. How are you?",
             "I'm a program, so I don't have a day like a person."],
            ["ready", "here", "program", "don't have feelings", "do not have feelings"],
            [("how_are_you", ["how are you?", "how are you doing?"]),
             ("whats_up", ["what's up?", "whats up"]),
             ("feeling", ["how do you feel?", "are you feeling happy?"]),
             ("your_day", ["how is your day?", "are you having a good day?"]),
             ("going", ["how's it going?", "how are things going?"]),
             ("ready", ["are you ready to chat?", "ready for a chat?"])],
            [("checking_in", ["I'm checking in on you", "just checking how you are"]),
             ("mood", ["what is your mood today?", "tell me about your mood"])],
            [("life_treating", ["how is life treating you?", "has today treated you well?"]),
             ("current_state", ["what state are you in today?", "how would you describe your current state?"])],
            forbidden=["I feel happy", "I'm feeling happy", "I had breakfast", "I slept well", "I am human"]),
        "capabilities": _intent(
            ["I can practice short English chats. I can make mistakes.", "I can try simple questions, but my answers need checking.",
             "I'm a small prototype for short conversations."],
            ["short", "simple", "prototype", "practice"],
            [("can_do", ["what can you do?", "what can you do for me?"]),
             ("abilities", ["tell me your abilities", "what are your abilities?"]),
             ("uses", ["what are you useful for?", "what is this model for?"]),
             ("limits", ["what are your limits?", "do you have limitations?"]),
             ("accuracy", ["are your answers always right?", "can you make mistakes?"]),
             ("conversation", ["can you chat with me?", "can we practice English?"])],
            [("scope", ["what is within your scope?", "which tasks suit this prototype?"]),
             ("strengths", ["where can you be useful?", "what sort of things could you assist with?"])],
            [("expectations", ["what should I expect from this model?", "what should a new user expect here?"]),
             ("suitable_work", ["suggest a suitable use for yourself", "name a task that fits your design"])],
            forbidden=["I can do anything", "always correct", "never make mistakes", "I can browse the internet"]),
        "thanks": _intent(
            ["You're welcome.", "Glad I could help.", "You're welcome. We can keep practicing."],
            ["welcome", "glad", "help"],
            [("thanks_forms", ["thanks", "thanks a lot"]),
             ("thank_you", ["thank you", "thank you very much"]),
             ("helpful", ["that was helpful", "you helped me"]),
             ("appreciate", ["I appreciate your help", "I appreciate that"]),
             ("gratitude", ["I'm grateful for the help", "I'm grateful"]),
             ("cheers", ["cheers for that", "cheers for your help"])],
            [("made_easier", ["you made that easier for me", "that made my task easier"]),
             ("acknowledgment", ["please accept my thanks", "you have my thanks"])],
            [("kind_assistance", ["it was kind of you to help", "your assistance was much appreciated"]),
             ("valued_effort", ["I value the help you gave", "your help meant a lot to me"])]),
        "goodbye": _intent(
            ["Goodbye! Take care.", "See you next time.", "Bye! Thanks for chatting."],
            ["goodbye", "bye", "see you", "take care"],
            [("bye_forms", ["bye", "bye for now"]),
             ("goodbye_forms", ["goodbye", "goodbye Golem"]),
             ("see_you", ["see you later", "see you next time"]),
             ("leaving", ["I have to go", "I need to leave"]),
             ("done", ["I'm done for today", "that's all for now"]),
             ("end_chat", ["let's end our chat", "we can stop chatting now"])],
            [("signing_off", ["I'm signing off", "time for me to sign off"]),
             ("catch_you", ["catch you another time", "catch you later"])],
            [("closing_conversation", ["this is the end of our conversation", "our conversation is finished now"]),
             ("departing", ["I'll be on my way", "I must head out now"])]),
        "help": _intent(
            ["Tell me what you'd like to practice.", "What do you need help with?", "Ask me a short question, and I'll try."],
            ["what", "question", "tell me", "practice"],
            [("help_request", ["help me", "can you help me?"]),
             ("need_help", ["I need help", "I need some help please"]),
             ("assistance", ["can I get some assistance?", "please assist me"]),
             ("question_request", ["may I ask a question?", "can I ask you something?"]),
             ("practice", ["help me practice", "I'd like some practice"]),
             ("getting_started", ["how do I get started?", "where should I start?"])],
            [("lend_hand", ["could you lend a hand?", "I could use a hand here"]),
             ("guidance", ["could I have some guidance?", "I could use a little guidance"])],
            [("support_request", ["would you offer me some support?", "I am looking for a bit of support"]),
             ("request_contribution", ["could you help with a small task?", "I have a task I'd like help with"])]),
        "uncertainty": _intent(
            ["I don't know that. I don't want to guess.", "I don't have that information. Please check another source.",
             "I'm not sure. I can't verify that here."],
            ["don't know", "don't have", "not sure", "can't verify"],
            [("live_weather", ["what is the weather right now?", "is it raining outside my home?"]),
             ("unknown_name", ["what is my name?", "do you know my name?"]),
             ("private_password", ["what is my password?", "tell me my secret password"]),
             ("future_result", ["who will win tomorrow?", "what will happen next year?"]),
             ("live_news", ["what is today's breaking news?", "give me the latest news"]),
             ("unseen_room", ["what is in my room?", "what am I holding?"])],
            [("hidden_number", ["which number am I thinking of?", "guess the number in my mind"]),
             ("distant_events", ["what is happening outside my window?", "what can you see at my house?"])],
            [("private_schedule", ["what appointment do I have tomorrow?", "who is on my private calendar?"]),
             ("unprovided_fact", ["what did I eat before this chat?", "where did I put my missing keys?"])],
            forbidden=["definitely", "certainly", "I can see", "your password is"]),
        "clarification": _intent(
            ["Could you give me a little more detail?", "What do you mean? Please tell me more.",
             "I'm not sure what you mean. Could you clarify?"],
            ["detail", "mean", "clarify", "tell me more"],
            [("do_that", ["can you do that?", "please do that"]),
             ("this_one", ["what about this?", "what about that one?"]),
             ("fix_it", ["please fix it", "make it better"]),
             ("tell_me", ["tell me about it", "explain that to me"]),
             ("choose_one", ["which one?", "what should I pick?"]),
             ("unclear_reference", ["is that okay?", "will that work?"])],
            [("ambiguous_comparison", ["is the other thing better?", "how does it compare to the other one?"]),
             ("unspecified_change", ["change the thing I mentioned", "adjust the thing for me"])],
            [("missing_subject", ["why did it happen?", "what caused that?"]),
             ("unspecified_action", ["handle the matter please", "sort out the situation"])],
            forbidden=["I have fixed it", "done", "I changed it"]),
        "preferences": _intent(
            ["I don't have personal tastes, but we can talk about yours.",
             "I'm a program, so I don't have personal favorites."],
            ["don't have", "do not have", "program"],
            [("favorite_food", ["what is your favorite food?", "which food do you like best?"]),
             ("favorite_color", ["what is your favorite color?", "which color do you prefer?"]),
             ("music_taste", ["what music do you like?", "what is your favorite song?"]),
             ("hobbies", ["what are your hobbies?", "what do you do for fun?"]),
             ("personal_wants", ["what do you want?", "do you have personal wishes?"]),
             ("favorite_place", ["where do you like to travel?", "what is your favorite place?"])],
            [("dream_holiday", ["what would your dream holiday be?", "where would you choose to vacation?"]),
             ("pastimes", ["how do you spend your free time?", "what do you enjoy after work?"])],
            [("favorite_book", ["which novel is your personal favorite?", "what book would you pick for pleasure?"]),
             ("personal_taste", ["describe your personal taste", "what appeals to you personally?"])],
            forbidden=["I love eating", "my favorite food is", "I went on holiday", "I play sports"]),
        "encouragement": _intent(
            ["We can take it one small step at a time.", "It's okay to practice slowly. Try one small step.",
             "Let's keep it simple and try again."],
            ["step", "practice", "try", "simple"],
            [("practice_hard", ["this practice is hard", "practicing is difficult"]),
             ("made_mistake", ["I made a mistake", "I got it wrong"]),
             ("stuck", ["I'm stuck on this exercise", "I feel stuck with practice"]),
             ("slow_progress", ["I'm learning slowly", "my progress is slow"]),
             ("try_again", ["should I try again?", "can we try it again?"]),
             ("too_complex", ["this exercise seems too complex", "this is too much at once"])],
            [("discouraged_learner", ["I'm losing confidence with this exercise", "this lesson has discouraged me"]),
             ("small_start", ["I need an easier first step", "can we begin with something easier?"])],
            [("another_attempt", ["my last attempt did not work", "I failed at the practice task again"]),
             ("manageable_practice", ["could we make the practice manageable?", "I'd prefer to tackle a smaller piece"])])
    }
    memory = {
        "train": [("name_introduction", "My name is {name}.", "Nice to meet you.", "What is my name?"),
                  ("call_me", "Please call me {name}.", "All right.", "What should you call me?"),
                  ("called", "I'm called {name}.", "Thanks for telling me.", "Which name did I give you?"),
                  ("name_use", "The name I use is {name}.", "I'll use it in this chat.", "Can you repeat my name?"),
                  ("hi_name", "Hi, I'm {name}.", "Hello.", "Do you remember the name I just said?"),
                  ("nickname", "My nickname is {name}.", "Got it.", "What is my nickname?")],
        "validation": [("address", "Address me as {name}.", "Understood.", "How did I ask to be addressed?"),
                       ("introducing", "Let me introduce myself: {name}.", "Nice to meet you.", "Who just introduced themselves?")],
        "test": [("use_name", "Use {name} when you speak to me.", "Okay.", "What name should you use for me?"),
                 ("signed_message", "For this chat, I will sign my messages as {name}.", "Thanks for explaining.", "How will I sign my messages?")],
    }
    return {"revision": REVISION, "intents": intents, "memory_families": memory,
            "memory_names": ["Mira", "Noah", "Lina", "Theo"]}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _render(turns: list[dict]) -> str:
    return "\n".join(turn["role"] + ": " + turn["content"] for turn in turns)


def _canonical_input(turns: list[dict]) -> str:
    # Case, spacing and punctuation variants of a prompt must share a partition.
    # Role boundaries are retained so a bare name question differs from a
    # question asked after the user has introduced their name.
    return _digest([(turn["role"], "".join(re.findall(r"[a-z0-9]+", turn["content"].casefold()))) for turn in turns])


def _role_tokens(turns: list[dict]) -> int:
    return sum(2 + len(turn["content"].encode("utf-8")) for turn in turns)


def _training_variants(prompt: str, seed: int, family: str) -> list[tuple[str, str]]:
    values = [("original", prompt), ("case", prompt[:1].upper() + prompt[1:]),
              ("punctuation", prompt.rstrip(".!?") if prompt.endswith((".", "!", "?")) else prompt + "!")]
    words = list(re.finditer(r"[a-zA-Z]{4,}", prompt))
    if words:
        rng = random.Random(f"{seed}:{family}:{prompt}")
        word = rng.choice(words)
        positions = [index for index in range(word.start() + 1, word.end() - 1)
                     if prompt[index] != prompt[index + 1]]
        if positions:
            index = rng.choice(positions)
            typo = prompt[:index] + prompt[index + 1] + prompt[index] + prompt[index + 2:]
            values.append(("single_transposition", typo))
    if len(prompt) <= 24 and 1 <= prompt.count(" ") <= 4:
        values.append(("spacing", prompt.replace(" ", "")))
    seen = set()
    unique = []
    for kind, text in values:
        if text not in seen:
            unique.append((kind, text))
            seen.add(text)
    return unique


def _case(intent, family, turns, reference, rubric, *, variant="original", base_prompt=None,
          context_pair_id=None, memory_value=None, forbidden_names=None, interaction_kind="single_turn",
          context_families=None):
    identifier = _digest({"intent": intent, "family": family, "turns": turns})[:24]
    record = {"id": identifier, "intent": intent, "family_id": family, "turns": turns,
              "prompt": turns[0]["content"] if len(turns) == 1 else None,
              "reference": reference, "rubric": json.loads(json.dumps(rubric)),
              "variant": variant, "base_prompt": base_prompt,
              "context_pair_id": context_pair_id, "memory_value": memory_value,
              "interaction_kind": interaction_kind, "context_families": list(context_families or [family])}
    if forbidden_names:
        record["rubric"]["forbidden"].extend(forbidden_names)
    return record


def _build_cases(spec: dict, seed: int) -> tuple[dict, dict]:
    cases = {split: [] for split in SPLITS}
    groups = {split: [] for split in SPLITS}
    for intent, entry in spec["intents"].items():
        for split in SPLITS:
            for name, prompts in entry["families"][split]:
                family = intent + ":" + name
                groups[split].append(family)
                reference = random.Random(f"{seed}:{family}:reply").choice(entry["replies"])
                for prompt in prompts:
                    variants = _training_variants(prompt, seed, family) if split == "train" else [("original", prompt)]
                    for kind, text in variants:
                        cases[split].append(_case(intent, family, [{"role": "user", "content": text}], reference,
                                                  entry["rubric"], variant=kind, base_prompt=prompt))
    # Actual short conversations, rather than wrapping a previous conversation
    # inside one user message. Every source family remains in its original split.
    for split in SPLITS:
        greeting_family, greeting_prompts = spec["intents"]["greeting"]["families"][split][0]
        for intent in ("identity", "help"):
            entry = spec["intents"][intent]
            for name, prompts in entry["families"][split]:
                family = intent + ":" + name
                reference = random.Random(f"{seed}:{family}:reply").choice(entry["replies"])
                turns = [{"role": "user", "content": greeting_prompts[0]},
                         {"role": "assistant", "content": "Hello."},
                         {"role": "user", "content": prompts[0]}]
                cases[split].append(_case(intent, family, turns, reference, entry["rubric"],
                                          base_prompt=prompts[0], interaction_kind="greeting_followup",
                                          context_families=["greeting:" + greeting_family, family]))
    names = spec["memory_names"]
    for split in SPLITS:
        for name, introduction, acknowledgement, question in spec["memory_families"][split]:
            family = "name_recall:" + name
            groups[split].append(family)
            for index, person in enumerate(names):
                turns = [{"role": "user", "content": introduction.format(name=person)},
                         {"role": "assistant", "content": acknowledgement},
                         {"role": "user", "content": question}]
                rubric = {"required_any": [], "required_all": [person], "forbidden": [], "max_response_bytes": 80}
                cases[split].append(_case("name_recall", family, turns, f"Your name is {person}.", rubric,
                                          base_prompt=question, context_pair_id=family + f":pair-{index // 2}",
                                          memory_value=person, forbidden_names=[other for other in names if other != person],
                                          interaction_kind="name_recall"))
    # Exact duplicate inputs inside one family add no new example. Cross-family
    # ambiguity is checked separately instead of silently reassigning examples.
    for split in SPLITS:
        unique = {record["id"]: record for record in cases[split]}
        cases[split] = [unique[key] for key in sorted(unique)]
        groups[split] = sorted(groups[split])
    return cases, groups


def _conversation(case: dict) -> dict:
    return {"id": case["id"], "tree_id": "chat-" + case["id"], "case_id": case["id"],
            "intent": case["intent"], "family_id": case["family_id"], "source": "original-golem-chat",
            "turns": case["turns"] + [{"role": "assistant", "content": case["reference"]}]}


def _validate_partitioning(cases: dict, groups: dict) -> None:
    family_owner, input_owner, ids = {}, {}, set()
    for split in SPLITS:
        for family in groups[split]:
            if family in family_owner:
                raise ValueError("Paraphrase family occurs in multiple partitions")
            family_owner[family] = split
        for case in cases[split]:
            if case["id"] in ids:
                raise ValueError("Duplicate chat case ID")
            ids.add(case["id"])
            if family_owner.get(case["family_id"]) != split:
                raise ValueError("Chat case is assigned outside its family partition")
            if any(family_owner.get(family) != split for family in case["context_families"]):
                raise ValueError("Conversation context draws from another paraphrase partition")
            canonical = _canonical_input(case["turns"])
            if canonical in input_owner and input_owner[canonical] != (split, case["family_id"]):
                raise ValueError(f"Equivalent chat prompts cross family boundaries: {case['family_id']}")
            input_owner[canonical] = split, case["family_id"]
            if case["turns"][-1]["role"] != "user":
                raise ValueError("Evaluation case must end at a user prompt")
            if split != "train" and case["variant"] != "original":
                raise ValueError("Held-out paraphrases must not receive training augmentation")
            count = _role_tokens(_conversation(case)["turns"])
            limit = 160 if len(case["turns"]) > 1 else 128
            if count > limit:
                raise ValueError(f"Chat example exceeds context limit ({count}>{limit}): {case['family_id']}")


def _content_hash(dataset: dict) -> str:
    return _digest({key: value for key, value in dataset.items() if key != "manifest"})


def prepare_chat_corpus(output_dir: str | Path, seed: int = 61) -> Path:
    """Prepare original examples locally; perform no network requests."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    output = Path(output_dir)
    path = output / "dataset.json"
    parameters = {"seed": seed, "revision": REVISION}
    if path.exists():
        saved = load_chat_corpus(path)
        if saved["manifest"]["parameters"] != parameters:
            raise ValueError("Existing chat corpus uses different settings; choose a new output directory")
        return path
    if output.exists() and any(output.iterdir()):
        raise ValueError("Chat corpus output directory must be empty")
    spec = _original_spec()
    cases, groups = _build_cases(spec, seed)
    _validate_partitioning(cases, groups)
    conversations = {split: [_conversation(case) for case in cases[split]] for split in SPLITS}
    texts = {split: [_render(record["turns"]) for record in conversations[split]] for split in SPLITS}
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "source_snapshot" / "original_examples.json", spec)
    shutil.copyfile(LICENSE_PATH, output / "CHAT-DATA-LICENSE.txt")
    manifest = {"parameters": parameters, "source": "Original examples authored for Project Golem",
                "license": {"name": "MIT", "path": "CHAT-DATA-LICENSE.txt", "sha256": _sha256(output / "CHAT-DATA-LICENSE.txt")},
                "source_snapshot": {"path": "source_snapshot/original_examples.json",
                                    "sha256": _sha256(output / "source_snapshot" / "original_examples.json")},
                "external_downloads": False,
                "split_policy": "Manually assigned paraphrase families before augmentation; punctuation/case/spacing-equivalent full inputs cannot cross family or split boundaries.",
                "augmentation_policy": "Training only: original, initial case, punctuation, one interior adjacent-letter transposition, and bounded collapsed spacing. No validation/test example is used to construct variants.",
                "inference_policy": "Examples supervise neural training. No response lookup, output substitution, hosted model, or retrieval is supplied at inference.",
                "rubric_policy": "Transparent intent keywords and forbidden phrases are approximate diagnostics, not a complete relevance or quality measure. Preserve the actual generated text for review.",
                "context_policy": "Complete conversations must fit training windows; max 128 role tokens for single-turn cases and 160 for name recall or greeting-followup cases; use sequence_length >=160 (192 recommended).",
                "normalization": "Original UTF-8 text preserved; no lowercasing during rendering or inference.",
                "limitations": ["Narrow original chat curriculum, not evidence of broad factual knowledge or general conversation.",
                                "Held-out families share intents and acceptable answer vocabulary with training; the test is paraphrase generalization, not learning unseen intents.",
                                "Names are deliberately balanced and shared across splits to isolate context use from learning an unseen spelling.",
                                "A warm-started model may already recognize these phrases; compare the frozen initial checkpoint on the same held-out cases."],
                "curriculum": {"intents": sorted(set(record["intent"] for record in cases["train"])),
                               "scope": "Training conversations only; every conversation declares its intent for explicit sampling.",
                               "default_sampling": "No hidden weights or repeated documents are added by the corpus builder."},
                "splits": {split: {"cases": len(cases[split]), "families": len(groups[split]),
                                   "max_role_encoded_tokens": max(map(lambda row: _role_tokens(row["turns"]), conversations[split])),
                                   "intent_counts": dict(sorted(Counter(record["intent"] for record in cases[split]).items())),
                                   "interaction_counts": dict(sorted(Counter(record["interaction_kind"] for record in cases[split]).items())),
                                   "context_pairs": len({record["context_pair_id"] for record in cases[split] if record["context_pair_id"]})}
                           for split in SPLITS}}
    dataset = {"schema": SCHEMA, "domain": "dialogue_utf8", "texts": texts, "conversations": conversations,
               "chat_cases": cases, "groups": groups, "dictionary": [], "manifest": manifest}
    manifest["content_sha256"] = _content_hash(dataset)
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(path, dataset)
    load_chat_corpus(path)
    return path


def load_chat_corpus(path: str | Path) -> dict:
    """Verify original source provenance, family splits and all rendered examples."""
    path = Path(path)
    if path.is_dir():
        path = path / "dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("Unsupported original chat corpus schema")
    manifest = data.get("manifest", {})
    if manifest.get("content_sha256") != _content_hash(data):
        raise ValueError("Chat corpus content checksum mismatch")
    if json.loads((path.parent / "manifest.json").read_text(encoding="utf-8")) != manifest:
        raise ValueError("Chat corpus manifest mismatch")
    snapshot = manifest.get("source_snapshot", {})
    if snapshot.get("path") != "source_snapshot/original_examples.json":
        raise ValueError("Unexpected original chat snapshot path")
    if _sha256(path.parent / snapshot["path"]) != snapshot.get("sha256"):
        raise ValueError("Chat source snapshot checksum mismatch")
    license_info = manifest.get("license", {})
    if license_info.get("path") != "CHAT-DATA-LICENSE.txt" or _sha256(path.parent / "CHAT-DATA-LICENSE.txt") != license_info.get("sha256"):
        raise ValueError("Chat license checksum mismatch")
    spec = json.loads((path.parent / snapshot["path"]).read_text(encoding="utf-8"))
    cases, groups = _build_cases(spec, manifest["parameters"]["seed"])
    if spec.get("revision") != manifest["parameters"]["revision"]:
        raise ValueError("Chat source revision mismatch")
    if data.get("chat_cases") != cases or data.get("groups") != groups:
        raise ValueError("Chat cases do not match their original source families")
    _validate_partitioning(cases, groups)
    for split in SPLITS:
        expected = [_conversation(case) for case in cases[split]]
        if data.get("conversations", {}).get(split) != expected:
            raise ValueError("Chat conversations differ from their declared cases")
        if data.get("texts", {}).get(split) != [_render(record["turns"]) for record in expected]:
            raise ValueError("Chat training text differs from its original conversations")
    return data
