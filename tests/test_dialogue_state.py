import unittest

from english_world.dialogue_state import extract_user_facts, render_verified_facts


class DialogueStateTests(unittest.TestCase):
    def test_assistant_hallucination_is_never_user_memory(self):
        turns = [{'role': 'user', 'content': 'greet me'},
                 {'role': 'assistant', 'content': 'Your name is Noah.'},
                 {'role': 'user', 'content': 'it is?'},
                 {'role': 'assistant', 'content': 'My name is Noah.'},
                 {'role': 'user', 'content': 'ok. i am lonely'}]
        self.assertIsNone(extract_user_facts(turns)['user_name'])
        self.assertEqual(render_verified_facts(turns), '[verified user name: UNKNOWN]\n')

    def test_explicit_user_correction_overwrites_name_and_keeps_source(self):
        turns = [{'role': 'user', 'content': 'My name is Ava.'},
                 {'role': 'assistant', 'content': 'Your name is Noah.'},
                 {'role': 'user', 'content': 'No, my name is Zoë.'}]
        self.assertEqual(extract_user_facts(turns), {'user_name': 'Zoë', 'source_user_turn': 2})

    def test_quotes_questions_hypotheticals_and_negations_do_not_set_names(self):
        for text in ['You said my name is Noah', 'If my name is Noah', '"My name is Noah"',
                     'My name is not Noah', 'My name is Noah?', 'I am lonely', 'Your name is Golem']:
            self.assertIsNone(extract_user_facts([{'role': 'user', 'content': text}])['user_name'], text)

    def test_supported_introduction_forms_and_explicit_forgetting(self):
        for text in ['Call me Alex', 'I am called Alex', "I'm called Alex", 'I go by Alex', 'Actually, my name is Alex.']:
            turns = [{'role': 'user', 'content': text}]
            self.assertEqual(extract_user_facts(turns)['user_name'], 'Alex')
            turns.append({'role': 'user', 'content': 'Please forget my name.'})
            self.assertIsNone(extract_user_facts(turns)['user_name'])

    def test_independent_worlds_do_not_share_facts(self):
        self.assertEqual(extract_user_facts([{'role': 'user', 'content': 'My name is Ava'}])['user_name'], 'Ava')
        self.assertIsNone(extract_user_facts([{'role': 'user', 'content': 'Who are you?'}])['user_name'])


if __name__ == '__main__':
    unittest.main()
