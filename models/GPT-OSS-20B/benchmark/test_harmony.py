import unittest
from harmony import count_channels


class HarmonyCountTests(unittest.TestCase):
    def test_split_headers_and_reasoning(self):
        chunks = [
            (0.1, [200005]),
            (0.2, [35644, 200008, 12, 13]),
            (0.3, [200007, 200006, 173781, 200005, 17196]),
            (0.4, [200008, 14, 15, 200002]),
        ]
        r = count_channels(chunks)
        self.assertEqual(r["reasoning_tokens"], 2)
        self.assertEqual(r["final_answer_tokens"], 2)
        self.assertEqual(r["first_visible_answer_s"], 0.4)
        self.assertEqual(
            sum(
                r[k]
                for k in [
                    "reasoning_tokens",
                    "final_answer_tokens",
                    "other_body_tokens",
                    "format_tokens",
                ]
            ),
            sum(len(ids) for _, ids in chunks),
        )

    def test_malformed_final_code_channel(self):
        r = count_channels([(1, [200005, 17196, 3490, 200008, 42, 200002])])
        self.assertIsNone(r["first_visible_answer_s"])
        self.assertEqual(r["final_answer_tokens"], 0)
        self.assertEqual(r["other_body_tokens"], 1)

    def test_ignore_eos_continuation_is_not_useful_answer(self):
        r = count_channels(
            [
                (
                    1,
                    [
                        200005,
                        17196,
                        200008,
                        42,
                        200002,
                        200006,
                        200005,
                        17196,
                        200008,
                        99,
                    ],
                )
            ]
        )
        self.assertEqual(r["final_answer_tokens"], 1)
        self.assertEqual(r["post_stop_tokens"], 5)

    def test_valid_format_metadata(self):
        r = count_channels([(1, [200005, 17196, 200003, 3490, 200008, 42, 200002])])
        self.assertEqual(r["final_answer_tokens"], 1)

    def test_no_final_answer(self):
        r = count_channels([(1, [200005, 35644, 200008, 42, 200002])])
        self.assertIsNone(r["first_visible_answer_s"])
        self.assertEqual(r["final_answer_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
