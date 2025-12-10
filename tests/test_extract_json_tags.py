import unittest
from youtube_seo_automation import extract_json_tags


class TestExtractJsonTags(unittest.TestCase):
    def test_valid_json_with_markdown_fences(self):
        raw = """Some explanation
```json
["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8","tag9","tag10"]
```
Thank you
"""
        tags = extract_json_tags(raw)
        self.assertIsInstance(tags, list)
        self.assertEqual(len(tags), 10)
        self.assertEqual(tags[0], "tag1")

    def test_invalid_count_raises_value_error(self):
        raw = '["one","two","three"]'
        with self.assertRaises(ValueError):
            extract_json_tags(raw)

    def test_no_json_raises_value_error(self):
        raw = "No json here, just plaintext and markdown ```text```"
        with self.assertRaises(ValueError):
            extract_json_tags(raw)


if __name__ == "__main__":
    unittest.main()
