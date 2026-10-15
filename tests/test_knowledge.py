import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import knowledge


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self):
        return self.body

    def close(self):
        self.closed = True


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        knowledge.clear_cache()

    def tearDown(self):
        knowledge.clear_cache()

    def test_known_food_offline_is_local_and_explicitly_illustrative(self):
        result = knowledge.build_insights("chicken-wings", online=False)

        self.assertEqual(result["classId"], "chicken_wings")
        self.assertEqual(result["label"], "Chicken Wings")
        self.assertEqual(result["overview"]["status"], "local_illustrative")
        self.assertIn("American", result["overview"]["text"])
        self.assertEqual(result["recipe"]["status"], "local_illustrative")
        self.assertTrue(result["recipe"]["ingredients"])
        self.assertIn("not guaranteed", result["recipe"]["note"])
        self.assertEqual(result["nutrition"]["status"], "illustrative")
        self.assertEqual(result["nutrition"]["values"]["calories"], 290)
        self.assertIn("actual nutrition varies", result["nutrition"]["basis"])
        self.assertIn("unverified", result["nutrition"]["basis"].lower())
        self.assertEqual(result["retrieval"]["mode"], "offline")
        self.assertEqual(result["retrieval"]["status"], "local")
        self.assertEqual(result["restaurants"]["status"], "search_only")

    def test_unknown_label_offline_is_supported_without_fabrication(self):
        result = knowledge.build_insights("dragon-fruit-special", online=False)

        self.assertEqual(result["classId"], "dragon_fruit_special")
        self.assertEqual(result["label"], "Dragon Fruit Special")
        self.assertEqual(result["overview"]["status"], "unavailable")
        self.assertEqual(result["recipe"]["status"], "unavailable")
        self.assertEqual(result["recipe"]["ingredients"], [])
        self.assertEqual(result["recipe"]["steps"], [])
        self.assertEqual(result["nutrition"]["status"], "unavailable")
        self.assertNotIn("values", result["nutrition"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieval"]["status"], "local")

    @patch("backend.knowledge.urllib.request.urlopen")
    def test_external_failures_are_graceful_and_timeout_is_strict(self, urlopen):
        urlopen.side_effect = TimeoutError("simulated timeout")

        result = knowledge.build_insights("unlisted dish", online=True)

        self.assertEqual(result["overview"]["status"], "unavailable")
        self.assertEqual(result["recipe"]["status"], "unavailable")
        self.assertEqual(result["nutrition"]["status"], "unavailable")
        self.assertEqual(result["retrieval"]["status"], "unavailable")
        self.assertEqual(len(result["sources"]), 2)
        self.assertTrue(all(source["status"] == "unavailable" for source in result["sources"]))
        self.assertEqual(urlopen.call_count, 2)
        for call in urlopen.call_args_list:
            self.assertEqual(call.kwargs["timeout"], 4.0)
            self.assertTrue(call.args[0].full_url.startswith("https://"))

    @patch("backend.knowledge.urllib.request.urlopen")
    def test_online_recipe_is_sanitized_and_uses_safe_image_source(self, urlopen):
        def response_for(request, timeout):
            self.assertEqual(timeout, 4.0)
            if "wikipedia.org" in request.full_url:
                return FakeResponse({"extract": "A <b>short</b> overview."})
            return FakeResponse(
                {
                    "meals": [
                        {
                            "strMeal": " <b>Suggested Meal</b> ",
                            "strIngredient1": "<script>alert(1)</script>Rice",
                            "strMeasure1": "1 cup",
                            "strIngredient2": "<img src=x onerror=alert(2)>Herbs",
                            "strInstructions": "<p>Cook &amp; serve.</p>",
                            "strMealThumb": "javascript:alert(3)",
                            "strSource": "javascript:alert(4)",
                        }
                    ]
                }
            )

        urlopen.side_effect = response_for
        result = knowledge.build_insights("some-label", online=True)

        self.assertEqual(result["overview"]["status"], "online")
        self.assertEqual(result["recipe"]["status"], "online")
        self.assertEqual(result["recipe"]["title"], "Suggested Meal")
        self.assertEqual(result["recipe"]["ingredients"], ["1 cup Rice", "Herbs"])
        self.assertEqual(result["recipe"]["steps"], ["Cook & serve."])
        self.assertNotIn("imageUrl", result["recipe"])
        self.assertTrue(result["recipe"]["sourceUrl"].startswith("https://www.themealdb.com/"))
        self.assertNotIn("<", json.dumps(result))
        self.assertNotIn("javascript:", json.dumps(result))
        self.assertEqual(result["retrieval"]["status"], "online")

    @patch("backend.knowledge.urllib.request.urlopen")
    def test_online_wikipedia_and_mealdb_sources_are_returned(self, urlopen):
        def response_for(request, timeout):
            if "wikipedia.org" in request.full_url:
                return FakeResponse(
                    {
                        "extract": "A public summary.",
                        "content_urls": {
                            "desktop": {"page": "https://en.wikipedia.org/wiki/Example"}
                        },
                    }
                )
            return FakeResponse(
                {
                    "meals": [
                        {
                            "strMeal": "Recipe Suggestion",
                            "strIngredient1": "Potato",
                            "strMeasure1": "2",
                            "strInstructions": "Boil the potato.",
                            "strMealThumb": "https://www.themealdb.com/images/media/meals/example.jpg",
                        }
                    ]
                }
            )

        urlopen.side_effect = response_for
        result = knowledge.build_insights("new-food", online=True)

        self.assertEqual(result["overview"]["sourceUrl"], "https://en.wikipedia.org/wiki/Example")
        self.assertEqual(result["recipe"]["imageUrl"], "https://www.themealdb.com/images/media/meals/example.jpg")
        self.assertIn("not guaranteed", result["recipe"]["sourceLabel"])
        self.assertEqual(result["retrieval"]["status"], "online")
        self.assertEqual(len(result["sources"]), 2)
        self.assertTrue(all(source["status"] == "online" for source in result["sources"]))

    def test_invalid_labels_fail_before_any_lookup(self):
        with self.assertRaises(ValueError):
            knowledge.build_insights("---", online=False)
        with self.assertRaises(ValueError):
            knowledge.build_insights("   ", online=False)
        with self.assertRaises(TypeError):
            knowledge.build_insights(None, online=False)


if __name__ == "__main__":
    unittest.main()
