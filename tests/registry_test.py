# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the provider registry module.

Note: This file tests the deprecated registry module which is now an alias
for router. The no-name-in-module warning for providers.registry is expected.
Test helper classes also intentionally have few public methods.
"""
# pylint: disable=no-name-in-module

import re
<<<<<<< HEAD
from unittest import mock
=======
>>>>>>> origin

from absl.testing import absltest

from langextract import exceptions
<<<<<<< HEAD
from langextract import inference
from langextract.providers import registry


class FakeProvider(inference.BaseLanguageModel):
  """Fake provider for testing."""

  def infer(self, batch_prompts, **kwargs):
    return [[inference.ScoredOutput(score=1.0, output="test")]]
=======
from langextract.core import base_model
from langextract.core import types
from langextract.providers import router


class FakeProvider(base_model.BaseLanguageModel):
  """Fake provider for testing."""

  def infer(self, batch_prompts, **kwargs):
    return [[types.ScoredOutput(score=1.0, output="test")]]
>>>>>>> origin

  def infer_batch(self, prompts, batch_size=32):
    return self.infer(prompts)


<<<<<<< HEAD
class AnotherFakeProvider(inference.BaseLanguageModel):
  """Another fake provider for testing."""

  def infer(self, batch_prompts, **kwargs):
    return [[inference.ScoredOutput(score=1.0, output="another")]]
=======
class AnotherFakeProvider(base_model.BaseLanguageModel):
  """Another fake provider for testing."""

  def infer(self, batch_prompts, **kwargs):
    return [[types.ScoredOutput(score=1.0, output="another")]]
>>>>>>> origin

  def infer_batch(self, prompts, batch_size=32):
    return self.infer(prompts)


class RegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
<<<<<<< HEAD
    registry.clear()

  def tearDown(self):
    super().tearDown()
    registry.clear()
=======
    router.clear()

  def tearDown(self):
    super().tearDown()
    router.clear()
>>>>>>> origin

  def test_register_decorator(self):
    """Test registering a provider using the decorator."""

<<<<<<< HEAD
    @registry.register(r"^test-model")
    class TestProvider(FakeProvider):
      pass

    resolved = registry.resolve("test-model-v1")
=======
    @router.register(r"^test-model")
    class TestProvider(FakeProvider):
      pass

    resolved = router.resolve("test-model-v1")
>>>>>>> origin
    self.assertEqual(resolved, TestProvider)

  def test_register_lazy(self):
    """Test lazy registration with string target."""
<<<<<<< HEAD
    registry.register_lazy(r"^fake-model", target="registry_test:FakeProvider")

    resolved = registry.resolve("fake-model-v2")
=======
    # Use direct registration for test provider to avoid module path issues
    router.register(r"^fake-model")(FakeProvider)

    resolved = router.resolve("fake-model-v2")
>>>>>>> origin
    self.assertEqual(resolved, FakeProvider)

  def test_multiple_patterns(self):
    """Test registering multiple patterns for one provider."""
<<<<<<< HEAD
    registry.register_lazy(
        r"^gemini", r"^palm", target="registry_test:FakeProvider"
    )

    self.assertEqual(registry.resolve("gemini-pro"), FakeProvider)
    self.assertEqual(registry.resolve("palm-2"), FakeProvider)

  def test_priority_resolution(self):
    """Test that higher priority wins on conflicts."""
    registry.register_lazy(
        r"^model", target="registry_test:FakeProvider", priority=0
    )
    registry.register_lazy(
        r"^model", target="registry_test:AnotherFakeProvider", priority=10
    )

    resolved = registry.resolve("model-v1")
=======
    # Use direct registration to avoid module path issues in Bazel
    router.register(r"^gemini", r"^palm")(FakeProvider)

    self.assertEqual(router.resolve("gemini-pro"), FakeProvider)
    self.assertEqual(router.resolve("palm-2"), FakeProvider)

  def test_priority_resolution(self):
    """Test that higher priority wins on conflicts."""
    # Use direct registration to avoid module path issues in Bazel
    router.register(r"^model", priority=0)(FakeProvider)
    router.register(r"^model", priority=10)(AnotherFakeProvider)

    resolved = router.resolve("model-v1")
>>>>>>> origin
    self.assertEqual(resolved, AnotherFakeProvider)

  def test_no_provider_registered(self):
    """Test error when no provider matches."""
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError,
        "No provider registered for model_id='unknown-model'",
    ):
<<<<<<< HEAD
      registry.resolve("unknown-model")

  def test_caching(self):
    """Test that resolve results are cached."""
    registry.register_lazy(r"^cached", target="registry_test:FakeProvider")

    # First call
    result1 = registry.resolve("cached-model")
    # Second call should return cached result
    result2 = registry.resolve("cached-model")
=======
      router.resolve("unknown-model")

  def test_caching(self):
    """Test that resolve results are cached."""
    # Use direct registration for test provider to avoid module path issues
    router.register(r"^cached")(FakeProvider)

    # First call
    result1 = router.resolve("cached-model")
    # Second call should return cached result
    result2 = router.resolve("cached-model")
>>>>>>> origin

    self.assertIs(result1, result2)

  def test_clear_registry(self):
<<<<<<< HEAD
    """Test clearing the registry."""
    registry.register_lazy(r"^temp", target="registry_test:FakeProvider")

    # Should resolve before clear
    resolved = registry.resolve("temp-model")
    self.assertEqual(resolved, FakeProvider)

    # Clear registry
    registry.clear()

    # Should fail after clear
    with self.assertRaises(exceptions.InferenceConfigError):
      registry.resolve("temp-model")

  def test_list_entries(self):
    """Test listing registered entries."""
    registry.register_lazy(r"^test1", target="fake:Target1", priority=5)
    registry.register_lazy(
        r"^test2", r"^test3", target="fake:Target2", priority=10
    )

    entries = registry.list_entries()
=======
    """Test clearing the router."""
    # Use direct registration for test provider to avoid module path issues
    router.register(r"^temp")(FakeProvider)

    # Should resolve before clear
    resolved = router.resolve("temp-model")
    self.assertEqual(resolved, FakeProvider)

    # Clear registry
    router.clear()

    # Should fail after clear
    with self.assertRaises(exceptions.InferenceConfigError):
      router.resolve("temp-model")

  def test_list_entries(self):
    """Test listing registered entries."""
    router.register_lazy(r"^test1", target="fake:Target1", priority=5)
    router.register_lazy(
        r"^test2", r"^test3", target="fake:Target2", priority=10
    )

    entries = router.list_entries()
>>>>>>> origin
    self.assertEqual(len(entries), 2)

    patterns1, priority1 = entries[0]
    self.assertEqual(patterns1, ["^test1"])
    self.assertEqual(priority1, 5)

    patterns2, priority2 = entries[1]
    self.assertEqual(set(patterns2), {"^test2", "^test3"})
    self.assertEqual(priority2, 10)

  def test_lazy_loading_defers_import(self):
    """Test that lazy registration doesn't import until resolve."""
    # Register with a module that would fail if imported
<<<<<<< HEAD
    registry.register_lazy(r"^lazy", target="non.existent.module:Provider")

    # Registration should succeed without importing
    entries = registry.list_entries()
=======
    router.register_lazy(r"^lazy", target="non.existent.module:Provider")

    # Registration should succeed without importing
    entries = router.list_entries()
>>>>>>> origin
    self.assertTrue(any("^lazy" in patterns for patterns, _ in entries))

    # Only on resolve should it try to import and fail
    with self.assertRaises(ModuleNotFoundError):
<<<<<<< HEAD
      registry.resolve("lazy-model")
=======
      router.resolve("lazy-model")
>>>>>>> origin

  def test_regex_pattern_objects(self):
    """Test using pre-compiled regex patterns."""
    pattern = re.compile(r"^custom-\d+")

<<<<<<< HEAD
    @registry.register(pattern)
    class CustomProvider(FakeProvider):
      pass

    self.assertEqual(registry.resolve("custom-123"), CustomProvider)

    # Should not match without digits
    with self.assertRaises(exceptions.InferenceConfigError):
      registry.resolve("custom-abc")
=======
    @router.register(pattern)
    class CustomProvider(FakeProvider):
      pass

    self.assertEqual(router.resolve("custom-123"), CustomProvider)

    # Should not match without digits
    with self.assertRaises(exceptions.InferenceConfigError):
      router.resolve("custom-abc")
>>>>>>> origin

  def test_resolve_provider_by_name(self):
    """Test resolving provider by exact name."""

<<<<<<< HEAD
    @registry.register(r"^test-model", r"^TestProvider$")
=======
    @router.register(r"^test-model", r"^TestProvider$")
>>>>>>> origin
    class TestProvider(FakeProvider):
      pass

    # Resolve by exact class name pattern
<<<<<<< HEAD
    provider = registry.resolve_provider("TestProvider")
    self.assertEqual(provider, TestProvider)

    # Resolve by partial name match
    provider = registry.resolve_provider("test")
=======
    provider = router.resolve_provider("TestProvider")
    self.assertEqual(provider, TestProvider)

    # Resolve by partial name match
    provider = router.resolve_provider("test")
>>>>>>> origin
    self.assertEqual(provider, TestProvider)

  def test_resolve_provider_not_found(self):
    """Test resolve_provider raises for unknown provider."""
    with self.assertRaises(exceptions.InferenceConfigError) as cm:
<<<<<<< HEAD
      registry.resolve_provider("UnknownProvider")
=======
      router.resolve_provider("UnknownProvider")
>>>>>>> origin
    self.assertIn("No provider found matching", str(cm.exception))

  def test_hf_style_model_id_patterns(self):
    """Test that Hugging Face style model ID patterns work.

    This addresses issue #129 where HF-style model IDs like
    'meta-llama/Llama-3.2-1B-Instruct' weren't being recognized.
    """

<<<<<<< HEAD
    @registry.register(
=======
    @router.register(
>>>>>>> origin
        r"^meta-llama/[Ll]lama",
        r"^google/gemma",
        r"^mistralai/[Mm]istral",
        r"^microsoft/phi",
        r"^Qwen/",
        r"^TinyLlama/",
        priority=100,
    )
<<<<<<< HEAD
    class TestHFProvider(inference.BaseLanguageModel):  # pylint: disable=too-few-public-methods
=======
    class TestHFProvider(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods
>>>>>>> origin

      def infer(self, batch_prompts, **kwargs):
        return []

    hf_model_ids = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/llama-2-7b",
        "google/gemma-2b",
        "mistralai/Mistral-7B-v0.1",
        "microsoft/phi-3-mini",
        "Qwen/Qwen2.5-7B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]

    for model_id in hf_model_ids:
      with self.subTest(model_id=model_id):
<<<<<<< HEAD
        provider_class = registry.resolve(model_id)
=======
        provider_class = router.resolve(model_id)
>>>>>>> origin
        self.assertEqual(provider_class, TestHFProvider)


if __name__ == "__main__":
  absltest.main()
