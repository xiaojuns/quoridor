from __future__ import annotations

import unittest

from quoridor.phase4f46_result_verify import SCHEMA_VERSION


class Phase4F46ResultVerifyTests(unittest.TestCase):
    def test_schema_version_is_frozen(self) -> None:
        self.assertEqual(SCHEMA_VERSION, 1)


if __name__ == "__main__":
    unittest.main()
