import unittest

import workflow_alert as alert


class ErrorExcerptTests(unittest.TestCase):

    def test_echoed_run_script_is_not_an_executed_error(self):
        raw = (
            '2026-11-10T04:29:59Z ##[group]Run if [ -z "$OPENCODE_API_KEY" ]; then\n'
            '2026-11-10T04:29:59Z \x1b[36;1m  echo "::error::OPENCODE_API_KEY secret 이 비어 있습니다."\x1b[0m\n'
            '2026-11-10T04:29:59Z ##[endgroup]\n'
            '2026-11-10T04:29:59Z Imported 1 credential\n'
            '2026-11-10T04:35:07Z ##[error]Process completed with exit code 1.\n'
        ).encode()
        excerpt = alert.error_excerpt(raw)
        self.assertNotIn('OPENCODE_API_KEY', excerpt)
        self.assertIn('Process completed with exit code 1.', excerpt)

    def test_actual_runtime_error_and_diagnostic_groups_are_preserved(self):
        raw = (
            '2026-11-10T04:29:59Z ##[group]Run check-key\n'
            '2026-11-10T04:29:59Z echo "::error::missing key"\n'
            '2026-11-10T04:29:59Z ##[endgroup]\n'
            '2026-11-10T04:29:59Z ##[error]missing key\n'
            '2026-11-10T04:30:00Z ##[group]Diagnostics\n'
            '2026-11-10T04:30:00Z Error: provider unavailable\n'
            '2026-11-10T04:30:00Z ##[endgroup]\n'
        ).encode()
        excerpt = alert.error_excerpt(raw)
        self.assertNotIn('echo', excerpt)
        self.assertIn('##[error]missing key', excerpt)
        self.assertIn('Error: provider unavailable', excerpt)
