"""Regression tests for independent CI heartbeat and authentication checks."""

import unittest
from datetime import UTC, datetime, timedelta

import external_watchdog as watchdog


class WatchdogTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 17, tzinfo=UTC)
        self.run = {
            "id": 1,
            "status": "completed",
            "conclusion": "success",
            "display_title": "NAS heartbeat " + self.now.isoformat(),
            "head_branch": "main",
            "event": "workflow_dispatch",
            "created_at": (self.now - timedelta(minutes=5)).isoformat(),
        }
        self.job = {
            "name": "heartbeat",
            "status": "completed",
            "conclusion": "success",
            "completed_at": self.now.isoformat(),
            "labels": ["self-hosted", "local-ryujaehun.github.io"],
        }

    def test_fresh_successful_job_is_liveness_evidence(self):
        self.assertTrue(watchdog.heartbeat_valid(self.run, [self.job], self.now))

    def test_stale_queued_run_cannot_become_fresh_when_completed(self):
        self.run["created_at"] = (self.now - timedelta(minutes=31)).isoformat()
        self.assertFalse(watchdog.heartbeat_valid(self.run, [self.job], self.now))

    def test_future_clock_and_unavailable_or_skipped_job_fail_closed(self):
        self.run["created_at"] = (self.now + timedelta(minutes=2)).isoformat()
        self.assertFalse(watchdog.heartbeat_valid(self.run, [self.job], self.now))
        self.run["created_at"] = self.now.isoformat()
        self.assertFalse(watchdog.heartbeat_valid(self.run, [], self.now))
        self.job["conclusion"] = "skipped"
        self.assertFalse(watchdog.heartbeat_valid(self.run, [self.job], self.now))

    def test_authentication_boundary_rejects_data_and_foreign_redirect(self):
        self.assertFalse(watchdog.protected(200, ""))
        self.assertFalse(watchdog.protected(302, "https://evil.test/outpost.goauthentik.io/start"))
        self.assertTrue(watchdog.protected(302, "/outpost.goauthentik.io/start?rd=private"))
        self.assertTrue(watchdog.protected(403, ""))

    def test_stale_monitor_observation_is_rejected(self):
        self.run["display_title"] = "NAS heartbeat " + (self.now - timedelta(hours=1)).isoformat()
        self.assertFalse(watchdog.heartbeat_valid(self.run, [self.job], self.now))

    def test_recovery_after_stale_heartbeat(self):
        self.run["created_at"] = (self.now - timedelta(hours=1)).isoformat()
        self.assertFalse(watchdog.heartbeat_valid(self.run, [self.job], self.now))
        self.run["created_at"] = self.now.isoformat()
        self.assertTrue(watchdog.heartbeat_valid(self.run, [self.job], self.now))


if __name__ == "__main__":
    unittest.main()
