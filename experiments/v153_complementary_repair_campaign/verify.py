"""Run synthetic tests plus read-only real artifact audit; retain evidence separation."""
import io
import unittest
from pathlib import Path
from bridge import HERE, c, protect_snapshot, verify_snapshot
import campaign


def main():
    snapshot = protect_snapshot()
    own_ledger = c.sha(HERE/'online_scores.json')
    loader = unittest.TestLoader()
    suite = loader.discover(str(HERE), pattern='test_*.py')
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    print(log.getvalue())
    c.require(c.sha(HERE/'online_scores.json') == own_ledger, 'Tests modified real V153 score ledger')
    protected = verify_snapshot(snapshot)
    audit = campaign.audit(HERE)
    report = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'success': result.wasSuccessful(), 'protected_files_unchanged': protected,
              'real_v153_ledger_unchanged': True, 'audit': audit, 'verified_at': c.now(),
              'test_evidence': 'synthetic temporary cases; NOT public scores'}
    c.write(HERE/'verification_report.json', report)
    c.require(result.wasSuccessful(), 'Verification tests failed')
    print(report)


if __name__ == '__main__': main()
