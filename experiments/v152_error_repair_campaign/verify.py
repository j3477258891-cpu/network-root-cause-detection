"""Run only V152 tests and save a reproducible implementation-verification report."""
from pathlib import Path
import io
import sys
import unittest

from core import BASE, EXP, now, sha, write


def main():
    folder=Path(__file__).resolve().parent
    protected=[BASE, EXP/'v149_online_calibrated_addition_campaign/online_scores.json',
               EXP/'v150_adaptive_addition_campaign/online_scores.json']
    if (folder/'online_scores.json').exists(): protected.append(folder/'online_scores.json')
    before={str(p):sha(p) for p in protected}
    output=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(folder),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    unchanged=all(sha(p)==value for p,value in before.items())
    success=result.wasSuccessful() and unchanged
    report={'created_at':now(),'source_class':'local_implementation_tests_not_public_gain',
        'tests_run':result.testsRun,'passed':success,'failures':[(str(t),e) for t,e in result.failures],
        'errors':[(str(t),e) for t,e in result.errors], 'protected_files_unchanged':unchanged,
        'protected_sha256':before,'code_sha256':{p.name:sha(p) for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix in ('.py','.mjs')},
        'test_log':output.getvalue(),
        'warning':'Synthetic outcomes exist only in temporary test directories. Passing these tests is not evidence of leaderboard improvement.'}
    write(folder/'verification_report.json',report)
    print(f'Tests: {result.testsRun}; passed: {success}; protected files unchanged: {unchanged}')
    print(f'Report: {folder / "verification_report.json"}')
    if not success: print(output.getvalue())
    return 0 if success else 1


if __name__=='__main__': sys.exit(main())
