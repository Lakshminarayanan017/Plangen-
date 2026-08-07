"""
engine.rules — the reviewer's rule catalog.

Importing this package registers every rule module. The registry lives in
`base.RULES`; the runner is `engine.validator.BasicValidator`.

Coverage law (implementation plan §4): every observed flaw class maps to a
named rule that detects it with geometric evidence — tracked in
docs/reviewer_coverage.md, enforced by tests/test_reviewer_rules.py where
each rule must catch a synthetically planted defect.
"""

from modules.step4_generate.engine.rules import base                                  # noqa: F401
from modules.step4_generate.engine.rules import structure                             # noqa: F401
from modules.step4_generate.engine.rules import circulation                           # noqa: F401
from modules.step4_generate.engine.rules import openings                              # noqa: F401
from modules.step4_generate.engine.rules import nbc                                   # noqa: F401
from modules.step4_generate.engine.rules import light                                 # noqa: F401
from modules.step4_generate.engine.rules import freespace                             # noqa: F401
from modules.step4_generate.engine.rules import walls                                 # noqa: F401
from modules.step4_generate.engine.rules import doors                                 # noqa: F401
from modules.step4_generate.engine.rules import stairs                                # noqa: F401

RULES = base.RULES
ReviewContext = base.ReviewContext
