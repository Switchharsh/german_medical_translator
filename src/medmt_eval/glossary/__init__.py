"""Bilingual terminology used as *input* to translation, not only as a check.

The bank in ``taxonomy.clinical`` exists to score whether a term survived. This
package holds the larger, provenance-tracked glossary that gets shown to a model
before it translates. They are deliberately separate: injecting the same terms
that are later checked for would guarantee an improvement in the terminology
metric and measure nothing.
"""

from medmt_eval.glossary.bank import Glossary, GlossaryEntry

__all__ = ["Glossary", "GlossaryEntry"]
