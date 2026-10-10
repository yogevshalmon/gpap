#!/usr/bin/env python3
"""Hold the paper to the data and the data to the paper.

    eval/artifact/check_paper_numbers.py --generated OUT/paper_numbers.json
    eval/artifact/check_paper_numbers.py --generated … --published docs/papers/…/x.json

Exit 0: every number the PDF prints is reproduced by the generator. Exit 1: at least one is not,
each named with both values. Exit 2: usage, or a file that cannot be read.

WHY THIS EXISTS. The paper's tables are typeset by hand — they drop columns, rename rows and
reorder them to fit a two-column LNCS page — so no byte comparison against a generated `.tex`
can work, and for a while nothing checked them at all. Table 1 and Table 2 print ABSOLUTE rate
geomeans that the generator did not even compute: thirty numbers in the submitted PDF whose only
provenance was a spreadsheet step nobody could repeat. This script closes that: the shapes stay
free to differ, the NUMBERS are a diff.

HOW A VALUE IS COMPARED: at its own printed precision. The PDF says `2.31e6`, which is a claim
about three significant figures, so any generated value that rounds to `2.31e6` at three
significant figures passes and anything else fails. A published value given as a JSON number
(a count — `n`, `wins`, `eligible`) must match exactly, because a count has no precision to
round to. That asymmetry is the point: `2.31e6` and `2309817.4` agree, `n = 16` and `n = 17`
never do.

READING A FAILURE. The published file changes only when the PDF does. So:
  * a failure right after editing the PDF's numbers → the transcription is wrong, or the number
    in the PDF is;
  * a failure with the published file untouched → the data moved under the paper. Find out which
    row moved before touching this file. Re-transcribing to make the check pass is how a paper
    comes to state a number no campaign produced.
"""
import argparse
import json
import math
import os
import sys


def sig_round(x, sig):
    """`x` rounded to `sig` significant figures, or None."""
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x) or x == 0:
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def sig_digits(text):
    """How many significant figures a printed number claims.

    `2.31e6` → 3, `0.94` → 2, `12899` → 5, `1.53e9` → 3. Leading zeros do not count and a
    trailing zero inside the mantissa does (`1.10e5` claims three, not two — that is the whole
    reason the PDF writes it that way)."""
    mant = text.strip().lower().split("e")[0].lstrip("+-").replace(".", "")
    stripped = mant.lstrip("0")
    return len(stripped) or 1


def agrees(published, generated):
    """(ok, detail). A string is compared at its own precision; a number exactly."""
    if generated is None:
        return False, "the generator produced nothing for it"
    if isinstance(published, str):
        try:
            want = float(published)
        except ValueError:
            return False, f"unparseable published value {published!r}"
        sig = sig_digits(published)
        got = sig_round(float(generated), sig)
        if sig_round(want, sig) == got:
            return True, ""
        return False, f"paper {published} ({sig} s.f.) vs generated {float(generated):.6g}"
    if isinstance(published, float):            # a published float: compare to its own decimals
        if abs(float(generated) - published) <= 10 ** -_decimals(published) / 2:
            return True, ""
        return False, f"paper {published} vs generated {float(generated):.6g}"
    return (published == generated, "" if published == generated
            else f"paper {published} vs generated {generated}")


def _decimals(x):
    t = repr(float(x))
    return len(t.split(".")[1]) if "." in t else 0


def _key(entry):
    """The generated row an entry checks, as a comparable tuple of its `row` fields."""
    return tuple(sorted((k, v) for k, v in entry["row"].items()))


def _rows(gen, table):
    """Generated rows of `table`, keyed the way `_key` keys a published entry."""
    out = {}
    for r in gen.get(table) or []:
        if table == "table2":
            out[(("rival", r["rival"]), ("task", r["task"]))] = r
        else:
            out[(("depth", r["depth"]), ("set", r["set"]), ("target", r["target"]))] = r
    return out


def check(pub, gen):
    """Yield one `(where, field, detail)` per disagreement."""
    for table in ("table1", "table2", "table3"):
        rows = _rows(gen, table)
        for entry in pub.get(table) or []:
            k = _key(entry)
            where = f"{table} [{entry.get('paper') or k}]"
            got = rows.get(k)
            if got is None:
                yield where, "(whole row)", ("no generated row matches "
                                             + ", ".join(f"{a}={b!r}" for a, b in k))
                continue
            for field, want in (entry.get("values") or {}).items():
                ok, detail = agrees(want, got.get(field))
                if not ok:
                    yield where, field, detail
    for block in ("text", "uniformity"):
        want_b, got_b = pub.get(block) or {}, gen.get(block) or {}
        for field, want in want_b.items():
            if field.startswith("_") or field == "params":
                continue
            if field == "arms":                 # per-sampler verdict counts
                for arm, counts in want.items():
                    got_arm = (got_b.get("arms") or {}).get(arm)
                    if got_arm is None:
                        yield f"{block} [{arm}]", "(whole arm)", "no generated arm of that name"
                        continue
                    for c, wv in counts.items():
                        ok, detail = agrees(wv, got_arm.get(c))
                        if not ok:
                            yield f"{block} [{arm}]", c, detail
                continue
            ok, detail = agrees(want, got_b.get(field))
            if not ok:
                yield block, field, detail


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_pub = os.path.join(here, "..", "..", "docs", "papers", "tacas-27",
                               "published_numbers.json")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", required=True,
                    help="paper_tables.py --out-json output")
    ap.add_argument("--published", default=os.path.normpath(default_pub),
                    help="what the submitted PDF prints (default: the TACAS'27 file)")
    args = ap.parse_args()

    for p in (args.generated, args.published):
        if not os.path.isfile(p):
            print(f"no such file: {p}", file=sys.stderr)
            return 2
    try:
        gen = json.load(open(args.generated))
        pub = json.load(open(args.published))
    except ValueError as e:
        print(f"unreadable JSON: {e}", file=sys.stderr)
        return 2

    print(f"paper     : {pub.get('paper')} ({pub.get('venue')})")
    print(f"PDF       : {pub.get('pdf')}")
    print(f"generated : {gen.get('generated')} at {gen.get('commit')}, "
          f"GPU {gen.get('gpu')}, engine {', '.join(gen.get('engine') or []) or '—'}")
    print()

    bad = list(check(pub, gen))
    checked = sum(len(e.get("values") or {}) for t in ("table1", "table2", "table3")
                  for e in (pub.get(t) or []))
    checked += sum(1 for k in (pub.get("text") or {}) if not k.startswith("_"))
    checked += sum(len(v) if k == "arms" else 1
                   for k, v in (pub.get("uniformity") or {}).items()
                   if not k.startswith("_") and k != "params")
    if not bad:
        print(f"{checked} published values, all reproduced by the generator.")
        print("\nThe paper's tables and the campaign agree.")
        return 0
    for where, field, detail in bad:
        print(f"  MISMATCH  {where}  {field}: {detail}")
    print()
    print(f"{len(bad)} of {checked} published values are NOT reproduced by the generator.")
    print("Before editing the published file: find out which row moved. A value re-transcribed "
          "to make this pass is a number the campaign never produced.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
