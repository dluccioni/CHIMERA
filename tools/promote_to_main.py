"""Rebuild main from development, leaving out the development-only paths.

Work is committed to development. main is the same history with the paths
listed in tools/devonly_paths.txt removed from every commit, so nothing
reachable from main contains the tests or the investigation scripts.
Commits that only touch those paths are dropped; every other commit keeps
its message, author and dates, and a commit that needed no change keeps its
hash.

The rebuild is deterministic: the same development history always gives
the same main, so running it again only appends the new commits. It will
not move main if main holds commits that the rebuild would drop (those
belong on development), and it never touches the working tree, so run it
with development checked out:

    python tools/promote_to_main.py              # rebuild main and move it
    python tools/promote_to_main.py --dry-run    # report only
    python tools/promote_to_main.py --force      # allow dropping main-only commits

Pushing is left to you. The first push after the history rewrite of
2026-11-08 needs `git push --force-with-lease origin main`.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile

PATTERNS_FILE = "tools/devonly_paths.txt"


def git(*args, inp=None, env=None, check=True):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(["git", *args], input=inp, capture_output=True, env=e)
    if check and r.returncode:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stderr.decode(errors='replace')}")
    return r.stdout


def rev(ref):
    out = git("rev-parse", "--verify", "--quiet", ref + "^{commit}", check=False).strip()
    return out.decode() or None


def load_patterns(source):
    """Patterns from the source branch's own copy of the list, so the result
    depends only on what is committed."""
    text = git("show", f"{source}:{PATTERNS_FILE}", check=False).decode()
    if not text:
        sys.exit(f"{source} has no {PATTERNS_FILE}")
    pats = [ln.strip() for ln in text.splitlines()]
    return [p for p in pats if p and not p.startswith("#")]


def dev_only(path, patterns):
    for p in patterns:
        if p.endswith("/"):
            if path.startswith(p):
                return True
        elif fnmatch.fnmatchcase(path, p):
            return True
    return False


_tree_cache: dict[str, tuple[str, list[str]]] = {}


def filtered_tree(tree, patterns):
    """(new tree id, removed paths) for a tree with the dev-only paths taken out."""
    if tree in _tree_cache:
        return _tree_cache[tree]
    entries = git("ls-tree", "-r", "-z", "--full-tree", tree).split(b"\0")
    keep, removed = [], []
    for ent in entries:
        if not ent:
            continue
        path = ent.split(b"\t", 1)[1].decode("utf-8")
        (removed if dev_only(path, patterns) else keep).append(ent)
    if not removed:
        res = (tree, [])
    else:
        fd, idx = tempfile.mkstemp(prefix="promote_index_")
        os.close(fd)
        os.unlink(idx)               # git wants to create the index itself
        env = {"GIT_INDEX_FILE": idx}
        try:
            git("update-index", "-z", "--index-info", inp=b"\0".join(keep) + b"\0", env=env)
            new = git("write-tree", env=env).decode().strip()
        finally:
            if os.path.exists(idx):
                os.unlink(idx)
        res = (new, [e.split(b"\t", 1)[1].decode("utf-8") for e in removed])
    _tree_cache[tree] = res
    return res


def parse_commit(sha):
    raw = git("cat-file", "commit", sha)
    head, _, msg = raw.partition(b"\n\n")
    hdr = {"parent": []}
    for ln in head.decode("utf-8").split("\n"):
        if ln.startswith(" "):
            continue
        key, _, val = ln.partition(" ")
        if key == "parent":
            hdr["parent"].append(val)
        else:
            hdr.setdefault(key, val)
    return hdr, msg


def ident(line):
    m = re.match(r"^(.*) <(.*)> (\d+) ([+-]\d{4})$", line)
    return m.group(1), m.group(2), f"{m.group(3)} {m.group(4)}"


def rebuild(source, patterns):
    """Map every commit of `source` to its filtered copy. Returns (tip, stats)."""
    revs = git("rev-list", "--topo-order", "--reverse", source).decode().split()
    new_of: dict[str, str | None] = {}
    tree_of: dict[str, str] = {}
    kept = reused = dropped = 0
    for sha in revs:
        hdr, msg = parse_commit(sha)
        tree, _ = filtered_tree(hdr["tree"], patterns)
        parents = []
        for p in hdr["parent"]:
            q = new_of[p]
            if q is not None and q not in parents:
                parents.append(q)
        if len(parents) == 1 and tree == tree_of[parents[0]]:
            new_of[sha] = parents[0]        # touched dev-only paths and nothing else
            dropped += 1
            continue
        if not parents and tree == git("hash-object", "-t", "tree", "--stdin", inp=b"").decode().strip():
            new_of[sha] = None              # a root holding only dev-only paths
            dropped += 1
            continue
        if tree == hdr["tree"] and parents == hdr["parent"]:
            new = sha
            reused += 1
        else:
            if "encoding" in hdr:
                print(f"warning: {sha[:7]} declares encoding {hdr['encoding']}; written as UTF-8")
            an, ae, ad = ident(hdr["author"])
            cn, ce, cd = ident(hdr["committer"])
            env = dict(GIT_AUTHOR_NAME=an, GIT_AUTHOR_EMAIL=ae, GIT_AUTHOR_DATE=ad,
                       GIT_COMMITTER_NAME=cn, GIT_COMMITTER_EMAIL=ce, GIT_COMMITTER_DATE=cd)
            args = ["commit-tree", tree]
            for p in parents:
                args += ["-p", p]
            new = git(*args, "-F", "-", inp=msg, env=env).decode().strip()
            kept += 1
        new_of[sha] = new
        tree_of[new] = tree
    tip = new_of[revs[-1]]
    return tip, dict(commits=len(revs), rewritten=kept, reused=reused, dropped=dropped)


def stale_imports(src, tip, patterns):
    """Python files kept on the target that import a module left out of it."""
    _, removed = filtered_tree(git("rev-parse", f"{src}^{{tree}}").decode().strip(), patterns)
    gone = set()
    for p in removed:
        if p.endswith(".py"):
            gone.add(os.path.splitext(os.path.basename(p))[0])
    if not gone:
        return []
    hits = []
    files = git("ls-tree", "-r", "--name-only", tip).decode().split("\n")
    pat = re.compile(r"^\s*(?:from\s+(?:scripts\.)?(\w+)\s+import|import\s+(?:scripts\.)?(\w+))", re.M)
    for f in files:
        if not f.endswith(".py"):
            continue
        text = git("show", f"{tip}:{f}").decode("utf-8", errors="replace")
        for m in pat.finditer(text):
            mod = m.group(1) or m.group(2)
            if mod in gone:
                hits.append(f"{f} imports {mod}")
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="development")
    ap.add_argument("--target", default="main")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="move the target even if it drops commits only the target has")
    args = ap.parse_args()

    os.chdir(git("rev-parse", "--show-toplevel").decode().strip())
    src = rev(args.source)
    if src is None:
        sys.exit(f"no branch {args.source}")
    head = git("symbolic-ref", "-q", "--short", "HEAD", check=False).decode().strip()
    if head == args.target and not args.dry_run:
        sys.exit(f"{args.target} is checked out; switch to {args.source} first "
                 "(the working tree is never touched)")

    patterns = load_patterns(args.source)
    tip, st = rebuild(args.source, patterns)
    if tip is None:
        sys.exit("nothing left after removing the development-only paths")
    hits = stale_imports(src, tip, patterns)
    old = rev(args.target)
    print(f"{args.source}: {st['commits']} commits -> {args.target}: "
          f"{st['reused']} unchanged, {st['rewritten']} filtered, "
          f"{st['dropped']} dropped (development-only)")
    for h in hits:
        print(f"  warning: {h}, which is development-only")
    if old == tip:
        print(f"{args.target} is already up to date at {tip[:7]}")
        return
    if old is not None:
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", old, tip]).returncode == 0
        if not anc:
            lost = git("rev-list", "--oneline", f"{tip}..{old}").decode().strip()
            print(f"{args.target} has commits the rebuild would drop:\n{lost}")
            if not args.force:
                sys.exit("not moved; commit those to development, or pass --force")
    if args.dry_run:
        print(f"dry run: {args.target} would move {old[:7] if old else '(new)'} -> {tip[:7]}")
        return
    git("update-ref", "-m", f"promote_to_main from {args.source} {src[:7]}",
        f"refs/heads/{args.target}", tip, *( [old] if old else [] ))
    print(f"{args.target}: {old[:7] if old else '(new)'} -> {tip[:7]}")


if __name__ == "__main__":
    main()
