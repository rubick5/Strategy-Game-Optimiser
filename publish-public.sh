#!/bin/bash
# Regenerate the public mirror: the README, the measurement logs, and rustdoc with
# the rendered source stripped out.
#
#   ./publish-public.sh [path-to-public-repo]
#
# This repository is canonical for README.md. The mirror gets a banner prepended
# explaining that the source is private, so editing the copy by hand will be
# overwritten — edit here.

set -euo pipefail
PUB="${1:-../poke-sim-public}"
cd "$(dirname "$0")"
[ -d "$PUB/.git" ] || { echo "no git repo at $PUB"; exit 1; }

# `cargo doc` alone will not rebuild a site it thinks is current, and copying a
# partially regenerated target/doc produces a mirror missing search-index shards
# — a site that loads but whose search is quietly broken. Clean first.
cargo clean --doc
cargo doc --no-deps --release

rm -rf "$PUB/docs" "$PUB/results"
mkdir -p "$PUB/docs"
cp -R results "$PUB/results"
cp -R target/doc/. "$PUB/docs/"

# rustdoc renders every source file to HTML and indexes them. Both go, along with
# the anchors pointing at them — a stripped link is better than a 404.
rm -rf "$PUB/docs/src" "$PUB/docs/src-files.js"
python3 - "$PUB" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1]) / 'docs'
pats = [re.compile(rb'<a class="src[^"]*"[^>]*>.*?</a>', re.S),
        re.compile(rb'<a class=\\"src[^>]*?>Source</a>', re.S)]
for f in root.rglob('*'):
    if not f.is_file() or f.suffix not in ('.html', '.js'):
        continue
    b = original = f.read_bytes()
    for p in pats:
        b = p.sub(b'', b)
    if b != original:
        f.write_bytes(b)

# Refuse to publish if any rendered source survived.
left = sum(len(re.findall(rb'src/strat_optimizer', f.read_bytes()))
           for f in root.rglob('*') if f.is_file())
if left:
    sys.exit(f"ABORT: {left} references to rendered source remain")
print("source stripped, nothing left behind")
PY

touch "$PUB/docs/.nojekyll"
cat > "$PUB/docs/index.html" <<'HTML'
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>strat-optimizer documentation</title>
<meta http-equiv="refresh" content="0; url=strat_optimizer/index.html">
</head><body>
<p>Redirecting to <a href="strat_optimizer/index.html">the crate documentation</a>.</p>
</body></html>
HTML

# The banner lives here, not in the canonical README, so this repo's copy stays
# usable as the real project's front page.
python3 - "$PUB" <<'PY'
import pathlib, sys
banner = pathlib.Path('public-banner.md').read_text()
body = pathlib.Path('README.md').read_text()
(pathlib.Path(sys.argv[1]) / 'README.md').write_text(banner + body)
PY

echo
echo "mirror updated at $PUB — review and commit there:"
echo "    cd $PUB && git add -A && git commit"
