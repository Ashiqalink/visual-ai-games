"""Locate the visual_ai engine, or explain clearly why it can't be found.

The games depend on the visual-ai-game-engine repo, which is not published to
PyPI. It is normally cloned as a sibling folder next to this repo. Without this
module a fresh clone fails with a bare ModuleNotFoundError that gives no hint
about the sibling-folder requirement.

Usage, from any game directory one level below the repo root:

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from engine_bootstrap import ensure_engine
    ensure_engine()

    from visual_ai import VisionPipeline
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR_NAME = 'visual ai game engine'
ENGINE_REPO_URL = 'https://github.com/Ashiqalink/visual-ai-game-engine.git'


def _candidate_paths():
    """Where the engine's src/ directory might live, in priority order."""
    override = os.environ.get('VISUAL_AI_ENGINE')
    if override:
        # Accept either the repo root or its src/ directory.
        yield os.path.join(override, 'src')
        yield override
    yield os.path.join(REPO_ROOT, '..', ENGINE_DIR_NAME, 'src')


def _fail(message):
    sys.stderr.write('\n' + '=' * 70 + '\n')
    sys.stderr.write('Cannot start: the visual_ai engine is unavailable.\n')
    sys.stderr.write('=' * 70 + '\n')
    sys.stderr.write(message.rstrip() + '\n')
    sys.stderr.write('=' * 70 + '\n\n')
    raise SystemExit(1)


def ensure_engine():
    """Make `visual_ai` importable, or exit with an actionable message."""
    try:
        import visual_ai  # noqa: F401
        return
    except ImportError:
        pass

    searched = []
    for path in _candidate_paths():
        path = os.path.normpath(path)
        searched.append(path)
        if not os.path.isdir(path):
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            import visual_ai  # noqa: F401
            return
        except ImportError as exc:
            # The folder exists but the package still won't load. The usual
            # cause is the compiled engine_core extension being built for a
            # different Python version or platform than the one running now.
            _fail(
                'Found the engine at:\n'
                '    {path}\n'
                'but importing it failed:\n'
                '    {exc}\n\n'
                'This is usually the compiled C++ extension (engine_core) not\n'
                'matching your interpreter. The checked-in binary is built for\n'
                'CPython 3.12 on Windows x86-64; you are running Python {ver} on\n'
                '{plat}. Rebuild it from the engine repo with:\n'
                '    pip install -e "{path_parent}"\n'
                '(that needs a C++ compiler installed).'.format(
                    path=path, exc=exc,
                    ver='{}.{}'.format(*sys.version_info[:2]),
                    plat=sys.platform,
                    path_parent=os.path.dirname(path),
                )
            )

    _fail(
        'The engine repo was not found. These games import it from a sibling\n'
        'folder; cloning this repo on its own is not enough.\n\n'
        'Looked in:\n'
        + ''.join('    {}\n'.format(p) for p in searched)
        + '\nFix it by cloning the engine next to this repo:\n'
        '    cd "{parent}"\n'
        '    git clone {url} "{name}"\n\n'
        'The layout should end up as:\n'
        '    {parent}\n'
        '    |-- {name}\n'
        '    \\-- {repo}\n\n'
        'If the engine lives somewhere else, point at it instead:\n'
        '    set VISUAL_AI_ENGINE=C:\\path\\to\\engine    (Windows)\n'
        '    export VISUAL_AI_ENGINE=/path/to/engine     (macOS/Linux)'.format(
            parent=os.path.normpath(os.path.join(REPO_ROOT, '..')),
            url=ENGINE_REPO_URL,
            name=ENGINE_DIR_NAME,
            repo=os.path.basename(REPO_ROOT),
        )
    )
