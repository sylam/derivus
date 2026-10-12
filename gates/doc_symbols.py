"""Every code name the developer pages cite still exists in the code.

    python gates/doc_symbols.py                     # the developer pages, limitations, AGENTS.md
    python gates/doc_symbols.py PAGE.md ...         # the pages named

A page names code in backticks - `structures.quote`, `Solve.STRIKE_BRACKET`, a test's name,
`gates/reach.py` - and a rename or a deletion leaves the name standing in prose nobody re-reads.
This reads every inline code span outside fenced blocks and checks the ones that look like code
against an index built from the AST of the repository's Python:

  - a path ending in .py, .json or .md must be a file in the repository, or one the code spells by
    name (a file the service writes at run time, `xva.json`);
  - a dotted name whose head is a module or a class must resolve member by member - a module's
    members including its submodules' and its classes', a class's its bases' and the attributes
    its methods assign on `self` - except a price factor's name (`FXVol.USD.ZAR`), whose upper-case
    tail is data;
  - any other name must be one the code defines or reads (a module, class, function, attribute,
    variable, parameter, keyword), spells as a string - a JSON key is a string the code reads -
    names in the web client's TypeScript, or one torch, numpy or pandas defines.

A name under a heading saying it is not built (the roadmap's "Designed, not built", a page's "Open"
or "Non-goals") may not exist yet and is not checked, nor is one in `ABSENT`, which a page names to
say it does not exist. This cannot see whether a page DESCRIBES the
code correctly, only that the names it cites are there. Exit status 1 names each missing name with
its page and line.
"""

import ast
import builtins
import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: The Python the pages may cite, beside every top-level script.
SOURCES = ('derivus', 'derivus_spine', 'derivus_mcp', 'derivus_bloomberg', 'gates', 'tests',
           'excel_integration')
#: The web client, whose identifiers a page may cite too.
WEB = ('web/src', 'web/scripts')
#: Directories holding no file a page cites.
SKIP = {'.git', 'node_modules', 'artifacts', 'data', 'notebooks', 'docs', 'site', 'dist',
        '__pycache__', '.claude', '.vscode', '.idea'}
#: The module whose classes are price factor types.
FACTORS = 'riskfactors'
#: Libraries whose names a page may cite.
LIBRARIES = ('torch', 'torch.Tensor', 'numpy', 'pandas')
#: Names a page cites to say they are deliberately absent - mcp.md: no tracked `.mcp.json`.
ABSENT = {'.mcp.json'}
#: A heading over names that may not exist yet.
UNBUILT = re.compile(r'^#+ .*(not built|\bOpen\b|Non-goals)', re.IGNORECASE)
SPAN = re.compile(r'`([^`\n]+)`')
NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*(\(\))?$')
PATH = re.compile(r'^[\w./-]+\.(py|json|md)$')


def repository_files():
    """Every file under the repository root, as posix paths, the skipped directories left out."""
    found, stack = set(), [ROOT]
    while stack:
        for entry in stack.pop().iterdir():
            if entry.is_dir():
                if entry.name not in SKIP:
                    stack.append(entry)
            else:
                found.add(entry.relative_to(ROOT).as_posix())
    return found


class Index(object):
    """What the code defines: names, the members of each class and module, and its strings."""

    def __init__(self, sources, scripts=()):
        self.names, self.strings = set(dir(builtins)), set()
        self.modules, self.classes, self.contains, self.factors = {}, {}, {}, set()
        self.children = {}
        self.files = repository_files()
        self.libraries = None
        for path in scripts:
            self.names.update(re.findall(r'[A-Za-z_]\w*', path.read_text(encoding='utf-8')))
            self.names.add(path.stem)
        for path in sources:
            module = path.stem if path.stem != '__init__' else path.parent.name
            self.modules.setdefault(module, set())
            # a package contains its modules, so a member of one is reachable through it
            self.contains.setdefault(path.parent.name, set()).add(module)
            self.names.update(path.with_suffix('').parts)
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'))
            except (SyntaxError, UnicodeDecodeError):
                continue
            self.module(module, tree)

    def module(self, module, tree):
        members = self.modules[module]
        for node in tree.body:
            members.update(self.bound(node))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.klass(node)
                self.contains.setdefault(module, set()).add(node.name)
                if module == FACTORS:
                    self.factors.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.names.add(node.name)
            elif isinstance(node, ast.arg):
                self.names.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                self.names.add(node.arg)
            elif isinstance(node, ast.Attribute):
                self.names.add(node.attr)
            elif isinstance(node, ast.Name):
                self.names.add(node.id)
            elif isinstance(node, ast.alias):
                self.names.update((node.asname or node.name).split('.'))
                self.names.update(node.name.split('.'))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.strings.update(re.findall(r'[A-Za-z_][\w.-]*', node.value))

    @staticmethod
    def bound(node):
        """The names a module- or class-level statement binds."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [node.name]
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            return [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return [(a.asname or a.name).split('.')[0] for a in node.names]
        return []

    def klass(self, node):
        entry = self.classes.setdefault(node.name, {'members': set(), 'bases': set()})
        bases = {b.id if isinstance(b, ast.Name) else b.attr for b in node.bases
                 if isinstance(b, (ast.Name, ast.Attribute))}
        entry['bases'].update(bases)
        for base in bases:
            self.children.setdefault(base, set()).add(node.name)
        for item in node.body:
            entry['members'].update(self.bound(item))
        for item in ast.walk(node):
            if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) \
                    and item.value.id in ('self', 'cls') and isinstance(item.ctx, ast.Store):
                entry['members'].add(item.attr)
            # a declared field is the class's: `Calculation.Base_Date`, `...Parameters.Objective`
            elif isinstance(item, ast.Call) and getattr(item.func, 'id', None) == 'F' and \
                    item.args and isinstance(item.args[0], ast.Constant):
                entry['members'].add(item.args[0].value)

    def member(self, owner, name, seen=()):
        """Whether `name` is a member of the module or class `owner` - for a class, of its bases or
        of a subclass, a protocol method (`StochasticProcess.generate`) living on the latter."""
        if owner in seen:
            return False
        seen = seen + (owner,)
        if owner in self.modules:
            if name in self.modules[owner] or any(
                    name == inner or self.member(inner, name, seen)
                    for inner in self.contains.get(owner, ())):
                return True
        entry = self.classes.get(owner)
        if entry is None:
            return False
        if name in entry['members'] or (name.startswith('__') and hasattr(type, name)):
            return True
        return any(self.member(base, name, seen) for base in entry['bases']) or any(
            self.member(child, name, seen) for child in self.children.get(owner, ()))

    def library(self, name):
        """Whether torch, numpy or pandas defines `name`, the libraries imported on first ask."""
        if self.libraries is None:
            self.libraries = []
            for path in LIBRARIES:
                head, _, attr = path.partition('.')
                try:
                    found = importlib.import_module(head)
                    self.libraries.append(getattr(found, attr) if attr else found)
                except ImportError:
                    pass
        return any(hasattr(owner, name) for owner in self.libraries)

    def known(self, name):
        return name in self.names or name in self.strings or self.library(name)

    def resolves(self, span):
        """Whether a code span names something the code has; `None` where it is not a code name."""
        if span in ABSENT:
            return True
        if PATH.match(span):
            path = span.lstrip('./')
            return any(f == path or f.endswith('/' + path) for f in self.files) \
                or path.rsplit('/', 1)[-1] in self.strings
        if not NAME.match(span):
            return None
        parts = span.rstrip('()').split('.')
        head = parts[0]
        if head in self.modules or head in self.classes:
            owner = head
            for at, part in enumerate(parts[1:], 1):
                if not self.member(owner, part):
                    # a factor's name: its type, then data
                    return head in self.factors and all(
                        p.upper() == p for p in parts[at:])
                owner = part
            return True
        return self.known(parts[-1])


def spans(path):
    """`(line, span, built)` for every inline code span of a page, fenced blocks skipped; `built`
    is False under a heading over names that may not exist yet."""
    fenced, built = False, True
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.lstrip().startswith('```'):
            fenced = not fenced
            continue
        if fenced:
            continue
        if line.startswith('#'):
            built = not UNBUILT.match(line)
        for match in SPAN.finditer(line):
            yield number, match.group(1).strip(), built


def main(argv):
    pages = [Path(p) for p in argv] or sorted((ROOT / 'docs_src' / 'developer').glob('*.md')) + [
        ROOT / 'docs_src' / 'limitations.md', ROOT / 'AGENTS.md']
    sources = sorted(ROOT.glob('*.py'))
    for source in SOURCES:
        sources.extend(sorted((ROOT / source).rglob('*.py')))
    scripts = [path for base in WEB for pattern in ('*.ts', '*.tsx', '*.mjs')
               for path in sorted((ROOT / base).rglob(pattern))]
    index = Index(sources, scripts)
    missing = [(page, number, span) for page in pages for number, span, built in spans(page)
               if built and index.resolves(span) is False]
    for page, number, span in missing:
        where = page.resolve()
        print('{}:{}: `{}` names nothing in the code'.format(
            where.relative_to(ROOT).as_posix() if where.is_relative_to(ROOT) else page, number,
            span))
    print('{} page(s), {} missing name(s)'.format(len(pages), len(missing)))
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
