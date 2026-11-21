// Reads what the screens can SAY, where the eye cannot: every string and every run of markup text
// in `web/src`, off the parsed source, so a comment or a docstring is never read. A reader at a
// desk starts no service, so none of them names a flag, a variable, a route or an install.
//
//   node web/scripts/words_check.mjs        # exits 1 on a miss

import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { check, done } from './check.mjs';

const SRC = fileURLToPath(new URL('../src/', import.meta.url));

/** What no screen says, each by what it is. */
const OPERATOR = [
  ['a command-line flag', /(?<![\w-])--[a-z][\w-]*/g],
  ['an environment variable', /\bDV_[A-Z]\w*/g],
  ['a route', /\b(GET|POST|PUT|PATCH|DELETE) \/\S*/g],
  ['an install', /\bpip install\b/g],
];

/** The properties the stylesheet declares, which a style names with two dashes as a flag is. */
const STYLED = new Set(readFileSync(`${SRC}app.css`, 'utf8').match(/--[a-z][\w-]*(?=\s*:)/g));

const TEXT = new Set([ts.SyntaxKind.StringLiteral, ts.SyntaxKind.NoSubstitutionTemplateLiteral,
                      ts.SyntaxKind.TemplateHead, ts.SyntaxKind.TemplateMiddle,
                      ts.SyntaxKind.TemplateTail, ts.SyntaxKind.JsxText]);

/** Every operator's word in the strings and markup text of `file`, as `file:line what: text`. */
function said(file) {
  const source = ts.createSourceFile(file, readFileSync(SRC + file, 'utf8'), ts.ScriptTarget.Latest,
                                     true, ts.ScriptKind.TSX);
  const found = [];
  const walk = (node) => {
    if (TEXT.has(node.kind)) {
      const { line } = source.getLineAndCharacterOfPosition(node.getStart());
      for (const [what, shape] of OPERATOR) {
        for (const hit of node.text.match(shape) ?? []) {
          if (!STYLED.has(hit)) found.push(`${file}:${line + 1} ${what}: ${hit}`);
        }
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  return found;
}

const files = readdirSync(SRC, { recursive: true }).map((name) => name.replaceAll('\\', '/'))
  .filter((name) => /\.tsx?$/.test(name) && !name.endsWith('.d.ts'));

check('no string or markup text a screen can show names a command-line flag, an environment '
      + 'variable, a route or an install, in any source file',
      "a flag put back into a hint ('start the service with --book'); a route put into one "
      + "('POST /book/open refused'); or the walk reading no markup text, which lets either "
      + 'stand between two tags',
      [files.length > 30, files.flatMap(said)], [true, []]);

done();
