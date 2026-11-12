#!/usr/bin/env node
/**
 * The `archx` npm package is a signpost, not an implementation.
 *
 * `archx` is a Python tool. This package exists for exactly one situation: someone hears
 * about archx, types `npm i archx`, and gets something that installs. What it must never do
 * is look like it worked. So it prints where the real tool is, and exits non-zero, because
 * `npm i archx && archx train.py` must fail loudly rather than appear to succeed.
 *
 * One exception is deliberate: `archx --version` and `archx --help` exit 0, because those
 * are questions about *this* package and answering them with a failure would be a lie in
 * the other direction. Everything else, including a bare invocation, exits 1.
 *
 * No Python is bundled, no native bindings, no wrapper that shells out to pip. There is
 * nothing here that can drift away from the real tool.
 */

"use strict";

const PYPI = "archx-mdl";
const DOCS = "https://github.com/archnemix/Starting-A-new-thing-lets-see-wether-i-complete-it-or-not";
const VERSION = require("../package.json").version;
const NAME = require("../package.json").name;

const INSTALL = `pip install ${PYPI}`;

const BANNER = `
  ${NAME} on npm is a signpost. The real thing is a Python package.

  Install it with:

      ${INSTALL}

  Then:

      archx record  train.py --out run.archx.mdl
      archx info    run.archx.mdl
      archx replay  run.archx.mdl

  Docs:  ${DOCS}
  PyPI:  https://pypi.org/project/${PYPI}/

  This npm package contains no recorder, no CLI and no Python. It exists so that
  \`npm i ${NAME}\` lands somewhere that tells you this, instead of on nothing.
`;

function help() {
  return `Usage: archx <anything>

This is the npm signpost for archx, a Python tool. It has no functionality.

  archx --version      print this package's version
  archx --help         print this message
  archx <anything>     explain where the real tool is and exit 1

Install the real tool with:  ${INSTALL}
`;
}

function main(argv) {
  const args = argv.slice(2);
  const first = args[0];

  if (first === "--version" || first === "-v") {
    // Asked about this package, so this package answers.
    process.stdout.write(`${NAME} npm signpost ${VERSION}\n`);
    process.stdout.write(`the real archx is a Python package: ${INSTALL}\n`);
    return 0;
  }

  if (first === "--help" || first === "-h" || first === "help") {
    process.stdout.write(help());
    return 0;
  }

  // Everything else: a real attempt to use a tool that is not here.
  process.stderr.write(BANNER);
  process.stderr.write(
    "\n  Exit code 1 is deliberate: nothing was recorded, nothing was run.\n\n"
  );

  if (args.length) {
    process.stderr.write(
      `  You asked for: archx ${args.join(" ")}\n` +
        `  That is the real tool's command line, which lives in the Python package.\n\n`
    );
  }
  return 1;
}

if (require.main === module) {
  process.exitCode = main(process.argv);
}

module.exports = { main, PYPI, INSTALL };
