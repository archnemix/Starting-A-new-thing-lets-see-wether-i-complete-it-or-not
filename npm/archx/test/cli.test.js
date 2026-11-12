/**
 * Tests for the npm signpost. Run with `npm test` (node --test, no dependencies).
 *
 * These exist for one reason: a signpost that quietly starts exiting 0, or grows an
 * implementation, is worse than no package at all. Every assertion here is about the
 * package failing to look like the real tool.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert");
const { execFileSync, spawnSync } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const CLI = path.join(__dirname, "..", "bin", "archx.js");

function run(args, options = {}) {
  return spawnSync(process.execPath, [CLI, ...args], {
    encoding: "utf8",
    ...options,
  });
}

test("a bare invocation fails and says where the real tool is", () => {
  const result = run([]);
  assert.strictEqual(result.status, 1, "a signpost must not exit 0");
  assert.match(result.stderr, /is a signpost/);
  assert.match(result.stderr, /pip install archx-mdl/);
  assert.match(result.stderr, /github\.com\/archnemix/);
  assert.strictEqual(result.stdout, "", "nothing goes to stdout: there is no output");
});

test("any real command line fails, because the real tool is elsewhere", () => {
  for (const args of [
    ["train.py"],
    ["record", "train.py", "--out", "run.archx.mdl"],
    ["info", "run.archx.mdl"],
    ["replay", "run.archx.mdl"],
    ["--json"],            // takes a flag, still not an implementation
  ]) {
    const result = run(args);
    assert.strictEqual(result.status, 1, `archx ${args.join(" ")} must fail`);
    assert.match(result.stderr, /nothing was recorded, nothing was run/);
  }
});

test("a failed attempt echoes what was asked for, so the error is actionable", () => {
  const result = run(["record", "train.py"]);
  assert.match(result.stderr, /You asked for: archx record train\.py/);
});

test("--version answers about this package and exits 0", () => {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "package.json"), "utf8")
  );
  const result = run(["--version"]);
  assert.strictEqual(result.status, 0, "asking about the package is a fair question");
  assert.match(result.stdout, new RegExp(`npm signpost ${pkg.version.replace(/\./g, "\\.")}`));
  assert.match(result.stdout, /pip install archx-mdl/);
});

test("--help exits 0 and names the escape hatch", () => {
  const result = run(["--help"]);
  assert.strictEqual(result.status, 0);
  assert.match(result.stdout, /npm signpost for archx/);
  assert.match(result.stdout, /pip install archx-mdl/);
});

test("the package ships no Python, no native code, and no wrapper", () => {
  const root = path.join(__dirname, "..");
  const files = fs.readdirSync(root, { recursive: true }).map(String);
  const offenders = files.filter((f) =>
    /\.(py|pyc|node|so|dylib|dll|wasm)$/.test(f) || f.includes("node_modules")
  );
  assert.deepStrictEqual(offenders, [], "a signpost may not grow an implementation");

  const cli = fs.readFileSync(CLI, "utf8");
  assert.doesNotMatch(cli, /child_process|spawn|exec/, "it must not shell out to pip");
  assert.doesNotMatch(cli, /python/i.test(cli) ? /require\(["'].*python/ : /$^/);
});

test("the bin entry point is executable and named archx", () => {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "package.json"), "utf8")
  );
  assert.deepStrictEqual(pkg.bin, { archx: "bin/archx.js" });
  assert.strictEqual(pkg.name, "archx");
  assert.notStrictEqual(pkg.private, true, "it is meant to be published");
  fs.accessSync(path.join(__dirname, "..", pkg.bin.archx), fs.constants.X_OK);
});

test("the module can be required without running anything", () => {
  const out = execFileSync(
    process.execPath,
    ["-e", `const m = require(${JSON.stringify(CLI)}); console.log(JSON.stringify(Object.keys(m)))`],
    { encoding: "utf8" }
  );
  assert.deepStrictEqual(JSON.parse(out), ["main", "PYPI", "INSTALL"]);
});
