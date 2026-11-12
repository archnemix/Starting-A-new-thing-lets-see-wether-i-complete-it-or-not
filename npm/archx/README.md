# archx (npm) - a signpost, not the tool

**archx is a Python tool.** This npm package contains no recorder, no CLI and no Python. It
exists so that someone who types `npm i archx` lands somewhere that tells them where the real
tool is, instead of on nothing.

```console
$ npm i archx
$ npx archx --version
archx npm signpost 1.0.1
the real archx is a Python package: pip install archx-mdl
```

## The real install

```console
pip install archx-mdl
```

Then:

```console
archx record  train.py --out run.archx.mdl
archx info    run.archx.mdl
archx replay  run.archx.mdl
```

Docs: <https://github.com/archnemix/Starting-A-new-thing-lets-see-wether-i-complete-it-or-not>

## Why the exit code is 1

```console
$ npx archx train.py
  archx on npm is a signpost. The real thing is a Python package.
  ...
  Exit code 1 is deliberate: nothing was recorded, nothing was run.
$ echo $?
1
```

`npm i archx && archx train.py` must **fail loudly**. A signpost that exits 0 would let a
mistaken install look like a working one, and the first thing the user would lose is a training
run they thought was being recorded.

`archx --version` and `archx --help` do exit 0: those are questions about this package, and
answering them with a failure would be a lie in the other direction.

## What is in here

| file | what it is |
|---|---|
| `bin/archx.js` | the whole thing: prints the install command, exits 1 |
| `test/cli.test.js` | 8 tests, including "it must not grow an implementation" |

No dependencies. No `child_process`. No Python bundled and no wrapper that shells out to pip -
a signpost that tries to be a bridge is the thing that drifts.
