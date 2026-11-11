# Docs: Secrets

> Purpose: the exact name of every secret this project needs in the GitHub repository,
> what it is for, and who sets it. The assistant never sets a secret; the user does.

How to set: repository -> Settings -> Secrets and variables -> Actions -> New
repository secret. The name must match this file character for character.

---

## Currently required

None. v1 needs no external API keys: benchmarks use tiny public/committed datasets and
the tests run on GitHub-hosted runners.

## Planned, only if the step that needs it is approved

| Name | Needed by | What it is for | Notes |
|---|---|---|---|
| `ARCHX_HF_TOKEN` | Phase 9 benchmarks, if a gated dataset or model is used | read access to Hugging Face Hub | not needed for the char-level and MNIST-sized CPU benchmarks |
| `ARCHX_RELEASE_PYPI_TOKEN` | future PyPI publishing | uploads the package | not needed until the first release |

Add a row here *before* a workflow references it, never after.
