# internels: chat

> Folder-level README. Local-only source material: the conversations the project came
> from. This README is committed; the chat files themselves are not.

---

## Rule (Docs/Rules.md 25)

Chat exports here are **never committed or pushed**, not even inside a deep folder. They
are matched by `.gitignore` (`internels/chat/*`) and exist only in the working copy.

## Why keep them at all

They are the evidence behind `Plan/Origin_Chat.md`. When a decision is questioned ("did
the user actually ask for that?"), the raw text answers it without re-asking the user.

## What lives here

```
2026-11-10-to-11-11_origin-chat.md    first conversation: model formats -> the idea
```

Each file gets a header comment (purpose, dates, local-only status) and is otherwise
verbatim. Never edit the body; extract conclusions into `Plan/` instead.
