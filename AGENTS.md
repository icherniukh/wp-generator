# Agent Instructions

## Beads

This repo uses `bd` for issue tracking.

- Start with `bd ready` and `bd list --status=in_progress`.
- Claim work with `bd update <id> --claim`.
- Create new persistent tasks with `bd create`.
- Close finished work with `bd close <id>`.
- Use `bd remember` for durable project notes.

Do not use markdown TODO files or duplicate the same task outside beads.

## Shell

Use non-interactive forms for file operations:

```bash
cp -f source dest
mv -f source dest
rm -f file
rm -rf directory
cp -rf source dest
```

Avoid commands that wait for confirmation prompts.
