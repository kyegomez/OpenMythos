# OpenMythos

<!-- agent-scope-rules -->
## Memory and scope

- This folder is the only active project. Do not read, apply, or write notes
  from any other project folder, and do not carry state from another terminal.
- Machine-wide facts: /Users/Shared/mounts/vault/agent-memory/global/MEMORY.md
  (read it; it covers disks, network, service ports, and known traps).
- This project's shared notes: ./.agents/MEMORY.md. Read it at the start.
  Append dated, verified facts only. Never secrets.
- Your own private memory for this project lives under ./.agents/<your-name>/
  (claude, codex, goose, opencode, copilot).
- Verified state in this folder beats any memory entry. When two notes
  disagree, report the conflict; do not blend them.
- Services on this machine run from /Users/home-server/services/<name>, not
  from the vault. launchd cannot read the vault.
