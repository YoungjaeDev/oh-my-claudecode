# oh-my-claudecode Installation

Wrapper plugins don't work due to Claude Code's plugin architecture limitations.
Install oh-my-claudecode directly from the source repository.

## Installation

```bash
/plugin marketplace add Yeachan-Heo/oh-my-claudecode
```

Then install the plugin:

```bash
/plugin install oh-my-claudecode@Yeachan-Heo-oh-my-claudecode
```

## Why Not Wrapper?

Claude Code's plugin system does not support recursive marketplace resolution.
When a plugin references another marketplace, the nested reference is not automatically resolved.
Plugins are copied to a cache directory, and relative paths break during this process.

For details, see: https://code.claude.com/docs/en/plugin-marketplaces
