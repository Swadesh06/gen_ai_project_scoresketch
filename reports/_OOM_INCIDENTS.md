# Phase G OOM incidents

Populated by the agent when a dry-run hits the 14 GB limit and the OOM protocol triggers.

Format per incident:

```
## <exp_id>
- model: <name + size>
- estimated peak: <gb>
- observed peak (dry-run): <gb>
- batch sizes attempted: <list>
- final outcome: <ran at b=K | abandoned at b=1>
- log: logs/vram_<exp_id>.log
- date: <YYYY-MM-DD>
```

## Incidents

(none recorded; the OOM-protocol-required experiments in Phase G are G-13 Lakh
LoRA and any MusicGen-Large invocation. Dry-run logs will live at
`logs/vram_<exp_id>.log` regardless of whether they trip the limit.)
