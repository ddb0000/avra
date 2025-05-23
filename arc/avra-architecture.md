[user] ⇄ [interface layer] ⇄ [core loop]
                                   ⇅
                           [memory + mind db]
                                   ⇅
                         [LLM / local + remote]
                                   ⇅
                             [shell agent]

🧭 yolo/autonomous mode

    persistent daemon

    runs routines, logs data, self-improves

    optionally schedules tasks, auto updates itself

    can talk to other agents, APIs, tools

🧩 future integrations

    obsidian sync

    audio interface (vosk + tts)

    robotics control interface

    API mode (serve as backend to other bots/devices)


---

code executor: sandboxed python exec with stdout/stderr capture, classic but effective.

feedback loop: ai replies, generates code, local executes, gets feedback — then loops again. this is key. most “agent” bs out there doesn’t do this right.

memory groundwork: logs, responses, could easily pipe into sqlite/jsonl/vector store for long/short term.


memory:
persistent store — sqlite with:

    dialogs (timestamp, user, avra, context)

    events (cmds run, results, errors)

    notes (diary, reflections)
    attach this to the chat_session object as memory context.

agentic shell:
move toward:

    subprocess isolation

    timeouts

    sandboxing via firejail, docker, or pyee + asyncio wrappers
    this lets you build trust in autonomous mode.

autonomous routines:
let avra wake, check files, note a thought, remind you, do cron stuff.
even better? event listeners. e.g., when file changes, avra reacts.

long session threading:
add ability to load/save chat sessions.
this way you build a “relationship” with avra.