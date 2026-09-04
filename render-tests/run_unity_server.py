"""Compatibility entry point for the current Unity inference server.

The former online-evolution prototype used an obsolete observation and coupled
six-component action.  Keep this filename for existing launch scripts while
delegating to the validated 36-D, nine-component implementation.
"""

from rl_server import main


if __name__ == "__main__":
    main()
