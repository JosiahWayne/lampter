"""A modal screen that follows one job's output file.

This is the only part of the tool that holds an SSH channel open rather than making
a single round trip, which is exactly why it is user-initiated: nothing here runs
unless somebody presses the key to watch a log.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog

from ..transport import Transport, TransportError


class LogScreen(Screen[None]):
    """Streams a remote file into a scrollable, searchable log view."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape,q", "dismiss", "Close"),
        Binding("f", "toggle_follow", "Follow"),
        Binding("g,end", "scroll_end", "Bottom"),
    ]

    def __init__(
        self,
        transport: Transport,
        path: str,
        *,
        header: str = "",
        follow: bool = True,
        lines: int = 300,
    ) -> None:
        super().__init__()
        self.transport = transport
        self.path = path
        self.header = header
        self.follow = follow
        self.lines = lines

    def compose(self) -> ComposeResult:
        yield Header()
        # Wrap off: log lines carry meaning in their alignment, and a progress bar
        # that wraps is unreadable.
        yield RichLog(id="log", wrap=False, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.header or "log"
        self._describe()
        log = self.query_one(RichLog)
        self._write(log, f"# tail -n {self.lines}{' -f' if self.follow else ''} {self.path}")
        self._stream(log)

    def _describe(self) -> None:
        self.sub_title = f"{'following' if self.follow else 'read'} {self.path}"

    @work(thread=True, exclusive=True)
    def _stream(self, log: RichLog) -> None:
        """Runs off the UI thread: this blocks for as long as the log is followed.

        ``exclusive=True`` means starting a new stream (toggling follow) cancels the
        previous worker, which closes the generator and so terminates the remote
        ``tail`` rather than leaving it running.
        """
        try:
            for line in self.transport.stream_file(
                self.path, lines=self.lines, follow=self.follow
            ):
                self._write(log, line)
        except TransportError as exc:
            self._write(log, f"[error] {exc}", style="bold red")
        except Exception as exc:  # defensive: a worker must never kill the app
            self._write(log, f"[error] {type(exc).__name__}: {exc}", style="bold red")
        else:
            if not self.follow:
                self._write(log, "# end of file", style="dim")

    def _write(self, log: RichLog, line: str, style: str = "") -> None:
        """Write from either thread, always on the UI thread."""
        text = Text(line, style=style) if style else Text(line)
        try:
            self.app.call_from_thread(log.write, text)
        except Exception:  # pragma: no cover - screen closed mid-write
            # The screen can be dismissed while a line is in flight; dropping the
            # last line is preferable to a traceback on exit.
            pass

    def action_toggle_follow(self) -> None:
        self.follow = not self.follow
        self._describe()
        log = self.query_one(RichLog)
        self._write(log, f"# {'following again' if self.follow else 'stopped following'}")
        self._stream(log)

    def action_scroll_end(self) -> None:
        self.query_one(RichLog).scroll_end(animate=False)
