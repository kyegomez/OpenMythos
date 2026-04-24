"""
Output Formatter - Beautiful CLI Output Formatting

Provides:
- ANSI color support
- Markdown rendering
- Progress spinners
- Table formatting
- ASCII art headers
"""

import os
import re
import sys
from enum import Enum


class Color(Enum):
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


class Formatter:
    """Output formatter with color and formatting support."""
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and self._supports_colors()
    
    def _supports_colors(self) -> bool:
        if not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False
        if os.name == "nt":
            return os.environ.get("TERM") != "dumb"
        return True
    
    def _c(self, color: Color) -> str:
        return color.value if self.use_colors else ""
    
    def _cc(self, *colors: Color) -> str:
        return "".join(c.value for c in colors) if self.use_colors else ""
    
    @property
    def reset(self) -> str:
        return self._c(Color.RESET)
    
    @property
    def bold(self) -> str:
        return self._c(Color.BOLD)
    
    def text(self, text: str, *colors: Color) -> str:
        return f"{self._cc(*colors)}{text}{self.reset}"
    
    def red(self, text: str) -> str:
        return self.text(text, Color.RED)
    
    def green(self, text: str) -> str:
        return self.text(text, Color.GREEN)
    
    def yellow(self, text: str) -> str:
        return self.text(text, Color.YELLOW)
    
    def blue(self, text: str) -> str:
        return self.text(text, Color.BLUE)
    
    def cyan(self, text: str) -> str:
        return self.text(text, Color.CYAN)
    
    def magenta(self, text: str) -> str:
        return self.text(text, Color.MAGENTA)
    
    def bold_text(self, text: str) -> str:
        return self.text(text, Color.BOLD)
    
    def muted(self, text: str) -> str:
        return self.text(text, Color.DIM)
    
    def error(self, text: str) -> str:
        return self.text(f"[ERROR] {text}", Color.BRIGHT_RED)
    
    def warning(self, text: str) -> str:
        return self.text(f"[WARN] {text}", Color.YELLOW)
    
    def success(self, text: str) -> str:
        return self.text(f"[OK] {text}", Color.GREEN)
    
    def info(self, text: str) -> str:
        return self.text(f"[INFO] {text}", Color.BLUE)
    
    def header(self, text: str, level: int = 1) -> str:
        if level == 1:
            line = "=" * min(len(text) + 4, 80)
            return f"\n{self.bold}{self.cyan(line)}\n{self.bold}{self.cyan(text)}\n{self.bold}{self.cyan(line)}\n"
        elif level == 2:
            return f"\n{self.bold}{self.blue(text)}\n{self.muted('-' * len(text))}\n"
        else:
            return f"{self.bold}{text}\n"
    
    def bullet(self, text: str, indent: int = 0) -> str:
        prefix = "  " * indent + "* "
        return f"{self.muted(prefix)}{text}"
    
    def render_markdown(self, text: str) -> str:
        """Simple markdown rendering to ANSI."""
        lines = text.split("\n")
        output = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                output.append("")
                continue
            
            if in_code_block:
                output.append(self.muted(f"  {line}"))
                continue
            
            if line.startswith("# "):
                output.append(self.bold(self.blue(line[2:])))
                continue
            elif line.startswith("## "):
                output.append(self.bold(line[3:]))
                continue
            elif line.startswith("### "):
                output.append(self.underline(line[4:]))
                continue
            
            if line.startswith(">"):
                output.append(self.muted(f"| {line[1:].strip()}"))
                continue
            
            if re.match(r"^[-*] ", line):
                output.append(self.bullet(line[2:]))
                continue
            
            line = re.sub(r'\*\*(.+?)\*\*', lambda m: self.bold(m.group(1)), line)
            line = re.sub(r'\*(.+?)\*', lambda m: self.italic(m.group(1)), line)
            line = re.sub(r'`(.+?)`', lambda m: self.muted(m.group(1)), line)
            
            output.append(line)
        
        return "\n".join(output)
    
    def underline(self, text: str) -> str:
        return f"\033[4m{text}\033[0m"
    
    def italic(self, text: str) -> str:
        return f"\033[3m{text}\033[0m"
    
    def print_table(self, rows):
        """Print a formatted table."""
        if not rows:
            return
        
        num_cols = len(rows[0])
        widths = [0] * num_cols
        
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        for row in rows:
            print("| " + " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)) + " |")


formatter = Formatter()
