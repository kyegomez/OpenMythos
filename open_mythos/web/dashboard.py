"""
Web Dashboard for OpenMythos

Provides a web-based monitoring and control interface.
Similar to Hermes 'hermes web' command.

Usage:
    from open_mythos.web.dashboard import MythosDashboard
    dashboard = MythosDashboard(port=8080)
    dashboard.start()
"""

import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class MemoryStats:
    """Memory statistics."""
    working_count: int = 0
    short_term_count: int = 0
    long_term_count: int = 0
    total_entries: int = 0
    skills_count: int = 0


@dataclass
class ContextStats:
    """Context statistics."""
    current_messages: int = 0
    current_tokens: int = 0
    max_tokens: int = 0
    pressure_ratio: float = 0.0
    total_compressions: int = 0


@dataclass
class ToolStats:
    """Tool statistics."""
    total_tools: int = 0
    toolsets: Dict[str, int] = None
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0

    def __post_init__(self):
        if self.toolsets is None:
            self.toolsets = {}


@dataclass
class EvolutionStats:
    """Evolution statistics."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    patterns_detected: int = 0
    skills_generated: int = 0
    reminders_active: int = 0


@dataclass
class SystemStats:
    """Overall system statistics."""
    uptime_seconds: float = 0.0
    memory: MemoryStats = None
    context: ContextStats = None
    tools: ToolStats = None
    evolution: EvolutionStats = None
    timestamp: str = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = MemoryStats()
        if self.context is None:
            self.context = ContextStats()
        if self.tools is None:
            self.tools = ToolStats()
        if self.evolution is None:
            self.evolution = EvolutionStats()
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class DashboardData:
    """Collects and stores dashboard data."""

    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self._stats = SystemStats()

    def update_stats(self, memory=None, context=None, tools=None, evolution=None):
        """Update statistics from components."""
        self._stats.uptime_seconds = time.time() - self.start_time
        self._stats.timestamp = datetime.now().isoformat()

        if memory:
            self._stats.memory = MemoryStats(
                working_count=memory.working.count(),
                short_term_count=memory.short_term.count(),
                long_term_count=memory.long_term.count(),
                total_entries=memory.get_stats()["total_entries"],
                skills_count=len(memory.long_term.list_skills()) if hasattr(memory.long_term, 'list_skills') else 0
            )

        if context:
            ctx_stats = context.get_stats()
            self._stats.context = ContextStats(
                current_messages=len(context.messages),
                current_tokens=ctx_stats.get("current_tokens", 0),
                max_tokens=ctx_stats.get("max_tokens", 0),
                pressure_ratio=ctx_stats.get("pressure_ratio", 0.0),
                total_compressions=ctx_stats.get("total_compressions", 0)
            )

        if tools:
            from open_mythos.tools import registry
            toolsets = registry.get_available_toolsets()
            self._stats.tools = ToolStats(
                total_tools=len(registry.get_all_tool_names()),
                toolsets={name: info["tool_count"] for name, info in toolsets.items()},
                total_executions=0,
                successful_executions=0,
                failed_executions=0
            )

        if evolution:
            evo_stats = evolution.get_stats()
            self._stats.evolution = EvolutionStats(
                total_tasks=evo_stats.get("total_tasks", 0),
                successful_tasks=evo_stats.get("successful_tasks", 0),
                failed_tasks=evo_stats.get("failed_tasks", 0),
                patterns_detected=len(evolution.pattern_detector.detect_approach_patterns()) if hasattr(evolution, 'pattern_detector') else 0,
                skills_generated=0,
                reminders_active=len(evolution.reminder_system.get_active_reminders()) if hasattr(evolution, 'reminder_system') else 0
            )

        self.total_requests += 1

    def get_stats(self) -> SystemStats:
        """Get current statistics."""
        return self._stats

    def get_stats_json(self) -> str:
        """Get statistics as JSON."""
        return json.dumps(asdict(self._stats), indent=2)


class MythosDashboard:
    """
    Web Dashboard for OpenMythos.

    Provides:
    - Real-time system monitoring
    - Memory layer visualization
    - Tool execution tracking
    - Evolution progress
    """

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.data = DashboardData()
        self._running = False

    def start(self):
        """Start the dashboard server."""
        try:
            from fastapi import FastAPI
            from uvicorn import Config, Server
        except ImportError:
            print("Web dashboard requires: pip install fastapi uvicorn")
            return

        app = FastAPI(title="OpenMythos Dashboard")

        @app.get("/")
        async def index():
            return {
                "name": "OpenMythos Dashboard",
                "version": "1.0.0",
                "status": "running"
            }

        @app.get("/stats")
        async def stats():
            return asdict(self.data.get_stats())

        @app.get("/stats/json")
        async def stats_json():
            return self.data.get_stats_json()

        @app.get("/memory")
        async def memory():
            stats = self.data._stats.memory
            return asdict(stats) if stats else {}

        @app.get("/context")
        async def context():
            stats = self.data._stats.context
            return asdict(stats) if stats else {}

        @app.get("/tools")
        async def tools():
            stats = self.data._stats.tools
            return asdict(stats) if stats else {}

        @app.get("/evolution")
        async def evolution():
            stats = self.data._stats.evolution
            return asdict(stats) if stats else {}

        @app.post("/refresh")
        async def refresh():
            self.data.update_stats()
            return {"status": "refreshed"}

        config = Config(app=app, host=self.host, port=self.port)
        server = Server(config)

        print(f"Starting OpenMythos Dashboard at http://{self.host}:{self.port}")
        print(f"Dashboard endpoints:")
        print(f"  - http://{self.host}:{self.port}/ - Dashboard info")
        print(f"  - http://{self.host}:{self.port}/stats - Full statistics")
        print(f"  - http://{self.host}:{self.port}/memory - Memory stats")
        print(f"  - http://{self.host}:{self.port}/context - Context stats")
        print(f"  - http://{self.host}:{self.port}/tools - Tool stats")
        print(f"  - http://{self.host}:{self.port}/evolution - Evolution stats")

        self._running = True
        server.run()

    def stop(self):
        """Stop the dashboard server."""
        self._running = False

    def is_running(self) -> bool:
        """Check if dashboard is running."""
        return self._running


# HTML Dashboard template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenMythos Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
        .card { background: #16213e; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card h2 { margin-top: 0; color: #00d4ff; font-size: 1.2rem; }
        .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #0f3460; }
        .stat:last-child { border-bottom: none; }
        .stat-label { color: #aaa; }
        .stat-value { color: #00d4ff; font-weight: bold; }
        .refresh-btn { background: #00d4ff; color: #1a1a2e; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 1rem; }
        .refresh-btn:hover { background: #00b8d4; }
        .status { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #00ff00; }
        .status.offline { background: #ff0000; }
    </style>
</head>
<body>
    <div class="container">
        <h1><span class="status" id="status"></span> OpenMythos Dashboard</h1>
        <button class="refresh-btn" onclick="refresh()">Refresh Stats</button>
        <div class="grid" id="stats-grid">
            <div class="card">
                <h2>System</h2>
                <div id="system-stats"></div>
            </div>
            <div class="card">
                <h2>Memory</h2>
                <div id="memory-stats"></div>
            </div>
            <div class="card">
                <h2>Context</h2>
                <div id="context-stats"></div>
            </div>
            <div class="card">
                <h2>Tools</h2>
                <div id="tools-stats"></div>
            </div>
            <div class="card">
                <h2>Evolution</h2>
                <div id="evolution-stats"></div>
            </div>
        </div>
    </div>
    <script>
        async function refresh() {
            const response = await fetch('/stats');
            const data = await response.json();
            
            document.getElementById('system-stats').innerHTML = formatStats([
                ['Uptime', formatUptime(data.uptime_seconds)],
                ['Last Update', data.timestamp]
            ]);
            
            document.getElementById('memory-stats').innerHTML = formatStats([
                ['Working', data.memory.working_count],
                ['Short Term', data.memory.short_term_count],
                ['Long Term', data.memory.long_term_count],
                ['Total Entries', data.memory.total_entries],
                ['Skills', data.memory.skills_count]
            ]);
            
            document.getElementById('context-stats').innerHTML = formatStats([
                ['Messages', data.context.current_messages],
                ['Tokens', data.context.current_tokens + ' / ' + data.context.max_tokens],
                ['Pressure', (data.context.pressure_ratio * 100).toFixed(1) + '%'],
                ['Compressions', data.context.total_compressions]
            ]);
            
            document.getElementById('tools-stats').innerHTML = formatStats([
                ['Total Tools', data.tools.total_tools],
                ['Toolsets', Object.keys(data.tools.toolsets).length],
                ['Executions', data.tools.total_executions]
            ]);
            
            document.getElementById('evolution-stats').innerHTML = formatStats([
                ['Total Tasks', data.evolution.total_tasks],
                ['Success Rate', data.evolution.total_tasks > 0 ? 
                    (data.evolution.successful_tasks / data.evolution.total_tasks * 100).toFixed(1) + '%' : 'N/A'],
                ['Patterns', data.evolution.patterns_detected],
                ['Active Reminders', data.evolution.reminders_active]
            ]);
            
            document.getElementById('status').className = 'status';
        }
        
        function formatStats(stats) {
            return stats.map(([label, value]) => 
                '<div class="stat"><span class="stat-label">' + label + '</span><span class="stat-value">' + value + '</span></div>'
            ).join('');
        }
        
        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return h + 'h ' + m + 'm ' + s + 's';
        }
        
        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>
"""


class HTMLDashboard:
    """
    Standalone HTML dashboard that can be served statically.
    """

    def __init__(self):
        self.template = DASHBOARD_HTML

    def get_html(self) -> str:
        """Get the HTML dashboard."""
        return self.template

    def save_html(self, path: str):
        """Save the HTML dashboard to a file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.template)
        print(f"Dashboard saved to {path}")


__all__ = [
    "MythosDashboard",
    "HTMLDashboard",
    "DashboardData",
    "SystemStats",
    "MemoryStats",
    "ContextStats",
    "ToolStats",
    "EvolutionStats",
]
