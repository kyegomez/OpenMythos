"""
Mythos CLI - Interactive Command Line Interface

Usage:
    mythos                    # Start interactive mode
    mythos "Hello"            # Single prompt mode
    mythos --model gpt-4      # Specify model
    mythos --provider openai  # Specify provider
    
Commands:
    /exit, /quit              # Exit the CLI
    /clear                    # Clear conversation history
    /help                     # Show help
    /model                    # Show/change model
    /memory                   # Show memory status
    /stats                    # Show statistics
"""

import asyncio
import os
import sys
import readline
from typing import Optional

from .formatter import formatter
from .agent_loop import AIAgentLoop, AgentConfig, ProviderCallback
from .provider import create_provider, ProviderConfig, LLMProvider
from ..enhanced_hermes import EnhancedHermes
from ..tools import registry


class MythosCLI:
    """
    Interactive CLI for Mythos Agent.
    
    Usage:
        cli = MythosCLI()
        cli.run()
        
        # Or single prompt
        cli.run(prompt="Hello, world!")
    """
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or self._create_default_provider()
        self.hermes = EnhancedHermes()
        self.callbacks = ProviderCallback()
        self.agent = AIAgentLoop(
            provider=self.provider,
            config=AgentConfig(),
            callbacks=self.callbacks,
            enhanced_hermes=self.hermes
        )
        self._history: list = []
        self._running = False
    
    def _create_default_provider(self) -> LLMProvider:
        """Create default provider from environment."""
        provider_type = os.getenv("MYTHOS_PROVIDER", "openai")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""
        model = os.getenv("MYTHOS_MODEL", "gpt-4")
        base_url = os.getenv("MYTHOS_BASE_URL")
        
        config = ProviderConfig(
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        
        return create_provider(provider_type, config)
    
    def run(self, prompt: Optional[str] = None) -> None:
        """
        Run the CLI.
        
        Args:
            prompt: If provided, run single prompt mode and exit
        """
        # Initialize tools
        self.hermes.initialize_tools()
        
        # Discover tools
        discovered = registry.discover_builtin_tools()
        
        if prompt:
            # Single prompt mode
            asyncio.run(self._run_single(prompt))
        else:
            # Interactive mode
            self._print_welcome()
            asyncio.run(self._run_interactive())
    
    async def _run_single(self, prompt: str) -> None:
        """Run a single prompt."""
        async for event in self.agent.run(prompt):
            self._handle_event(event)
    
    async def _run_interactive(self) -> None:
        """Run interactive loop."""
        self._running = True
        
        while self._running:
            try:
                # Get input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input(f"\n{formatter.cyan('you')}> ")
                )
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue
                
                # Run agent
                self._history.append({"role": "user", "content": user_input})
                self.hermes.add_to_context("user", user_input)
                
                print(f"\n{formatter.muted('thinking...')}")
                
                async for event in self.agent.run(user_input):
                    self._handle_event(event)
                
            except KeyboardInterrupt:
                print(f"\n{formatter.yellow('Interrupted')}")
                continue
            except EOFError:
                print(f"\n{formatter.yellow('Exiting...')}")
                break
            except Exception as e:
                print(f"\n{formatter.error(str(e))}")
        
        print(f"\n{formatter.green('Goodbye!')}")
    
    def _handle_event(self, event) -> None:
        """Handle an agent loop event."""
        if event.type == "text":
            print(event.data["content"], end="")
        
        elif event.type == "tool_call":
            tool_name = event.data["name"]
            args = event.data["arguments"]
            print(f"\n{formatter.muted(f'[Calling tool: {tool_name}]')}")
            
            # Execute tool
            result = self.hermes.execute_tool(tool_name, args)
            
            # Submit result
            self.agent.submit_tool_result(
                event.data["id"],
                result.result,
                is_error=(result.status.value != "success")
            )
            
            if result.status.value == "success":
                print(f"{formatter.green('[Tool completed]')}")
            else:
                print(f"{formatter.red(f'[Tool failed: {result.error}]')}")
        
        elif event.type == "error":
            print(f"\n{formatter.error(event.data['error'])}")
        
        elif event.type == "end":
            response = event.data.get("response")
            if response:
                self._history.append({"role": "assistant", "content": response})
                self.hermes.add_to_context("assistant", response)
    
    def _handle_command(self, cmd: str) -> None:
        """Handle a slash command."""
        parts = cmd.split()
        command = parts[0].lower()
        
        if command in ("/exit", "/quit", "/q"):
            self._running = False
        
        elif command == "/clear":
            self.agent.reset()
            self._history.clear()
            print(f"{formatter.green('Conversation cleared')}")
        
        elif command == "/help":
            self._print_help()
        
        elif command == "/model":
            if len(parts) > 1:
                new_model = parts[1]
                self.agent.provider.config.model = new_model
                print(f"{formatter.green(f'Model changed to {new_model}')}")
            else:
                print(f"Current model: {self.agent.provider.config.model}")
        
        elif command == "/memory":
            self._print_memory_status()
        
        elif command == "/stats":
            self._print_stats()
        
        elif command == "/tools":
            tools = registry.get_all_tool_names()
            print(f"Available tools ({len(tools)}):")
            for tool in tools:
                print(f"  {formatter.cyan(tool)}")
        
        else:
            print(f"{formatter.warning(f'Unknown command: {command}')}")
    
    def _print_welcome(self) -> None:
        """Print welcome message."""
        print(formatter.header("Mythos Agent", level=1))
        print(f"Type {formatter.bold('/help')} for commands")
        print(f"Provider: {formatter.cyan(self.agent.provider.config.model)}")
        print(f"Tools: {formatter.cyan(str(len(registry.get_all_tool_names())))} available")
    
    def _print_help(self) -> None:
        """Print help message."""
        print(formatter.header("Commands", level=2))
        print(f"  {formatter.bold('/exit')}, {formatter.bold('/quit')}    Exit the CLI")
        print(f"  {formatter.bold('/clear')}              Clear conversation history")
        print(f"  {formatter.bold('/help')}               Show this help")
        print(f"  {formatter.bold('/model')} [name]       Show/change model")
        print(f"  {formatter.bold('/memory')}            Show memory status")
        print(f"  {formatter.bold('/stats')}              Show statistics")
        print(f"  {formatter.bold('/tools')}              List available tools")
    
    def _print_memory_status(self) -> None:
        """Print memory status."""
        stats = self.hermes.memory.get_stats()
        print(formatter.header("Memory Status", level=2))
        print(f"  Working: {stats.get('working_entries', 0)} entries")
        print(f"  Short-term: {stats.get('short_term_entries', 0)} entries")
        print(f"  Long-term: {stats.get('long_term_entries', 0)} entries")
        print(f"  Skills: {stats.get('skills_count', 0)}")
    
    def _print_stats(self) -> None:
        """Print statistics."""
        stats = self.hermes.get_full_stats()
        print(formatter.header("Statistics", level=2))
        print(f"  Turns: {stats.get('turns', 0)}")
        print(f"  Tools: {stats['tools']['total']}")
        print(f"  Evolution tasks: {stats['evolution'].get('total_tasks', 0)}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mythos Agent CLI")
    parser.add_argument("prompt", nargs="?", help="Prompt to run")
    parser.add_argument("--model", "-m", help="Model to use")
    parser.add_argument("--provider", "-p", help="Provider to use")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    
    args = parser.parse_args()
    
    # Configure
    if args.model:
        os.environ["MYTHOS_MODEL"] = args.model
    
    if args.provider:
        os.environ["MYTHOS_PROVIDER"] = args.provider
    
    if args.no_color:
        global formatter
        from .formatter import Formatter
        formatter = Formatter(use_colors=False)
    
    # Create and run CLI
    cli = MythosCLI()
    cli.run(prompt=args.prompt)


if __name__ == "__main__":
    main()
