#!/usr/bin/env python3
"""
OpenMythos CLI Entry Point

Usage:
    python mythos.py                    # Start interactive CLI
    python mythos.py --help             # Show help
    python mythos.py --version          # Show version
    python mythos.py --web              # Start web dashboard
    python mythos.py --example memory   # Run memory example
"""

import sys
import argparse


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenMythos - Enhanced Hermes-Style Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python mythos.py                    # Start interactive CLI
    python mythos.py --web              # Start web dashboard
    python mythos.py --example memory   # Run memory example
    python mythos.py --example tools    # Run tools example
    python mythos.py --example context  # Run context example
    python mythos.py --example evolution # Run evolution example
        """
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="OpenMythos 1.0.0"
    )
    
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Start web dashboard"
    )
    
    parser.add_argument(
        "--example", "-e",
        choices=["memory", "tools", "context", "evolution"],
        help="Run an example script"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    if args.example:
        run_example(args.example)
    elif args.web:
        run_dashboard()
    else:
        run_cli(debug=args.debug)


def run_example(example_name: str):
    """Run an example script."""
    examples = {
        "memory": "examples/memory_example.py",
        "tools": "examples/tools_example.py",
        "context": "examples/context_example.py",
        "evolution": "examples/evolution_example.py",
    }
    
    example_path = examples.get(example_name)
    if not example_path:
        print(f"Unknown example: {example_name}")
        sys.exit(1)
    
    try:
        with open(example_path, "r") as f:
            code = compile(f.read(), example_path, "exec")
            exec(code, {"__name__": "__main__"})
    except FileNotFoundError:
        print(f"Example file not found: {example_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running example: {e}")
        sys.exit(1)


def run_dashboard():
    """Start the web dashboard."""
    try:
        from open_mythos.web import MythosDashboard
        dashboard = MythosDashboard(host="localhost", port=8080)
        dashboard.start()
    except ImportError as e:
        print("Web dashboard requires additional dependencies:")
        print("  pip install fastapi uvicorn")
        print(f"Error: {e}")
        sys.exit(1)


def run_cli(debug: bool = False):
    """Start the interactive CLI."""
    try:
        from open_mythos.cli import MythosCLI
        
        cli = MythosCLI(debug=debug)
        cli.run()
    except ImportError as e:
        print(f"Error importing CLI: {e}")
        print("Make sure OpenMythos is properly installed.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
