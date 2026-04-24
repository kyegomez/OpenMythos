"""
OpenMythos - Enhanced Hermes-Style Agent System

Setup script for pip installation.
"""

from setuptools import setup, find_packages
import os

# Read README
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="open-mythos",
    version="1.0.0",
    description="Enhanced Hermes-Style Agent System with three-layer memory, context compression, and auto-evolution",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="OpenMythos Team",
    author_email="contact@openmythos.dev",
    url="https://github.com/openmythos/open-mythos",
    license="MIT",
    
    packages=find_packages(exclude=["tests", "tests.*", "docs", "*.tests"]),
    
    python_requires=">=3.10",
    
    install_requires=[
        "aiohttp>=3.9.0",
        "requests>=2.31.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
        ],
        "anthropic": [
            "anthropic>=0.18.0",
        ],
        "openai": [
            "openai>=1.12.0",
        ],
        "all": [
            "anthropic>=0.18.0",
            "openai>=1.12.0",
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "mythos=open_mythos.cli.main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Artificial Intelligence",
    ],
    
    keywords=[
        "agent",
        "ai",
        "hermes",
        "memory",
        "context",
        "evolution",
        "mcp",
        "tool",
    ],
)
