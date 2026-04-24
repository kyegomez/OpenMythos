"""
Tests for CLI System
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from open_mythos.cli.formatter import Formatter, Color
from open_mythos.cli.agent_loop import AIAgentLoop, AgentConfig, LoopState, TokenBudget
from open_mythos.cli.provider import (
    ProviderConfig, LLMProvider, OpenAIProvider,
    create_provider
)


class TestFormatter:
    """Tests for Formatter."""
    
    def setup_method(self):
        self.f = Formatter(use_colors=False)  # Disable colors for testing
    
    def test_text_coloring(self):
        """Test text coloring."""
        result = self.f.red("error")
        assert "error" in result
        
        result = self.f.green("success")
        assert "success" in result
    
    def test_bold(self):
        """Test bold text."""
        result = self.f.bold_text("title")
        assert "title" in result
    
    def test_error_message(self):
        """Test error formatting."""
        result = self.f.error("Something went wrong")
        assert "ERROR" in result
        assert "Something went wrong" in result
    
    def test_success_message(self):
        """Test success formatting."""
        result = self.f.success("Done")
        assert "OK" in result
        assert "Done" in result
    
    def test_header_level1(self):
        """Test level 1 header."""
        result = self.f.header("Title", level=1)
        assert "Title" in result
        assert "=" in result
    
    def test_header_level2(self):
        """Test level 2 header."""
        result = self.f.header("Subtitle", level=2)
        assert "Subtitle" in result
        assert "-" in result
    
    def test_bullet(self):
        """Test bullet formatting."""
        result = self.f.bullet("Item")
        assert "Item" in result
        assert "*" in result
    
    def test_muted(self):
        """Test muted text."""
        result = self.f.muted("dimmed")
        assert "dimmed" in result
    
    def test_italic(self):
        """Test italic text."""
        result = self.f.italic("text")
        assert "text" in result
    
    def test_underline(self):
        """Test underline."""
        result = self.f.underline("underlined")
        assert "underlined" in result
    
    def test_print_table(self, capsys):
        """Test table printing."""
        rows = [["Name", "Value"], ["foo", "bar"]]
        self.f.print_table(rows)
        
        captured = capsys.readouterr()
        assert "|" in captured.out
    
    def test_render_markdown_headers(self):
        """Test markdown header rendering."""
        result = self.f.render_markdown("# Header")
        assert "Header" in result
        
        result = self.f.render_markdown("## Subheader")
        assert "Subheader" in result
    
    def test_render_markdown_bold(self):
        """Test markdown bold rendering."""
        result = self.f.render_markdown("**bold**")
        assert "bold" in result
    
    def test_render_markdown_code(self):
        """Test markdown code rendering."""
        result = self.f.render_markdown("")
        assert "code" in result


class TestTokenBudget:
    """Tests for TokenBudget."""
    
    def test_initialization(self):
        """Test creating a budget."""
        budget = TokenBudget(max_tokens=1000)
        
        assert budget.max_tokens == 1000
        assert budget.remaining == 1000
        assert budget.prompt_tokens == 0
    
    def test_consume(self):
        """Test consuming tokens."""
        budget = TokenBudget(max_tokens=1000)
        
        budget.consume(prompt=100, completion=50)
        
        assert budget.prompt_tokens == 100
        assert budget.completion_tokens == 50
        assert budget.remaining == 850
    
    def test_reset(self):
        """Test resetting budget."""
        budget = TokenBudget(max_tokens=1000)
        
        budget.consume(100, 50)
        budget.reset()
        
        assert budget.prompt_tokens == 0
        assert budget.completion_tokens == 0
        assert budget.remaining == 1000


class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_defaults(self):
        """Test default configuration."""
        config = AgentConfig()
        
        assert config.max_iterations == 100
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.stream == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            max_iterations=50,
            max_retries=5,
            compress_on_pressure=0.9
        )
        
        assert config.max_iterations == 50
        assert config.max_retries == 5
        assert config.compress_on_pressure == 0.9


class TestProviderConfig:
    """Tests for ProviderConfig."""
    
    def test_defaults(self):
        """Test default provider config."""
        config = ProviderConfig(model="gpt-4")
        
        assert config.model == "gpt-4"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
    
    def test_custom_config(self):
        """Test custom provider config."""
        config = ProviderConfig(
            model="claude-3",
            api_key="test-key",
            base_url="https://api.anthropic.com",
            temperature=0.5
        )
        
        assert config.model == "claude-3"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5


class TestCreateProvider:
    """Tests for provider factory."""
    
    def test_create_openai(self):
        """Test creating OpenAI provider."""
        config = ProviderConfig(model="gpt-4", api_key="test")
        provider = create_provider("openai", config)
        
        assert isinstance(provider, OpenAIProvider)
    
    def test_create_unknown(self):
        """Test creating unknown provider raises error."""
        config = ProviderConfig(model="unknown")
        
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("unknown", config)


class TestAIAgentLoop:
    """Tests for AIAgentLoop."""
    
    def test_initialization(self):
        """Test creating agent loop."""
        mock_provider = Mock()
        agent = AIAgentLoop(provider=mock_provider)
        
        assert agent.state == LoopState.IDLE
        assert agent.iteration == 0
    
    def test_add_message(self):
        """Test adding messages."""
        mock_provider = Mock()
        agent = AIAgentLoop(provider=mock_provider)
        
        agent.add_message("user", "Hello")
        
        assert len(agent._messages) == 1
        assert agent._messages[0]["role"] == "user"
    
    def test_add_system_message(self):
        """Test adding system message."""
        mock_provider = Mock()
        agent = AIAgentLoop(provider=mock_provider)
        
        agent.add_system_message("You are helpful.")
        
        assert len(agent._messages) == 1
        assert agent._messages[0]["role"] == "system"
    
    def test_reset(self):
        """Test resetting agent."""
        mock_provider = Mock()
        agent = AIAgentLoop(provider=mock_provider)
        
        agent.add_message("user", "Hello")
        agent.reset()
        
        assert len(agent._messages) == 0
        assert agent.state == LoopState.IDLE
    
    def test_get_messages(self):
        """Test getting messages."""
        mock_provider = Mock()
        agent = AIAgentLoop(provider=mock_provider)
        
        agent.add_message("user", "Hello")
        agent.add_message("assistant", "Hi")
        
        messages = agent.get_messages()
        
        assert len(messages) == 2
    
    def test_submit_tool_result(self):
        """Test submitting tool result."""
        mock_provider = Mock()
        agent = AIAgentLoop(provider=mock_provider)
        
        # Add a pending tool call
        agent._pending_tool_calls["call_123"] = {
            "name": "test_tool",
            "arguments": {}
        }
        
        agent.submit_tool_result("call_123", "tool output")
        
        assert "call_123" not in agent._pending_tool_calls
        # Should have added tool result message
        assert any(m.get("role") == "tool" for m in agent._messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
