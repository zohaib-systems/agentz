# Run from the agentz\ project root:
#   .\run_gmail_mcp.ps1

Write-Host "[Gmail MCP] Starting SSE server on http://localhost:8001/sse ..." -ForegroundColor Cyan
uv run python mcp_servers/gmail_mcp.py
