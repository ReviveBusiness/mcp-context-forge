#!/bin/bash

# Test REST tool via API
# Usage: ./test-tool.sh

# Source virtual environment
source ~/.venv/mcpgateway/bin/activate

# Generate token
echo "Generating authentication token..."
TOKEN=$(python -m mcpgateway.utils.create_jwt_token \
  --username admin@example.com \
  --admin \
  --exp 10080 \
  --secret my-test-key 2>/dev/null)

if [ -z "$TOKEN" ]; then
  echo "ERROR: Failed to generate token"
  exit 1
fi

echo "Token generated successfully"
echo "=========================================="
echo "Testing REST tool: test-tool"
echo "=========================================="

# Invoke the tool
curl -X POST "http://localhost:4444/mcp" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -L \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "test-tool",
      "arguments": {
        "message": "Hello from test script!"
      }
    },
    "id": 1
  }'

echo ""
echo ""
echo "=========================================="
echo "Check your webhook.site dashboard at:"
echo "https://webhook.site/0f547174-a653-4ec1-bc97-278db9f51c5f"
echo "=========================================="
