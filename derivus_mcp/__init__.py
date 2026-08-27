"""The derivus MCP binding - a sibling package like `derivus_bloomberg`: shipped in the same
wheel, holding no logic, importing `requests` + `mcp` and never the engine (the gate in
`tests/test_mcp.py` reads the source). `DV_MCP` serves it over stdio.
"""
