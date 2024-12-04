Translate with LLM

This is just a very simple program to use a llama.cpp server as a translation engine. Feel free to use it :D

The command I am using to run the server:
QwQ-32B-Preview-Q4_0.gguf
/Users/<username>/llama.cpp/llama-server -c 36000 --host 0.0.0.0 --port 8080 -fa -ngl -1 -ctk f16 --mlock --path /Users/<username>/llama.cpp/examples/server/public_legacy -m /Users/<username>/llama.cpp/models/QwQ-32B-Preview-Q4_0.gguf
