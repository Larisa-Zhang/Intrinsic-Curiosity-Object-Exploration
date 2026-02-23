import subprocess, sys, os, time, webbrowser

# Windows needs full path to npx
NPX = r"C:\Program Files\nodejs\npx.cmd" if os.name == "nt" else "npx"

experiments = [
    ("exp_1",    5001, 5173, 1),
    ("exp_2",   5002, 5174, 1),
    ("exp_3",     5003, 5175, 1),
    ("exp_4", 5004, 5176, 1),
]

procs = []
for exp_name, backend_port, frontend_port, seed in experiments:
    # Debug
    print(f"Python: {sys.executable}")
    print(f"server.py exists: {os.path.exists('server.py')}")
    # Start backend
    procs.append(subprocess.Popen(
        [sys.executable, "server.py"],
        env={
            **os.environ,
            "ACTIVE_EXP": exp_name,
            "PORT":        str(backend_port),
            "SEED":        str(seed),
        }
    ))



    procs.append(subprocess.Popen(
        [NPX, "vite"],
        env={
            **os.environ,
            "VITE_PORT":         str(frontend_port),
            "VITE_BACKEND_PORT": str(backend_port),
        }
    ))

    print(f"✅ {exp_name}: backend=:{backend_port}  frontend=:{frontend_port}  seed={seed}")

# Give servers a moment to start, then open browsers
time.sleep(3)
for exp_name, _, frontend_port, seed in experiments:
    url = f"http://localhost:{frontend_port}/?seed={seed}"
    webbrowser.open(url)
    print(f"🌐 Opened: {url}")

print("\n▶️  All experiments running. Ctrl+C to stop.")
try:
    for p in procs: p.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    for p in procs: p.terminate()