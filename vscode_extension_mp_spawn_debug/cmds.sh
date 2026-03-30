# === Dev install (symlink — changes to source take effect on reload) ===
ln -s /home-nfs/whc/projects/test_vscode_debug_multiproces_python/vscode_extension_mp_spawn_debug ~/.vscode-server/extensions/mp-spawn-debug

# === Full install (copy — standalone, survives if source dir moves) ===
cp -r /home-nfs/whc/projects/test_vscode_debug_multiproces_python/vscode_extension_mp_spawn_debug ~/.vscode-server/extensions/mp-spawn-debug

# === Uninstall ===
rm -rf ~/.vscode-server/extensions/mp-spawn-debug

# After any install/uninstall: reload VSCode
# Cmd+Shift+P → "Developer: Reload Window"

# === Usage ===
# Cmd+Shift+C (mac) / Ctrl+Shift+C (linux) → continue all debug sessions
# Cmd+Shift+. (mac) / Ctrl+Shift+. (linux) → terminate all debug sessions
# Or: Cmd+Shift+P → "Debug: Continue All Sessions" / "Debug: Terminate All Sessions"
