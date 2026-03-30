# === Dev install (symlink — changes to source take effect on reload) ===
ln -s /home-nfs/whc/projects/test_vscode_debug_multiproces_python/vscode_extension_debug_continue_all ~/.vscode-server/extensions/debug-continue-all

# === Full install (copy — standalone, survives if source dir moves) ===
cp -r /home-nfs/whc/projects/test_vscode_debug_multiproces_python/vscode_extension_debug_continue_all ~/.vscode-server/extensions/debug-continue-all

# === Uninstall ===
rm -rf ~/.vscode-server/extensions/debug-continue-all

# After any install/uninstall: reload VSCode
# Cmd+Shift+P → "Developer: Reload Window"

# === Usage ===
# Cmd+Shift+C (mac) / Ctrl+Shift+C (linux) → continue all debug sessions
# Or: Cmd+Shift+P → "Debug: Continue All Sessions"
