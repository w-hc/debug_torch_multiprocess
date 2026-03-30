// VSCode extension API — provided by the host process at runtime, not a real npm package.
// The require('vscode') call is intercepted by VSCode's module loader.
const vscode = require('vscode');

// Track all active debug sessions so we can iterate over them later.
// VSCode doesn't expose a `debug.sessions` list, so we maintain our own
// by listening to start/terminate lifecycle events.
const activeSessions = new Set();

/**
 * Called by VSCode when the extension is activated.
 * Activation happens on the first debug session start (due to "onDebug"
 * in package.json's activationEvents).
 *
 * @param {vscode.ExtensionContext} context — disposable bag. Anything pushed
 *   into context.subscriptions is automatically cleaned up when the extension
 *   deactivates (or VSCode shuts down).
 */
function activate(context) {
    // --- Session tracking ---
    // These fire for ALL debug sessions, including child sessions created by
    // debugpy's subProcess mode. Each mp.spawn() worker becomes its own session.
    context.subscriptions.push(
        vscode.debug.onDidStartDebugSession(session => {
            activeSessions.add(session);
        })
    );
    context.subscriptions.push(
        vscode.debug.onDidTerminateDebugSession(session => {
            activeSessions.delete(session);
        })
    );

    // --- The "Continue All" command ---
    // Registered under the ID "debug.continueAll" which matches the command
    // declared in package.json's contributes.commands. Without that declaration,
    // the command wouldn't appear in the Command Palette.
    context.subscriptions.push(
        vscode.commands.registerCommand('debug.continueAll', async () => {
            // Fire continue requests to ALL sessions in parallel via Promise.all.
            // This matters for DDP: we want all ranks to resume as simultaneously
            // as possible to minimize the window where some ranks are running
            // while others are still paused (which can cause NCCL collective hangs).
            const results = await Promise.all(
                [...activeSessions].map(session =>
                    // customRequest sends a raw DAP (Debug Adapter Protocol)
                    // request to the debug adapter. "continue" is a standard DAP
                    // request that resumes execution.
                    //
                    // threadId: 1 — debugpy uses thread ID 1 for the main thread.
                    // The DAP spec requires a threadId, but debugpy actually
                    // resumes ALL threads in the process regardless of which
                    // threadId you specify.
                    //
                    // If a session is already running (not paused), this request
                    // will be rejected by the adapter — the catch returns null.
                    session.customRequest('continue', { threadId: 1 })
                        .then(() => true)
                        .catch(() => null)
                )
            );

            const resumed = results.filter(Boolean).length;
            if (resumed > 0) {
                vscode.window.setStatusBarMessage(`Continued ${resumed} session(s)`, 2000);
            }
        })
    );

    // --- The "Terminate All" command ---
    // In attach-listen mode, VSCode only offers "disconnect" which leaves the
    // process running. We use a two-stage approach from the DAP spec:
    //   1. "terminate" — asks debugpy to send SIGTERM (graceful, allows cleanup)
    //   2. "disconnect" with terminateDebuggee: true — sends SIGKILL (forceful)
    //
    // Note: debugpy#338 — debugpy may not kill all children in a process group.
    // torch.multiprocessing.spawn children could survive. If that happens,
    // manually kill the process group from the terminal.
    context.subscriptions.push(
        vscode.commands.registerCommand('debug.terminateAll', async () => {
            const sessions = [...activeSessions];
            // Stage 1: graceful terminate (SIGTERM)
            await Promise.all(
                sessions.map(session =>
                    session.customRequest('terminate', {})
                        .then(() => true)
                        .catch(() => null)
                )
            );
            // Brief pause for graceful shutdown
            await new Promise(resolve => setTimeout(resolve, 500));
            // Stage 2: forceful disconnect for any sessions still alive
            const remaining = [...activeSessions];
            if (remaining.length > 0) {
                await Promise.all(
                    remaining.map(session =>
                        session.customRequest('disconnect', { terminateDebuggee: true })
                            .then(() => true)
                            .catch(() => null)
                    )
                );
            }
            vscode.window.setStatusBarMessage(`Terminated ${sessions.length} session(s)`, 2000);
        })
    );
}

// Called when the extension is deactivated (VSCode shutdown or extension disabled).
function deactivate() {
    activeSessions.clear();
}

// VSCode expects CommonJS exports with these two names.
module.exports = { activate, deactivate };
