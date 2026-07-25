import { api } from '../../../../scripts/api.js';
const LABEL_READY = 'Ready';
const LABEL_INTERRUPT = 'Interrupted';
function formatExecutionError(event) {
    const detail = event.detail ?? {};
    const message = typeof detail.exception_message === 'string' && detail.exception_message.trim()
        ? detail.exception_message.trim()
        : 'Unknown execution error';
    const firstLine = message.split(/\r?\n/, 1)[0];
    const nodeType = detail.node_type ? String(detail.node_type) : 'node';
    const nodeId = detail.node_id !== undefined ? ` #${detail.node_id}` : '';
    const fullLabel = `Error: ${nodeType}${nodeId}: ${firstLine}`;
    const label = fullLabel.length > 120 ? `${fullLabel.slice(0, 117)}...` : fullLabel;
    const traceback = Array.isArray(detail.traceback) ? detail.traceback.join('') : '';
    return {
        detail,
        label,
        tooltip: traceback ? `${fullLabel}\n\n${traceback}` : fullLabel,
    };
}
function isQueueMessage(message) {
    const msg = message;
    return (Array.isArray(msg.index) &&
        msg.index.length === 1 &&
        typeof msg.index[0] === 'number' &&
        Array.isArray(msg.total) &&
        msg.total.length === 1 &&
        typeof msg.total[0] === 'number');
}
function queueMessageToData(message) {
    return {
        index: message.index[0],
        total: message.total[0],
    };
}
export function QUEUE_STATUS(node, inputName, _inputData, app) {
    if (!app)
        throw new Error('QUEUE_STATUS: app is undefined');
    let hasError = false;
    const widget = node.addWidget('button', inputName, 0, () => {
        if (hasError) {
            hasError = false;
            widget.label = LABEL_READY;
            widget.tooltip = undefined;
            reset();
        }
    });
    const reset = () => {
        widget.value = Math.floor(Math.random() * 10e9) * -1;
        widget.total = undefined;
    };
    reset();
    widget.label = LABEL_READY;
    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        originalOnConfigure?.apply(this);
        reset();
    };
    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        if (hasError)
            return;
        originalOnExecuted?.call(this, message);
        if (isQueueMessage(message)) {
            const data = queueMessageToData(message);
            if (!widget.total)
                widget.total = data.total;
            if (data.index < data.total - 1) {
                widget.value = data.index + 1;
                widget.label = `Processing: ${widget.value} / ${widget.total}`;
                app.queuePrompt(0, 1);
            }
            else {
                widget.label = `Processing: ${widget.value + 1} / ${widget.total}`;
                api.addEventListener('execution_success', () => {
                    widget.label = `Complete: ${widget.total} / ${widget.total}`;
                    reset();
                }, { once: true });
            }
        }
    };
    const original_api_interrupt = api.interrupt;
    api.interrupt = async function (...args) {
        await original_api_interrupt.apply(this, args);
        widget.label = LABEL_INTERRUPT;
        reset();
    };
    widget.beforeQueued = function () {
        if (hasError) {
            hasError = false;
            widget.label = LABEL_READY;
            widget.tooltip = undefined;
            reset();
        }
    };
    api.addEventListener('execution_error', (event) => {
        const error = formatExecutionError(event);
        hasError = true;
        widget.label = error.label;
        widget.tooltip = error.tooltip;
        app.extensionManager.toast.add({
            severity: 'error',
            summary: 'XY Plot generation stopped',
            detail: error.label,
            life: 10000,
        });
        console.error('ComfyLab XY Plot execution error', error.detail);
        reset();
    });
    return widget;
}
