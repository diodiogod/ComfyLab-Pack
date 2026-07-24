import { api } from '../../../../scripts/api.js';
const LABEL_READY = 'Ready';
const LABEL_INTERRUPT = 'Interrupted';
const LABEL_ERROR = 'Error detected';
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
    const widget = node.addWidget('button', inputName, 0, () => { });
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
        if (widget.label === LABEL_ERROR)
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
    api.addEventListener('execution_error', () => {
        widget.label = LABEL_ERROR;
        reset();
    });
    return widget;
}
