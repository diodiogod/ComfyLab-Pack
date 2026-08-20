import { api } from '../../../../scripts/api.js';
const LABEL_READY = 'Ready';
const LABEL_INTERRUPT = 'Interrupted';
const cancellationControllers = new Set();
let cancellationHooksInstalled = false;
function installCancellationHooks() {
    if (cancellationHooksInstalled)
        return;
    cancellationHooksInstalled = true;
    const originalApiInterrupt = api.interrupt;
    api.interrupt = async function (...args) {
        const controllers = [...cancellationControllers];
        for (const controller of controllers)
            controller.markCancellation();
        await Promise.allSettled(controllers.map((controller) => controller.drainPendingContinuations()));
        await originalApiInterrupt.apply(this, args);
        await Promise.allSettled(controllers.map((controller) => controller.drainPendingContinuations()));
    };
    api.addEventListener('execution_interrupted', () => {
        const controllers = [...cancellationControllers];
        for (const controller of controllers)
            controller.markCancellation();
        void Promise.allSettled(controllers.map((controller) => controller.drainPendingContinuations()));
    });
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
export function QUEUE_STATUS(node, inputName, _inputData, app) {
    if (!app)
        throw new Error('QUEUE_STATUS: app is undefined');
    installCancellationHooks();
    let hasError = false;
    let cancellationRequested = false;
    let automaticQueueing = false;
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
    const isContinuationForThisNode = (item) => {
        if (typeof item !== 'object' || item === null)
            return false;
        const prompt = item.prompt;
        if (!Array.isArray(prompt) || prompt.length < 3)
            return false;
        const graph = prompt[2];
        if (typeof graph !== 'object' || graph === null)
            return false;
        const graphNode = graph[String(node.id)];
        if (typeof graphNode !== 'object' || graphNode === null)
            return false;
        const classType = graphNode.class_type;
        const thisType = node.comfyClass ?? node.type;
        return typeof classType === 'string' && classType === thisType;
    };
    const cancelPendingContinuations = async () => {
        try {
            const queue = await api.getQueue();
            for (const item of queue.Pending ?? []) {
                if (!isContinuationForThisNode(item))
                    continue;
                const prompt = item.prompt;
                if (!Array.isArray(prompt) || typeof prompt[1] !== 'string')
                    continue;
                try {
                    await api.deleteItem('queue', prompt[1]);
                }
                catch (error) {
                    console.warn('ComfyLab could not remove a pending XY plot continuation', error);
                }
            }
        }
        catch (error) {
            console.warn('ComfyLab could not inspect pending XY plot continuations', error);
        }
    };
    const drainPendingContinuations = async () => {
        for (const delay of [0, 50, 150]) {
            if (delay > 0) {
                await new Promise((resolve) => window.setTimeout(resolve, delay));
            }
            await cancelPendingContinuations();
        }
    };
    const markCancellation = () => {
        cancellationRequested = true;
        widget.label = LABEL_INTERRUPT;
        reset();
    };
    const cancellationController = {
        markCancellation,
        drainPendingContinuations,
    };
    cancellationControllers.add(cancellationController);
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        cancellationControllers.delete(cancellationController);
        originalOnRemoved?.apply(this);
    };
    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        originalOnConfigure?.apply(this);
        reset();
    };
    const originalOnExecuted = node.onExecuted;
    const queueContinuation = async () => {
        if (cancellationRequested)
            return;
        automaticQueueing = true;
        try {
            if (!cancellationRequested)
                await app.queuePrompt(0, 1);
        }
        finally {
            automaticQueueing = false;
            if (cancellationRequested)
                await drainPendingContinuations();
        }
    };
    node.onExecuted = function (message) {
        if (hasError || cancellationRequested)
            return;
        originalOnExecuted?.call(this, message);
        if (isQueueMessage(message)) {
            const data = queueMessageToData(message);
            if (!widget.total)
                widget.total = data.total;
            if (data.index < data.total - 1) {
                widget.value = data.index + 1;
                widget.label = `Processing: ${widget.value} / ${widget.total}`;
                void queueContinuation();
            }
            else {
                widget.label = `Processing: ${widget.value + 1} / ${widget.total}`;
                api.addEventListener('execution_success', () => {
                    if (cancellationRequested)
                        return;
                    widget.label = `Complete: ${widget.total} / ${widget.total}`;
                    reset();
                }, { once: true });
            }
        }
    };
    widget.beforeQueued = function () {
        if (cancellationRequested && widget.value < 0 && !automaticQueueing) {
            cancellationRequested = false;
        }
        if (hasError) {
            hasError = false;
            widget.label = LABEL_READY;
            widget.tooltip = undefined;
            reset();
        }
    };
    api.addEventListener('execution_error', (event) => {
        const error = formatExecutionError(event);
        markCancellation();
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
        void drainPendingContinuations();
    });
    return widget;
}
