import type { LGraphNode } from '@comfyorg/litegraph'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'

import { api } from '~/.mock/scripts/api.js'
import type { InputSpec } from '~/.d.ts/comfyui-frontend-types_alt.js'

const LABEL_READY = 'Ready'
const LABEL_INTERRUPT = 'Interrupted'

interface IExecutionErrorDetail {
	node_id?: string | number
	node_type?: string
	exception_message?: string
	traceback?: string[]
}
interface IQueueData {
	index: number
	total: number
}
interface IQueueMessage {
	index: [number]
	total: [number]
}
function isQueueMessage(message: unknown): message is IQueueMessage {
	const msg = message as IQueueMessage
	return (
		Array.isArray(msg.index) &&
		msg.index.length === 1 &&
		typeof msg.index[0] === 'number' &&
		Array.isArray(msg.total) &&
		msg.total.length === 1 &&
		typeof msg.total[0] === 'number'
	)
}
function queueMessageToData(message: IQueueMessage): IQueueData {
	return {
		index: message.index[0],
		total: message.total[0],
	}
}
function formatExecutionError(event: Event) {
	const detail = (event as CustomEvent<IExecutionErrorDetail>).detail ?? {}
	const message =
		typeof detail.exception_message === 'string' && detail.exception_message.trim()
			? detail.exception_message.trim()
			: 'Unknown execution error'
	const firstLine = message.split(/\r?\n/, 1)[0]
	const nodeType = detail.node_type ? String(detail.node_type) : 'node'
	const nodeId = detail.node_id !== undefined ? ` #${detail.node_id}` : ''
	const fullLabel = `Error: ${nodeType}${nodeId}: ${firstLine}`
	const label = fullLabel.length > 120 ? `${fullLabel.slice(0, 117)}...` : fullLabel
	const traceback = Array.isArray(detail.traceback) ? detail.traceback.join('') : ''
	return {
		detail,
		label,
		tooltip: traceback ? `${fullLabel}\n\n${traceback}` : fullLabel,
	}
}

export function QUEUE_STATUS(
	node: LGraphNode,
	inputName: string,
	_inputData: InputSpec,
	app: ComfyApp | undefined, // forced to accept undefined as per ComfyWidgetConstructor
) {
	if (!app) throw new Error('QUEUE_STATUS: app is undefined')
	let hasError = false
	let cancellationRequested = false
	const widget = node.addWidget('button', inputName, 0, () => {
		if (hasError) {
			hasError = false
			widget.label = LABEL_READY
			widget.tooltip = undefined
			reset()
		}
	})

	const reset = () => {
		// set widget value to a random negative value, to ensure we can restart in any case
		widget.value = Math.floor(Math.random() * 10e9) * -1
		widget.total = undefined
	}

	// set initial value
	reset()
	widget.label = LABEL_READY

	const isContinuationForThisNode = (item: unknown) => {
		if (typeof item !== 'object' || item === null) return false
		const prompt = (item as { prompt?: unknown }).prompt
		if (!Array.isArray(prompt) || prompt.length < 3) return false
		const graph = prompt[2]
		if (typeof graph !== 'object' || graph === null) return false
		const graphNode = (graph as Record<string, unknown>)[String(node.id)]
		if (typeof graphNode !== 'object' || graphNode === null) return false
		const classType = (graphNode as { class_type?: unknown }).class_type
		const thisType = node.comfyClass ?? node.type
		return typeof classType === 'string' && classType === thisType
	}

	const cancelPendingContinuations = async () => {
		try {
			const queue = await api.getQueue()
			for (const item of queue.Pending ?? []) {
				if (!isContinuationForThisNode(item)) continue
				const prompt = (item as { prompt?: unknown }).prompt
				if (!Array.isArray(prompt) || typeof prompt[1] !== 'string') continue
				try {
					await api.deleteItem('queue', prompt[1])
				} catch (error) {
					console.warn('ComfyLab could not remove a pending XY plot continuation', error)
				}
			}
		} catch (error) {
			console.warn('ComfyLab could not inspect pending XY plot continuations', error)
		}
	}

	// handle page refresh: onConfigure is called after onNodeCreated, after applying serialized values, so we force reset here
	const originalOnConfigure = node.onConfigure
	node.onConfigure = function () {
		originalOnConfigure?.apply(this)
		reset()
	}

	const originalOnExecuted = node.onExecuted
	node.onExecuted = function (message: unknown) {
		if (hasError || cancellationRequested) return

		originalOnExecuted?.call(this, message)
		if (isQueueMessage(message)) {
			const data = queueMessageToData(message as IQueueMessage)
			if (!widget.total) widget.total = data.total
			if (data.index < data.total - 1) {
				widget.value = data.index + 1
				widget.label = `Processing: ${widget.value} / ${widget.total}`
				app.queuePrompt(0, 1)
			} else {
				widget.label = `Processing: ${widget.value + 1} / ${widget.total}`
				// wait for the execution to end before displaying the "Complete" label; note: the listener will be automatically deleted after use
				api.addEventListener(
					'execution_success',
					() => {
						if (cancellationRequested) return
						widget.label = `Complete: ${widget.total} / ${widget.total}`
						reset()
					},
					{ once: true },
				)
			}
		}
	}

	// handle API interruptions and errors
	const original_api_interrupt = api.interrupt
	api.interrupt = async function (...args) {
		cancellationRequested = true
		await cancelPendingContinuations()
		await original_api_interrupt.apply(this, args)
		widget.label = LABEL_INTERRUPT
		reset()
	}

	widget.beforeQueued = function () {
		// A negative value means this is a new user-started queue. Automatic
		// continuations set the next positive index before queueing themselves.
		if (cancellationRequested && widget.value < 0) {
			cancellationRequested = false
		}
		// An already queued continuation has run beforeQueued before an error.
		// A later user-started queue can therefore safely clear the old lock.
		if (hasError) {
			hasError = false
			widget.label = LABEL_READY
			widget.tooltip = undefined
			reset()
		}
	}
	api.addEventListener('execution_error', (event) => {
		const error = formatExecutionError(event)
		hasError = true
		widget.label = error.label
		widget.tooltip = error.tooltip
		app.extensionManager.toast.add({
			severity: 'error',
			summary: 'XY Plot generation stopped',
			detail: error.label,
			life: 10000,
		})
		console.error('ComfyLab XY Plot execution error', error.detail)
		reset()
	})

	return widget
}
