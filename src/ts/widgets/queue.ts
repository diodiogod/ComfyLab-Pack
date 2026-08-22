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
interface IQueueCancellationController {
	markCancellation: () => void
	drainPendingContinuations: () => Promise<void>
}

const cancellationControllers = new Set<IQueueCancellationController>()
let cancellationHooksInstalled = false

function installCancellationHooks() {
	if (cancellationHooksInstalled) return
	cancellationHooksInstalled = true

	const originalApiInterrupt = api.interrupt
	api.interrupt = async function (...args) {
		const controllers = [...cancellationControllers]
		for (const controller of controllers) controller.markCancellation()
		// Remove queued continuations before interrupting the active prompt. If we
		// interrupt first, ComfyUI may promote the next continuation to running
		// before it can be deleted.
		await Promise.allSettled(
			controllers.map((controller) => controller.drainPendingContinuations()),
		)
		await originalApiInterrupt.apply(this, args)
		await Promise.allSettled(
			controllers.map((controller) => controller.drainPendingContinuations()),
		)
	}

	// Older bundled frontend typings omit this event even though the backend
	// emits it and current ComfyUI handles it.
	;(api as unknown as EventTarget).addEventListener('execution_interrupted', () => {
		const controllers = [...cancellationControllers]
		for (const controller of controllers) controller.markCancellation()
		void Promise.allSettled(
			controllers.map((controller) => controller.drainPendingContinuations()),
		)
	})
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
	installCancellationHooks()
	let hasError = false
	let cancellationRequested = false
	let automaticQueueing = false
	let queuedThrough = 0
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

	const drainPendingContinuations = async () => {
		// A continuation POST can already be in flight when the user clicks X.
		// Re-scan briefly so a prompt arriving just after the interrupt is
		// removed before it can become the next running job.
		for (const delay of [0, 50, 150]) {
			if (delay > 0) {
				await new Promise<void>((resolve) => window.setTimeout(resolve, delay))
			}
			await cancelPendingContinuations()
		}
	}

	const markCancellation = () => {
		cancellationRequested = true
		widget.label = LABEL_INTERRUPT
		reset()
	}

	const cancellationController: IQueueCancellationController = {
		markCancellation,
		drainPendingContinuations,
	}
	cancellationControllers.add(cancellationController)

	const originalOnRemoved = node.onRemoved
	node.onRemoved = function () {
		cancellationControllers.delete(cancellationController)
		originalOnRemoved?.apply(this)
	}

	// handle page refresh: onConfigure is called after onNodeCreated, after applying serialized values, so we force reset here
	const originalOnConfigure = node.onConfigure
	node.onConfigure = function () {
		originalOnConfigure?.apply(this)
		reset()
	}

	const originalOnExecuted = node.onExecuted
	const queueContinuations = async (from: number, total: number) => {
		if (cancellationRequested) return
		const sourceGraph = node.graph
		if (!sourceGraph) {
			throw new Error('XY Plot Queue is not attached to a workflow graph')
		}
		automaticQueueing = true
		try {
			// Serialize every remaining cell from this node's own graph before
			// submitting any of them. app.queuePrompt() serializes the currently
			// active tab, which can change while an XY plot is running.
			const continuations = []
			for (let index = from; index < total && !cancellationRequested; index++) {
				widget.value = index
				const prompt = await app.graphToPrompt(sourceGraph)
				continuations.push({
					index,
					prompt,
				})
				// app.queuePrompt normally invokes these after serializing a job.
				// Preserve seed increment/randomize and other widget controls while
				// using the graph-bound low-level API instead.
				for (const graphNode of sourceGraph._nodes) {
					for (const graphWidget of graphNode.widgets ?? []) {
						;(graphWidget as typeof graphWidget & {
							afterQueued?: () => void
						}).afterQueued?.()
					}
				}
			}

			for (const continuation of continuations) {
				if (cancellationRequested) break
				await api.queuePrompt(0, continuation.prompt)
				queuedThrough = continuation.index
			}
		} finally {
			automaticQueueing = false
			if (cancellationRequested) await drainPendingContinuations()
		}
	}

	node.onExecuted = function (message: unknown) {
		if (hasError || cancellationRequested) return

		originalOnExecuted?.call(this, message)
		if (isQueueMessage(message)) {
			const data = queueMessageToData(message as IQueueMessage)
			if (data.index === 0) queuedThrough = 0
			if (!widget.total) widget.total = data.total
			if (data.index < data.total - 1) {
				widget.label = `Processing: ${data.index + 1} / ${data.total}`
				if (queuedThrough <= data.index) {
					void queueContinuations(data.index + 1, data.total)
				}
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

	widget.beforeQueued = function () {
		// A negative value means this is a new user-started queue. Automatic
		// continuations are tracked explicitly because an interrupt resets the
		// widget while an automatic queue request may still be in flight.
		if (cancellationRequested && widget.value < 0 && !automaticQueueing) {
			cancellationRequested = false
		}
		// Clicking Queue while another plot is active starts a fresh plot. Its
		// already-snapshotted continuations remain ahead of this new request.
		if (!automaticQueueing && widget.value >= 0) reset()
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
		markCancellation()
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
		void drainPendingContinuations()
	})

	return widget
}
