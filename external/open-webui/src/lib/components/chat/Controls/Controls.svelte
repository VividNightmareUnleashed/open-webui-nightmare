<script lang="ts">
	import { createEventDispatcher, getContext } from 'svelte';
	const dispatch = createEventDispatcher();
	const i18n = getContext('i18n');

	import XMark from '$lib/components/icons/XMark.svelte';
	import AdvancedParams from '../Settings/Advanced/AdvancedParams.svelte';
	import Valves from '$lib/components/chat/Controls/Valves.svelte';
	import FileItem from '$lib/components/common/FileItem.svelte';
	import Collapsible from '$lib/components/common/Collapsible.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import ExtendedThinking from '$lib/components/icons/ExtendedThinking.svelte';

	import { user, settings } from '$lib/stores';
	export let models = [];
	export let chatFiles = [];
	export let params = {};

	let showValves = false;
	let extendedThinkingActive = false;

	const ensureParams = () => {
		if (!params || typeof params !== 'object') {
			params = {};
		}
	};

	const toggleExtendedThinking = () => {
		ensureParams();
		const nextEffort = extendedThinkingActive ? null : 'high';
		const updatedParams = { ...(params ?? {}) };
		if (nextEffort === null) {
			delete updatedParams.reasoning_effort;
		} else {
			updatedParams.reasoning_effort = nextEffort;
		}
		params = updatedParams;
	};

	$: extendedThinkingActive = (params?.reasoning_effort ?? null) === 'high';
</script>

<div class=" dark:text-white">
	<div class=" flex items-center justify-between dark:text-gray-100 mb-2 gap-2">
		<div class=" text-lg font-medium self-center font-primary">{$i18n.t('Chat Controls')}</div>
		<div class="flex items-center gap-1.5">
			<Tooltip content={$i18n.t('Set reasoning effort to high for this chat.')} placement="top">
				<button
					type="button"
					class={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-secondary-200 border ${
						extendedThinkingActive
							? 'bg-accent-secondary-100/[18%] text-accent-secondary-000 border-accent-secondary-200/40'
							: 'bg-accent-secondary-100/[8%] text-accent-secondary-100 border-accent-secondary-200/20 hover:bg-accent-secondary-100/[12%] hover:text-accent-secondary-000'
					}`}
					on:click={toggleExtendedThinking}
					aria-pressed={extendedThinkingActive}
				>
					<ExtendedThinking
						className="size-4"
						title={extendedThinkingActive
							? $i18n.t('Extended Thinking enabled')
							: $i18n.t('Extended Thinking')}
					/>
					<span>{$i18n.t('Extended Thinking')}</span>
				</button>
			</Tooltip>
			<button
				class="self-center"
				on:click={() => {
					dispatch('close');
				}}
			>
				<XMark className="size-3.5" />
			</button>
		</div>
	</div>

	{#if $user?.role === 'admin' || ($user?.permissions.chat?.controls ?? true)}
		<div class=" dark:text-gray-200 text-sm font-primary py-0.5 px-0.5">
			{#if chatFiles.length > 0}
				<Collapsible title={$i18n.t('Files')} open={true} buttonClassName="w-full">
					<div class="flex flex-col gap-1 mt-1.5" slot="content">
						{#each chatFiles as file, fileIdx}
							<FileItem
								className="w-full"
								item={file}
								edit={true}
								url={file?.url ? file.url : null}
								name={file.name}
								type={file.type}
								size={file?.size}
								dismissible={true}
								small={true}
								on:dismiss={() => {
									// Remove the file from the chatFiles array

									chatFiles.splice(fileIdx, 1);
									chatFiles = chatFiles;
								}}
								on:click={() => {
									console.log(file);
								}}
							/>
						{/each}
					</div>
				</Collapsible>

				<hr class="my-2 border-gray-50 dark:border-gray-700/10" />
			{/if}

			{#if $user?.role === 'admin' || ($user?.permissions.chat?.valves ?? true)}
				<Collapsible bind:open={showValves} title={$i18n.t('Valves')} buttonClassName="w-full">
					<div class="text-sm" slot="content">
						<Valves show={showValves} />
					</div>
				</Collapsible>

				<hr class="my-2 border-gray-50 dark:border-gray-700/10" />
			{/if}

			{#if $user?.role === 'admin' || ($user?.permissions.chat?.system_prompt ?? true)}
				<Collapsible title={$i18n.t('System Prompt')} open={true} buttonClassName="w-full">
					<div class="" slot="content">
						<textarea
							bind:value={params.system}
							class="w-full text-xs outline-hidden resize-vertical {$settings.highContrastMode
								? 'border-2 border-gray-300 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800 p-2.5'
								: 'py-1.5 bg-transparent'}"
							rows="4"
							placeholder={$i18n.t('Enter system prompt')}
						/>
					</div>
				</Collapsible>

				<hr class="my-2 border-gray-50 dark:border-gray-700/10" />
			{/if}

			{#if $user?.role === 'admin' || ($user?.permissions.chat?.params ?? true)}
				<Collapsible title={$i18n.t('Advanced Params')} open={true} buttonClassName="w-full">
					<div class="text-sm mt-1.5" slot="content">
						<div>
							<AdvancedParams admin={$user?.role === 'admin'} custom={true} bind:params />
						</div>
					</div>
				</Collapsible>
			{/if}
		</div>
	{/if}
</div>
