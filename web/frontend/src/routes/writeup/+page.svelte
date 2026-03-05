<script lang="ts">
</script>

<svelte:head><title>CheckersRL — Writeup</title></svelte:head>

<div class="max-w-2xl flex flex-col gap-8">

	<div>
		<h1 class="text-2xl font-semibold text-gray-900">Project Writeup</h1>
		<p class="text-gray-500 mt-2 text-sm">Currently in progress...</p>
	</div>

	{#each [
		['1', 'Section 1'],
		['2', 'Section 2'],

	] as [title, desc]}
		<div class="border-t border-gray-100 pt-5">
			<h2 class="font-medium text-gray-800">{title}</h2>
			<p class="text-sm text-gray-400 mt-1">{desc}</p>
		</div>
	{/each}

</div>
