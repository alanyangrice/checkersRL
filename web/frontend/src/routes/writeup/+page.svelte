<script lang="ts">
</script>

<svelte:head><title>CheckersRL — Writeup</title></svelte:head>

<div class="max-w-2xl flex flex-col gap-8">

	<div>
		<h1 class="text-2xl font-medium text-gray-900">Project Writeup</h1>
		<p class="text-gray-500 mt-2 text-sm">A detailed breakdown of training AlphaZero and PPO agents to play checkers from scratch.</p>
	</div>

	<p class="text-sm text-gray-400 border border-gray-200 rounded px-4 py-3">
		Full writeup coming soon.
	</p>

	{#each [
		['Overview & Motivation', 'Why checkers, what the goal was, and what was interesting about the problem.'],
		['Checkers Environment', 'Building a custom gymnasium environment with mandatory captures and multi-jump chains.'],
		['AlphaZero Implementation', 'Self-play MCTS, the WDL value head, curriculum training, and the gating mechanism.'],
		['PPO & League Play', 'Reward shaping, parallel self-play, and training three specialised agents.'],
		['Results', 'How the models performed against each other and against random play.'],
		['What I Learned', 'Reflections on the training process and what I would do differently.'],
	] as [title, desc]}
		<div class="border-t border-gray-100 pt-5">
			<h2 class="font-medium text-gray-800">{title}</h2>
			<p class="text-sm text-gray-400 mt-1">{desc}</p>
		</div>
	{/each}

</div>
