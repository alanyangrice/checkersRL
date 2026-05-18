<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';
</script>

<div class="border-b border-gray-200">
	<nav class="max-w-5xl mx-auto px-4 sm:px-6 h-12 flex items-center justify-between">
		<a href="/" class="text-sm font-medium text-gray-900">CheckersRL</a>
		<div class="flex items-center gap-4 sm:gap-6 text-sm text-gray-500">
			<a href="/" class="{$page.url.pathname === '/' ? 'text-gray-900' : 'hover:text-gray-700'}">Play</a>
			<a href="/training" class="{$page.url.pathname === '/training' ? 'text-gray-900' : 'hover:text-gray-700'}">Training</a>
		</div>
	</nav>
</div>

<main class="max-w-5xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
	<slot />
</main>
