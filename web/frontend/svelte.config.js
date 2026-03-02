import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: '200.html',
			precompress: false,
			strict: false,
		}),
		prerender: {
			handleHttpError: ({ path, message }) => {
				// Ignore missing static assets during prerender
				if (path.startsWith('/favicon')) return;
				throw new Error(message);
			},
		},
	},
};

export default config;
