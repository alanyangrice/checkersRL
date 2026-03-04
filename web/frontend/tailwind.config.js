/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {
			colors: {
				board: {
					light: '#F0D9B5',
					dark:  '#B58863',
				},
				piece: {
					blue:  '#2563EB',
					'blue-light': '#3B82F6',
					red:   '#DC2626',
					'red-light':  '#EF4444',
				},
			},
			fontFamily: {
				sans: ['Inter', 'system-ui', 'sans-serif'],
				mono: ['JetBrains Mono', 'monospace'],
			},
		},
	},
	plugins: [],
};
